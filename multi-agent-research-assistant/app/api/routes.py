import asyncio
import secrets
import uuid
from typing import Dict, Set
from fastapi import APIRouter, Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette.sse import EventSourceResponse
from app.api.schemas import (
    CreateResearchRequest,
    CreateResearchResponse,
    ResearchProgressSummary,
    ResearchStatusResponse,
    ResearchTraceResponse,
)
from app.config.settings import Settings, get_settings
from app.graph.workflow import ResearchOrchestrator
from app.models.research import SubQuestionStatus
from app.models.schemas import RunStatus
from app.observability.logging import get_logger
from app.services.budget_service import BudgetService
from app.services.redis_service import RedisPersistenceService, RunCheckpoint

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

_ACTIVE_TASKS: Set[asyncio.Task] = set()


def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
    settings: Settings = Depends(get_settings),
) -> str:
    if credentials is None or not secrets.compare_digest(
        credentials.credentials, settings.api_auth_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Bearer authentication token.",
        )
    return credentials.credentials


def get_redis_service(request: Request) -> RedisPersistenceService:
    return request.app.state.redis_service


def get_orchestrator(request: Request) -> ResearchOrchestrator:
    return request.app.state.orchestrator


def get_budget_service(request: Request) -> BudgetService:
    return request.app.state.budget_service


def schedule_research_task(
    orchestrator: ResearchOrchestrator,
    redis_service: RedisPersistenceService,
    checkpoint: RunCheckpoint,
) -> asyncio.Task:
    async def _runner() -> None:
        try:
            await orchestrator.execute_run(checkpoint)
        except Exception as exc:
            logger.error(
                "unhandled_orchestrator_crash",
                run_id=checkpoint.run_id,
                error=str(exc),
            )
            latest = await redis_service.load_checkpoint(checkpoint.run_id)
            if latest and latest.status not in (
                RunStatus.COMPLETED,
                RunStatus.BUDGET_EXCEEDED,
            ):
                latest.status = RunStatus.FAILED
                latest.error_message = f"Unhandled execution error: {exc}"
                await redis_service.save_checkpoint(latest)
                await redis_service.emit_event(
                    checkpoint.run_id,
                    "execution_failed",
                    {"error": latest.error_message},
                )

    task = asyncio.create_task(_runner())
    _ACTIVE_TASKS.add(task)
    task.add_done_callback(_ACTIVE_TASKS.discard)
    return task


@router.get("/health", tags=["System"])
async def health_check(
    redis_service: RedisPersistenceService = Depends(get_redis_service),
) -> Dict[str, str]:
    try:
        await redis_service.redis.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis unavailable: {exc}",
        )


@router.post(
    "/research",
    response_model=CreateResearchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Research"],
    dependencies=[Depends(verify_bearer_token)],
)
async def create_research_run(
    payload: CreateResearchRequest,
    settings: Settings = Depends(get_settings),
    redis_service: RedisPersistenceService = Depends(get_redis_service),
    orchestrator: ResearchOrchestrator = Depends(get_orchestrator),
    budget_service: BudgetService = Depends(get_budget_service),
) -> CreateResearchResponse:
    if len(payload.question.strip()) > settings.max_question_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Question exceeds maximum length of {settings.max_question_length} characters.",
        )

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    overrides = (
        payload.budget_overrides.model_dump(exclude_none=True)
        if payload.budget_overrides
        else None
    )
    budget = budget_service.initialize_budget(overrides=overrides)

    checkpoint = RunCheckpoint(
        run_id=run_id,
        original_question=payload.question.strip(),
        status=RunStatus.QUEUED,
        current_stage="initialized",
        budget=budget,
    )

    await redis_service.save_checkpoint(checkpoint)
    await redis_service.emit_event(
        run_id,
        "research_initialized",
        {"question": checkpoint.original_question},
    )

    schedule_research_task(orchestrator, redis_service, checkpoint)

    return CreateResearchResponse(
        run_id=run_id,
        status=checkpoint.status,
        created_at=checkpoint.created_at,
    )


@router.get(
    "/research/{run_id}",
    response_model=ResearchStatusResponse,
    tags=["Research"],
    dependencies=[Depends(verify_bearer_token)],
)
async def get_research_status(
    run_id: str,
    redis_service: RedisPersistenceService = Depends(get_redis_service),
) -> ResearchStatusResponse:
    cp = await redis_service.load_checkpoint(run_id)
    if not cp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    total_sq = len(cp.plan.sub_questions) if cp.plan else 0
    completed_sq = sum(
        1
        for st in cp.sub_question_statuses.values()
        if st in (SubQuestionStatus.COMPLETED, SubQuestionStatus.FAILED)
    )

    progress = ResearchProgressSummary(
        total_sub_questions=total_sq,
        completed_sub_questions=completed_sq,
        active_sub_question_id=cp.active_sub_question_id,
        sub_question_statuses=cp.sub_question_statuses,
        findings_collected=len(cp.findings),
        sources_evaluated=len(cp.sources),
    )

    return ResearchStatusResponse(
        run_id=cp.run_id,
        original_question=cp.original_question,
        status=cp.status,
        current_stage=cp.current_stage,
        created_at=cp.created_at,
        updated_at=cp.updated_at,
        progress=progress,
        plan=cp.plan,
        report=cp.report,
        budget=cp.budget,
        error_message=cp.error_message,
    )


@router.get(
    "/research/{run_id}/trace",
    response_model=ResearchTraceResponse,
    tags=["Observability"],
    dependencies=[Depends(verify_bearer_token)],
)
async def get_research_trace(
    run_id: str,
    redis_service: RedisPersistenceService = Depends(get_redis_service),
) -> ResearchTraceResponse:
    cp = await redis_service.load_checkpoint(run_id)
    if not cp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    return ResearchTraceResponse(
        run_id=cp.run_id,
        status=cp.status,
        total_steps=len(cp.trace_steps),
        total_input_tokens=cp.budget.input_tokens_used,
        total_output_tokens=cp.budget.output_tokens_used,
        total_estimated_cost_usd=cp.budget.estimated_cost_usd,
        supervisor_decisions=cp.supervisor_decisions,
        steps=cp.trace_steps,
        error_message=cp.error_message,
    )


@router.get(
    "/research/{run_id}/events",
    tags=["Streaming"],
    dependencies=[Depends(verify_bearer_token)],
)
async def stream_research_events(
    run_id: str,
    request: Request,
    redis_service: RedisPersistenceService = Depends(get_redis_service),
) -> EventSourceResponse:
    cp = await redis_service.load_checkpoint(run_id)
    if not cp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Research run '{run_id}' not found.",
        )

    async def event_generator():
        async for event in redis_service.subscribe_events(run_id):
            if await request.is_disconnected():
                break
            yield {
                "event": event.event_type,
                "data": event.model_dump_json(),
            }

    return EventSourceResponse(event_generator())
