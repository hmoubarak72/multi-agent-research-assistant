from contextlib import asynccontextmanager
from typing import Optional
import redis.asyncio as redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.writer import WriterAgent
from app.api.routes import router, schedule_research_task
from app.config.settings import Settings, get_settings
from app.graph.workflow import ResearchOrchestrator
from app.observability.logging import configure_logging, get_logger
from app.services.budget_service import BudgetService
from app.services.extraction_service import ExtractionService
from app.services.llm_service import LLMService
from app.services.redis_service import RedisPersistenceService
from app.services.search_service import SearchService
from app.services.source_service import SourceService

logger = get_logger(__name__)


async def recover_interrupted_runs(
    redis_service: RedisPersistenceService,
    orchestrator: ResearchOrchestrator,
) -> int:
    """Finds incomplete runs in Redis after a process crash/restart and resumes them."""
    incomplete_ids = await redis_service.list_incomplete_run_ids()
    recovered_count = 0
    for run_id in incomplete_ids:
        cp = await redis_service.load_checkpoint(run_id)
        if not cp:
            continue
        logger.info(
            "recovering_interrupted_run",
            run_id=run_id,
            last_stage=cp.current_stage,
            completed_nodes=len(cp.completed_node_keys),
        )
        await redis_service.emit_event(
            run_id,
            "execution_recovered",
            {
                "resumed_from_stage": cp.current_stage,
                "completed_node_keys": cp.completed_node_keys,
            },
        )
        schedule_research_task(orchestrator, redis_service, cp)
        recovered_count += 1
    return recovered_count


def create_app(
    settings_override: Optional[Settings] = None,
    redis_client_override: Optional[redis.Redis] = None,
    auto_recover_on_startup: bool = True,
) -> FastAPI:
    configure_logging()
    settings = settings_override or get_settings()

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[f"{settings.rate_limit_per_minute}/minute"],
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis_client = redis_client_override or redis.from_url(
            settings.redis_url, decode_responses=False
        )
        redis_service = RedisPersistenceService(
            redis_client=redis_client,
            ttl_seconds=settings.redis_run_ttl_seconds,
        )
        budget_service = BudgetService(settings=settings)
        source_service = SourceService(
            max_results_per_domain=settings.max_results_per_domain
        )
        extraction_service = ExtractionService(
            timeout_seconds=settings.extraction_timeout_seconds,
            max_chars=settings.max_extracted_chars,
        )
        search_service = SearchService(api_key=settings.tavily_api_key)
        llm_service = LLMService(
            api_key=settings.anthropic_api_key,
            model=settings.claude_model,
        )

        planner = PlannerAgent(llm_service=llm_service)
        researcher = ResearcherAgent(
            llm_service=llm_service,
            search_service=search_service,
            extraction_service=extraction_service,
            source_service=source_service,
            budget_service=budget_service,
        )
        supervisor = SupervisorAgent(llm_service=llm_service)
        writer = WriterAgent(
            llm_service=llm_service, source_service=source_service
        )

        orchestrator = ResearchOrchestrator(
            planner=planner,
            researcher=researcher,
            supervisor=supervisor,
            writer=writer,
            budget_service=budget_service,
            redis_service=redis_service,
        )

        app.state.settings = settings
        app.state.redis_service = redis_service
        app.state.budget_service = budget_service
        app.state.orchestrator = orchestrator

        if auto_recover_on_startup:
            try:
                count = await recover_interrupted_runs(redis_service, orchestrator)
                logger.info("startup_recovery_complete", recovered_runs=count)
            except Exception as exc:
                logger.warning("startup_recovery_skipped", reason=str(exc))

        yield

        if redis_client_override is None:
            await redis_client.aclose()

    app = FastAPI(
        title="Multi-Agent Research Assistant API",
        description=(
            "Production-grade supervisor-worker research platform with Redis checkpointing, "
            "deterministic cost/token budgets, source provenance, and step-level execution traces."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_api_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    app.include_router(router)
    return app


app = create_app()