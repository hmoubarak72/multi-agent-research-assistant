from unittest.mock import AsyncMock
import pytest
from app.agents.researcher import ResearcherExecutionResult
from app.graph.workflow import ResearchOrchestrator
from app.main import recover_interrupted_runs
from app.models.findings import FindingRecord, SourceMetadata, SourceStatus
from app.models.research import (
    ReportSection,
    ResearchPlan,
    ResearchReport,
    SubQuestion,
    SubQuestionStatus,
    SupervisorDecision,
    SupervisorDecisionType,
)
from app.models.schemas import RunStatus
from app.services.redis_service import RunCheckpoint


@pytest.mark.asyncio
async def test_crashed_run_resumes_from_redis_without_replaying_completed_nodes(
    redis_service, budget_service
):
    """
    Simulates a process killed mid-run after Planner and SQ-1 Researcher succeeded,
    while SQ-2 was still PENDING. Verifies startup recovery resumes at SQ-2 and
    skips re-executing Planner and SQ-1.
    """
    run_id = "run_crash_test_01"
    budget = budget_service.initialize_budget()

    plan = ResearchPlan(
        original_question="Evaluate durable execution engines",
        objective_summary="Compare checkpointing engines",
        scope_boundaries="2025-2026",
        sub_questions=[
            SubQuestion(
                id="SQ-1",
                question="How does Temporal persist workflow history?",
                rationale="Baseline",
                search_queries=["temporal workflow history"],
                expected_evidence="Architecture docs",
            ),
            SubQuestion(
                id="SQ-2",
                question="How does LangGraph persist checkpoints in Redis?",
                rationale="Target system",
                search_queries=["langgraph redis checkpointer"],
                expected_evidence="Code & docs",
            ),
        ],
    )

    sq1_finding = FindingRecord(
        finding_id="F-001",
        sub_question_id="SQ-1",
        claim="Temporal persists an append-only event history per workflow execution.",
        supporting_evidence="Temporal architecture guide",
        source_url="https://docs.temporal.io/workflows",
        source_title="Temporal Workflows",
        retrieved_passage="Event history is durably persisted.",
    )

    # Seed Redis as if worker died right after SQ-1 was approved
    crashed_checkpoint = RunCheckpoint(
        run_id=run_id,
        original_question=plan.original_question,
        status=RunStatus.RESEARCHING,
        current_stage="supervisor_reviewed_SQ-1_approve",
        plan=plan,
        sub_question_statuses={
            "SQ-1": SubQuestionStatus.COMPLETED,
            "SQ-2": SubQuestionStatus.PENDING,
        },
        findings=[sq1_finding],
        budget=budget,
        completed_node_keys=[
            "planner:v1",
            "researcher:SQ-1:pass_1",
            "supervisor:SQ-1:review_1",
        ],
    )
    await redis_service.save_checkpoint(crashed_checkpoint)
    await redis_service.set_cached_node_output(
        run_id,
        "planner:v1",
        {
            "plan": plan.model_dump(),
            "sub_question_statuses": {
                "SQ-1": "completed",
                "SQ-2": "pending",
            },
        },
    )

    # Configure mocks for the restarted worker process
    mock_planner = AsyncMock()
    mock_researcher = AsyncMock()
    mock_supervisor = AsyncMock()
    mock_writer = AsyncMock()

    sq2_finding = FindingRecord(
        finding_id="F-002",
        sub_question_id="SQ-2",
        claim="LangGraph serializes state checkpoints after each node into Redis.",
        supporting_evidence="RedisSaver implementation",
        source_url="https://langchain-ai.github.io/langgraph/",
        source_title="LangGraph Docs",
        retrieved_passage="Checkpoints are written after every step.",
    )
    mock_researcher.investigate.return_value = ResearcherExecutionResult(
        sub_question_id="SQ-2",
        new_findings=[sq2_finding],
        new_sources=[
            SourceMetadata(
                canonical_url="https://langchain-ai.github.io/langgraph",
                original_url="https://langchain-ai.github.io/langgraph/",
                domain="langchain-ai.github.io",
                title="LangGraph Docs",
                status=SourceStatus.SUCCESS,
            )
        ],
        input_tokens=200,
        output_tokens=80,
    )

    mock_supervisor.review_sub_question.return_value = (
        SupervisorDecision(
            sub_question_id="SQ-2",
            decision=SupervisorDecisionType.APPROVE,
            reasoning="Sufficient evidence collected.",
        ),
        90,
        40,
    )

    mock_writer.write_report.return_value = (
        ResearchReport(
            title="Durable Execution Engines",
            executive_summary="Both Temporal [F-001] and LangGraph [F-002] persist state.",
            sections=[
                ReportSection(
                    heading="Comparison",
                    content="Temporal uses event logs [F-001]; LangGraph uses state snapshots [F-002].",
                )
            ],
            cited_finding_ids=["F-001", "F-002"],
            rendered_markdown="# Durable Execution Engines\n...",
        ),
        300,
        200,
    )

    orchestrator = ResearchOrchestrator(
        planner=mock_planner,
        researcher=mock_researcher,
        supervisor=mock_supervisor,
        writer=mock_writer,
        budget_service=budget_service,
        redis_service=redis_service,
    )

    # Trigger startup recovery
    recovered = await recover_interrupted_runs(redis_service, orchestrator)
    assert recovered == 1

    # Wait briefly for background recovery task to finish
    import asyncio
    for _ in range(30):
        cp = await redis_service.load_checkpoint(run_id)
        if cp and cp.status == RunStatus.COMPLETED:
            break
        await asyncio.sleep(0.05)

    final_cp = await redis_service.load_checkpoint(run_id)
    assert final_cp is not None
    assert final_cp.status == RunStatus.COMPLETED
    assert len(final_cp.findings) == 2

    # Critical assertion: Planner was NOT called again (served from Redis cache)
    mock_planner.create_plan.assert_not_called()
    # Critical assertion: Researcher was called ONLY for SQ-2, never SQ-1
    assert mock_researcher.investigate.call_count == 1
    assert mock_researcher.investigate.call_args.kwargs["sub_question"].id == "SQ-2"