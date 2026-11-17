from unittest.mock import AsyncMock
import pytest
from app.agents.planner import PlannerAgent
from app.agents.researcher import PageExtractionOutput, ExtractedClaimItem, ResearcherAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.writer import WriterAgent
from app.models.findings import FindingRecord, SourceStatus
from app.models.research import (
    ReportSection,
    ResearchPlan,
    ResearchReport,
    SubQuestion,
    SupervisorDecisionType,
)
from app.services.extraction_service import ExtractionResult
from app.services.llm_service import StructuredLLMResponse
from app.services.search_service import RawSearchHit, SearchBatchResult


@pytest.mark.asyncio
async def test_planner_enforces_sub_question_cap():
    mock_llm = AsyncMock()
    mock_llm.generate_structured.return_value = StructuredLLMResponse(
        parsed=ResearchPlan(
            original_question="Ignored",
            objective_summary="Compare vector DBs",
            scope_boundaries="Open source only",
            sub_questions=[
                SubQuestion(
                    id=f"SQ-{i}",
                    question=f"Sub-question number {i} details?",
                    rationale="Needed",
                    search_queries=[f"query {i}"],
                    expected_evidence="Benchmarks",
                )
                for i in range(1, 5)
            ],
        ),
        input_tokens=120,
        output_tokens=250,
    )

    planner = PlannerAgent(llm_service=mock_llm)
    plan, in_tok, out_tok = await planner.create_plan("Compare vector DBs", max_sub_questions=2)
    assert len(plan.sub_questions) == 2
    assert plan.original_question == "Compare vector DBs"
    assert in_tok == 120 and out_tok == 250


@pytest.mark.asyncio
async def test_researcher_handles_paywall_and_extracts_valid_source(
    budget_service, source_service
):
    mock_search = AsyncMock()
    mock_search.search.return_value = SearchBatchResult(
        status=SourceStatus.SUCCESS,
        hits=[
            RawSearchHit(url="https://paywalled.example.com/a", title="Locked", snippet="..."),
            RawSearchHit(url="https://open.example.com/b", title="Open Docs", snippet="..."),
        ],
    )

    mock_extract = AsyncMock()
    mock_extract.fetch_and_extract.side_effect = [
        ExtractionResult(
            status=SourceStatus.PAYWALLED, http_status=403, error_detail="HTTP 403"
        ),
        ExtractionResult(
            status=SourceStatus.SUCCESS,
            text="LangGraph checkpoints persist state after every super-step.",
            title="LangGraph Persistence Docs",
            http_status=200,
        ),
    ]

    mock_llm = AsyncMock()
    mock_llm.generate_structured.return_value = StructuredLLMResponse(
        parsed=PageExtractionOutput(
            is_relevant=True,
            extracted_claims=[
                ExtractedClaimItem(
                    claim="LangGraph persists graph state after each super-step.",
                    supporting_evidence="Official documentation states checkpoints save every step.",
                    verbatim_passage="LangGraph checkpoints persist state after every super-step.",
                )
            ],
        ),
        input_tokens=300,
        output_tokens=100,
    )

    researcher = ResearcherAgent(
        llm_service=mock_llm,
        search_service=mock_search,
        extraction_service=mock_extract,
        source_service=source_service,
        budget_service=budget_service,
    )

    sq = SubQuestion(
        id="SQ-1",
        question="How does LangGraph persist execution state?",
        rationale="Core architecture",
        search_queries=["langgraph checkpoint persistence"],
        expected_evidence="Documentation",
    )
    budget = budget_service.initialize_budget()
    res = await researcher.investigate(
        sub_question=sq, budget=budget, existing_sources=[], existing_finding_count=0
    )

    assert len(res.new_sources) == 2
    assert res.new_sources[0].status == SourceStatus.PAYWALLED
    assert res.new_sources[1].status == SourceStatus.SUCCESS
    assert len(res.new_findings) == 1
    assert res.new_findings[0].finding_id == "F-001"


@pytest.mark.asyncio
async def test_supervisor_enforces_single_revision_limit():
    mock_llm = AsyncMock()
    supervisor = SupervisorAgent(llm_service=mock_llm)
    sq = SubQuestion(
        id="SQ-1",
        question="What is the throughput of Redis 7?",
        rationale="Perf",
        search_queries=["redis 7 throughput"],
        expected_evidence="Ops/sec",
    )
    finding = FindingRecord(
        finding_id="F-001",
        sub_question_id="SQ-1",
        claim="Redis 7 handles over 100k ops/sec on a single node.",
        supporting_evidence="Benchmark report",
        source_url="https://redis.io/bench",
        source_title="Redis Bench",
        retrieved_passage="Over 100,000 operations per second.",
    )

    decision, in_tok, out_tok = await supervisor.review_sub_question(
        sub_question=sq,
        findings=[finding],
        sources=[],
        revisions_used_for_sq=1,
        max_revisions=1,
    )
    assert decision.decision == SupervisorDecisionType.APPROVE
    assert in_tok == 0 and out_tok == 0
    mock_llm.generate_structured.assert_not_called()


@pytest.mark.asyncio
async def test_writer_strips_unverified_citations_and_renders_references(source_service):
    mock_llm = AsyncMock()
    mock_llm.generate_structured.return_value = StructuredLLMResponse(
        parsed=ResearchReport(
            title="State Persistence Report",
            executive_summary="Checkpointing ensures fault tolerance [F-001] [F-404].",
            sections=[
                ReportSection(
                    heading="Architecture",
                    content="Redis stores serialized state snapshots [F-001].",
                )
            ],
            research_limitations=["Limited production latency benchmarks."],
        ),
        input_tokens=400,
        output_tokens=300,
    )

    writer = WriterAgent(llm_service=mock_llm, source_service=source_service)
    plan = ResearchPlan(
        original_question="How do agents persist state?",
        objective_summary="Analyze durable state",
        scope_boundaries="Python frameworks",
        sub_questions=[
            SubQuestion(
                id="SQ-1",
                question="How does Redis store state?",
                rationale="Persistence",
                search_queries=["redis state"],
                expected_evidence="Docs",
            )
        ],
    )
    findings = [
        FindingRecord(
            finding_id="F-001",
            sub_question_id="SQ-1",
            claim="Redis stores serialized JSON checkpoints with TTLs.",
            supporting_evidence="Redis SET with EX",
            source_url="https://redis.io/docs",
            source_title="Redis Docs",
            retrieved_passage="SET key value EX seconds",
        )
    ]

    report, _, _ = await writer.write_report(plan=plan, findings=findings, conflicts=[])
    assert report.cited_finding_ids == ["F-001"]
    assert report.invalid_citations_stripped == ["F-404"]
    assert "[F-404]" not in (report.rendered_markdown or "")
    assert "https://redis.io/docs" in (report.rendered_markdown or "")
