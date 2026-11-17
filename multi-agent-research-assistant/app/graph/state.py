from typing import Dict, List, Optional, TypedDict
from app.models.findings import ConflictRecord, FindingRecord, SourceMetadata
from app.models.research import (
    ResearchPlan,
    ResearchReport,
    SubQuestionStatus,
    SupervisorDecision,
)
from app.models.schemas import BudgetState, RunStatus, TraceStep
from app.services.redis_service import RunCheckpoint


class ResearchGraphState(TypedDict):
    run_id: str
    original_question: str
    status: RunStatus
    current_stage: str
    created_at_iso: str

    plan: Optional[ResearchPlan]
    sub_question_statuses: Dict[str, SubQuestionStatus]
    revisions_by_sq: Dict[str, int]
    refined_queries_by_sq: Dict[str, List[str]]
    active_sub_question_id: Optional[str]

    findings: List[FindingRecord]
    sources: List[SourceMetadata]
    conflicts: List[ConflictRecord]
    supervisor_decisions: List[SupervisorDecision]

    budget: BudgetState
    trace_steps: List[TraceStep]
    completed_node_keys: List[str]

    report: Optional[ResearchReport]
    error_message: Optional[str]


def checkpoint_to_graph_state(cp: RunCheckpoint) -> ResearchGraphState:
    return ResearchGraphState(
        run_id=cp.run_id,
        original_question=cp.original_question,
        status=cp.status,
        current_stage=cp.current_stage,
        created_at_iso=cp.created_at.isoformat(),
        plan=cp.plan,
        sub_question_statuses=dict(cp.sub_question_statuses),
        revisions_by_sq=dict(cp.revisions_by_sq),
        refined_queries_by_sq=dict(cp.refined_queries_by_sq),
        active_sub_question_id=cp.active_sub_question_id,
        findings=list(cp.findings),
        sources=list(cp.sources),
        conflicts=list(cp.conflicts),
        supervisor_decisions=list(cp.supervisor_decisions),
        budget=cp.budget,
        trace_steps=list(cp.trace_steps),
        completed_node_keys=list(cp.completed_node_keys),
        report=cp.report,
        error_message=cp.error_message,
    )


def graph_state_to_checkpoint(
    state: ResearchGraphState, existing_cp: RunCheckpoint
) -> RunCheckpoint:
    return RunCheckpoint(
        run_id=state["run_id"],
        original_question=state["original_question"],
        status=state["status"],
        current_stage=state["current_stage"],
        created_at=existing_cp.created_at,
        plan=state["plan"],
        sub_question_statuses=state["sub_question_statuses"],
        revisions_by_sq=state["revisions_by_sq"],
        refined_queries_by_sq=state["refined_queries_by_sq"],
        active_sub_question_id=state["active_sub_question_id"],
        findings=state["findings"],
        sources=state["sources"],
        conflicts=state["conflicts"],
        supervisor_decisions=state["supervisor_decisions"],
        budget=state["budget"],
        trace_steps=state["trace_steps"],
        completed_node_keys=state["completed_node_keys"],
        report=state["report"],
        error_message=state["error_message"],
    )
