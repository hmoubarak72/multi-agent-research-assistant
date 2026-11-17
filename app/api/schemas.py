from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.models.research import (
    ResearchPlan,
    ResearchReport,
    SubQuestionStatus,
    SupervisorDecision,
)
from app.models.schemas import BudgetState, RunStatus, TraceStep


class BudgetOverrideRequest(BaseModel):
    max_sub_questions: Optional[int] = Field(default=None, ge=1, le=10)
    max_searches_per_sub_question: Optional[int] = Field(default=None, ge=1, le=10)
    max_total_searches: Optional[int] = Field(default=None, ge=1, le=50)
    max_total_tokens: Optional[int] = Field(default=None, ge=1000)
    max_cost_usd: Optional[float] = Field(default=None, gt=0.0)
    max_duration_seconds: Optional[int] = Field(default=None, ge=10)


class CreateResearchRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=2000)
    budget_overrides: Optional[BudgetOverrideRequest] = None


class CreateResearchResponse(BaseModel):
    run_id: str
    status: RunStatus
    created_at: datetime


class ResearchProgressSummary(BaseModel):
    total_sub_questions: int
    completed_sub_questions: int
    active_sub_question_id: Optional[str] = None
    sub_question_statuses: Dict[str, SubQuestionStatus]
    findings_collected: int
    sources_evaluated: int


class ResearchStatusResponse(BaseModel):
    run_id: str
    original_question: str
    status: RunStatus
    current_stage: str
    created_at: datetime
    updated_at: datetime
    progress: ResearchProgressSummary
    plan: Optional[ResearchPlan] = None
    report: Optional[ResearchReport] = None
    budget: BudgetState
    error_message: Optional[str] = None


class ResearchTraceResponse(BaseModel):
    run_id: str
    status: RunStatus
    total_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_estimated_cost_usd: float
    supervisor_decisions: List[SupervisorDecision]
    steps: List[TraceStep]
    error_message: Optional[str] = None