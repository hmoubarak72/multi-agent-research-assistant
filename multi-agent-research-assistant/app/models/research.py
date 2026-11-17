from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_REVISION = "needs_revision"
    FAILED = "failed"
    BUDGET_STOPPED = "budget_stopped"


class SubQuestion(BaseModel):
    id: str = Field(..., description="Identifier like SQ-1")
    question: str = Field(..., min_length=10)
    rationale: str = Field(..., description="Why this sub-question is necessary")
    search_queries: List[str] = Field(..., min_length=1, max_length=3)
    expected_evidence: str


class ResearchPlan(BaseModel):
    original_question: str
    objective_summary: str
    scope_boundaries: str
    sub_questions: List[SubQuestion] = Field(..., min_length=1)


class SupervisorDecisionType(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    SKIP_INSUFFICIENT = "skip_insufficient"


class SupervisorDecision(BaseModel):
    sub_question_id: str
    decision: SupervisorDecisionType
    reasoning: str
    missing_aspects: List[str] = Field(default_factory=list)
    refined_queries: List[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str
    content: str = Field(
        ...,
        description="Markdown text containing inline citations formatted strictly as [F-XXX].",
    )


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: List[ReportSection] = Field(..., min_length=1)
    conflicting_evidence: List[str] = Field(default_factory=list)
    research_limitations: List[str] = Field(default_factory=list)
    cited_finding_ids: List[str] = Field(default_factory=list)
    invalid_citations_stripped: List[str] = Field(default_factory=list)
    rendered_markdown: Optional[str] = None
