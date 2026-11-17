from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    RESEARCHING = "researching"
    REVIEWING = "reviewing"
    WRITING = "writing"
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


class BudgetState(BaseModel):
    max_sub_questions: int
    max_searches_per_sub_question: int
    max_total_searches: int
    max_total_tokens: int
    max_cost_usd: float
    max_duration_seconds: int
    max_revision_attempts: int

    searches_used: int = 0
    searches_by_sub_question: Dict[str, int] = Field(default_factory=dict)
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    started_at_epoch: float = 0.0
    exceeded_reason: Optional[str] = None

    @property
    def total_tokens_used(self) -> int:
        return self.input_tokens_used + self.output_tokens_used


class TraceStep(BaseModel):
    step_number: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    node_name: str
    agent_name: str
    input_summary: Dict[str, Any]
    output_summary: Dict[str, Any]
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    step_cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached_hit: bool = False
    error: Optional[str] = None