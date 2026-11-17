import time
from typing import Optional, Tuple
from app.config.settings import Settings
from app.models.schemas import BudgetState


class BudgetExceededError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class BudgetService:
    """Enforces deterministic token, cost, search, and wall-clock limits."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def initialize_budget(self, overrides: Optional[dict] = None) -> BudgetState:
        overrides = overrides or {}
        return BudgetState(
            max_sub_questions=min(
                overrides.get("max_sub_questions", self.settings.max_sub_questions),
                self.settings.max_sub_questions,
            ),
            max_searches_per_sub_question=min(
                overrides.get(
                    "max_searches_per_sub_question",
                    self.settings.max_searches_per_sub_question,
                ),
                self.settings.max_searches_per_sub_question,
            ),
            max_total_searches=min(
                overrides.get("max_total_searches", self.settings.max_total_searches),
                self.settings.max_total_searches,
            ),
            max_total_tokens=min(
                overrides.get("max_total_tokens", self.settings.max_total_tokens),
                self.settings.max_total_tokens,
            ),
            max_cost_usd=min(
                overrides.get("max_cost_usd", self.settings.max_cost_usd),
                self.settings.max_cost_usd,
            ),
            max_duration_seconds=min(
                overrides.get(
                    "max_duration_seconds", self.settings.max_duration_seconds
                ),
                self.settings.max_duration_seconds,
            ),
            max_revision_attempts=self.settings.max_revision_attempts,
            started_at_epoch=time.time(),
        )

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.settings.claude_input_cost_per_1m
        output_cost = (
            output_tokens / 1_000_000
        ) * self.settings.claude_output_cost_per_1m
        return round(input_cost + output_cost, 6)

    def check_execution_allowed(self, budget: BudgetState) -> Tuple[bool, Optional[str]]:
        if budget.exceeded_reason:
            return False, budget.exceeded_reason

        elapsed = time.time() - budget.started_at_epoch
        if elapsed > budget.max_duration_seconds:
            reason = f"Wall-clock limit exceeded ({elapsed:.1f}s > {budget.max_duration_seconds}s)"
            budget.exceeded_reason = reason
            return False, reason

        if budget.total_tokens_used >= budget.max_total_tokens:
            reason = f"Token ceiling reached ({budget.total_tokens_used} >= {budget.max_total_tokens})"
            budget.exceeded_reason = reason
            return False, reason

        if budget.estimated_cost_usd >= budget.max_cost_usd:
            reason = f"Cost ceiling reached (${budget.estimated_cost_usd:.4f} >= ${budget.max_cost_usd:.4f})"
            budget.exceeded_reason = reason
            return False, reason

        return True, None

    def record_llm_usage(
        self, budget: BudgetState, input_tokens: int, output_tokens: int
    ) -> float:
        step_cost = self.calculate_cost(input_tokens, output_tokens)
        budget.input_tokens_used += input_tokens
        budget.output_tokens_used += output_tokens
        budget.estimated_cost_usd = round(budget.estimated_cost_usd + step_cost, 6)
        self.check_execution_allowed(budget)
        return step_cost

    def can_execute_search(
        self, budget: BudgetState, sub_question_id: str
    ) -> Tuple[bool, Optional[str]]:
        allowed, reason = self.check_execution_allowed(budget)
        if not allowed:
            return False, reason

        if budget.searches_used >= budget.max_total_searches:
            return False, f"Total search cap reached ({budget.max_total_searches})"

        sq_count = budget.searches_by_sub_question.get(sub_question_id, 0)
        if sq_count >= budget.max_searches_per_sub_question:
            return (
                False,
                f"Sub-question search cap reached ({budget.max_searches_per_sub_question}) for {sub_question_id}",
            )

        return True, None

    def record_search(self, budget: BudgetState, sub_question_id: str) -> None:
        budget.searches_used += 1
        budget.searches_by_sub_question[sub_question_id] = (
            budget.searches_by_sub_question.get(sub_question_id, 0) + 1
        )
