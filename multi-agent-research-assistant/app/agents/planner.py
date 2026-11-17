from typing import Tuple
from app.models.research import ResearchPlan
from app.services.llm_service import LLMService


class PlannerAgent:
    """Decomposes a research question into bounded, non-overlapping sub-questions."""

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def create_plan(
        self, question: str, max_sub_questions: int
    ) -> Tuple[ResearchPlan, int, int]:
        system_prompt = (
            "You decompose complex research questions into focused, non-overlapping "
            "sub-questions suitable for web search investigation. "
            f"Produce between 1 and {max_sub_questions} sub-questions. "
            "Each sub-question must target distinct factual evidence, include 1 to 3 "
            "concrete web search queries, and directly serve the primary objective. "
            "Avoid vague, philosophical, or redundant sub-questions."
        )
        user_prompt = (
            f"Research Question: {question}\n"
            f"Maximum Allowed Sub-Questions: {max_sub_questions}"
        )

        resp = await self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ResearchPlan,
            max_tokens=2048,
            temperature=0.0,
        )
        plan: ResearchPlan = resp.parsed  # type: ignore[assignment]
        plan.original_question = question
        plan.sub_questions = plan.sub_questions[:max_sub_questions]
        return plan, resp.input_tokens, resp.output_tokens
