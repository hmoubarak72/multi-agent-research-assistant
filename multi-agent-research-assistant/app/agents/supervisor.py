from typing import List, Tuple
from app.models.findings import FindingRecord, SourceMetadata
from app.models.research import (
    SubQuestion,
    SupervisorDecision,
    SupervisorDecisionType,
)
from app.services.llm_service import LLMService


class SupervisorAgent:
    """
    Evaluates sub-question findings for completeness and source provenance,
    enforcing a hard maximum of one revision pass per sub-question.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def review_sub_question(
        self,
        sub_question: SubQuestion,
        findings: List[FindingRecord],
        sources: List[SourceMetadata],
        revisions_used_for_sq: int,
        max_revisions: int,
    ) -> Tuple[SupervisorDecision, int, int]:
        sq_findings = [
            f for f in findings if f.sub_question_id == sub_question.id
        ]

        if revisions_used_for_sq >= max_revisions:
            if sq_findings:
                return (
                    SupervisorDecision(
                        sub_question_id=sub_question.id,
                        decision=SupervisorDecisionType.APPROVE,
                        reasoning="Revision limit reached; proceeding with collected grounded findings.",
                    ),
                    0,
                    0,
                )
            return (
                SupervisorDecision(
                    sub_question_id=sub_question.id,
                    decision=SupervisorDecisionType.SKIP_INSUFFICIENT,
                    reasoning="Revision limit reached with zero valid findings; marking evidence gap.",
                ),
                0,
                0,
            )

        findings_digest = "\n".join(
            f"- [{f.finding_id}] {f.claim} (Source: {f.source_url})"
            for f in sq_findings
        ) or "NONE"

        source_status_summary = "\n".join(
            f"- {s.canonical_url} -> {s.status.value} ({s.error_detail or 'ok'})"
            for s in sources[-8:]
        ) or "NONE"

        system_prompt = (
            "Evaluate whether the collected findings sufficiently answer the sub-question "
            "based on its expected evidence. "
            "Choose 'approve' if credible, grounded findings address the core question. "
            "Choose 'revise' ONLY if critical evidence is missing AND refined search queries "
            "are likely to find it. Provide 1-2 targeted refined_queries when choosing 'revise'."
        )
        user_prompt = (
            f"Sub-Question ({sub_question.id}): {sub_question.question}\n"
            f"Expected Evidence: {sub_question.expected_evidence}\n\n"
            f"Collected Findings:\n{findings_digest}\n\n"
            f"Recent Source Attempt Outcomes:\n{source_status_summary}"
        )

        resp = await self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=SupervisorDecision,
            max_tokens=1024,
            temperature=0.0,
        )
        decision: SupervisorDecision = resp.parsed  # type: ignore[assignment]
        decision.sub_question_id = sub_question.id

        if (
            decision.decision == SupervisorDecisionType.REVISE
            and revisions_used_for_sq >= max_revisions
        ):
            decision.decision = (
                SupervisorDecisionType.APPROVE
                if sq_findings
                else SupervisorDecisionType.SKIP_INSUFFICIENT
            )

        return decision, resp.input_tokens, resp.output_tokens
