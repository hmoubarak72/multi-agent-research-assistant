from typing import List, Tuple
from app.models.findings import ConflictRecord, FindingRecord
from app.models.research import ResearchPlan, ResearchReport
from app.services.llm_service import LLMService
from app.services.source_service import SourceService


class WriterAgent:
    """
    Synthesizes verified findings into a structured markdown report and validates
    every inline citation [F-XXX] against the actual finding registry.
    """

    def __init__(self, llm_service: LLMService, source_service: SourceService):
        self.llm_service = llm_service
        self.source_service = source_service

    async def write_report(
        self,
        plan: ResearchPlan,
        findings: List[FindingRecord],
        conflicts: List[ConflictRecord],
    ) -> Tuple[ResearchReport, int, int]:
        valid_ids = {f.finding_id for f in findings}

        findings_context = "\n\n".join(
            f"[{f.finding_id}] (Sub-Question: {f.sub_question_id})\n"
            f"Claim: {f.claim}\n"
            f"Evidence: {f.supporting_evidence}\n"
            f"Source: {f.source_title} ({f.source_url})"
            for f in findings
        ) or "NO VERIFIED FINDINGS COLLECTED."

        conflicts_context = "\n".join(
            f"- {c.topic}: [{c.finding_id_a}] ({c.claim_a}) vs [{c.finding_id_b}] ({c.claim_b})"
            for c in conflicts
        ) or "None detected."

        system_prompt = (
            "Synthesize the verified research findings into an objective, executive-ready report. "
            "RULES:\n"
            "1. Every factual claim MUST carry an inline citation matching an exact finding ID, "
            "formatted strictly as [F-001], [F-002], etc.\n"
            "2. NEVER invent citation IDs or rely on outside knowledge not present in the findings.\n"
            "3. Clearly distinguish established facts from analytical synthesis.\n"
            "4. Explicitly document any research gaps or sub-questions lacking sufficient evidence."
        )
        user_prompt = (
            f"Original Research Question: {plan.original_question}\n"
            f"Objective Summary: {plan.objective_summary}\n\n"
            f"=== VERIFIED FINDINGS REGISTRY ===\n{findings_context}\n\n"
            f"=== DETECTED SOURCE CONFLICTS ===\n{conflicts_context}"
        )

        resp = await self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=ResearchReport,
            max_tokens=4096,
            temperature=0.1,
        )
        report: ResearchReport = resp.parsed  # type: ignore[assignment]

        # Post-generation deterministic citation validation
        all_used: List[str] = []
        all_stripped: List[str] = []

        cleaned_summary, u_sum, s_sum = self.source_service.validate_and_clean_citations(
            report.executive_summary, valid_ids
        )
        report.executive_summary = cleaned_summary
        all_used.extend(u_sum)
        all_stripped.extend(s_sum)

        for section in report.sections:
            cleaned_sec, u_sec, s_sec = self.source_service.validate_and_clean_citations(
                section.content, valid_ids
            )
            section.content = cleaned_sec
            for fid in u_sec:
                if fid not in all_used:
                    all_used.append(fid)
            for fid in s_sec:
                if fid not in all_stripped:
                    all_stripped.append(fid)

        report.cited_finding_ids = all_used
        report.invalid_citations_stripped = all_stripped
        report.rendered_markdown = self._render_markdown(report, findings)

        return report, resp.input_tokens, resp.output_tokens

    def _render_markdown(
        self, report: ResearchReport, findings: List[FindingRecord]
    ) -> str:
        parts = [
            f"# {report.title}",
            "## Executive Summary",
            report.executive_summary,
        ]
        for sec in report.sections:
            parts.append(f"## {sec.heading}")
            parts.append(sec.content)

        if report.conflicting_evidence:
            parts.append("## Conflicting Evidence")
            parts.extend(f"- {item}" for item in report.conflicting_evidence)

        if report.research_limitations:
            parts.append("## Research Limitations & Evidence Gaps")
            parts.extend(f"- {item}" for item in report.research_limitations)

        parts.append("## Verified References")
        ref_block = self.source_service.build_reference_list(
            findings, report.cited_finding_ids
        )
        parts.append(ref_block or "_No external sources cited._")
        return "\n\n".join(parts)