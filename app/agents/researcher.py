from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.models.findings import FindingRecord, SourceMetadata, SourceStatus
from app.models.research import SubQuestion
from app.models.schemas import BudgetState
from app.services.budget_service import BudgetService
from app.services.extraction_service import ExtractionService
from app.services.llm_service import LLMService
from app.services.search_service import SearchService
from app.services.source_service import SourceService


class ExtractedClaimItem(BaseModel):
    claim: str = Field(..., min_length=10)
    supporting_evidence: str = Field(..., min_length=10)
    verbatim_passage: str = Field(..., min_length=10)


class PageExtractionOutput(BaseModel):
    is_relevant: bool
    extracted_claims: List[ExtractedClaimItem] = Field(default_factory=list)


@dataclass
class ResearcherExecutionResult:
    sub_question_id: str
    new_findings: List[FindingRecord] = field(default_factory=list)
    new_sources: List[SourceMetadata] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    budget_halted: bool = False
    halt_reason: Optional[str] = None


class ResearcherAgent:
    """
    Executes bounded web searches for a single sub-question, fetches full pages,
    enforces domain/URL deduplication, and extracts provenance-backed findings.
    """

    def __init__(
        self,
        llm_service: LLMService,
        search_service: SearchService,
        extraction_service: ExtractionService,
        source_service: SourceService,
        budget_service: BudgetService,
    ):
        self.llm_service = llm_service
        self.search_service = search_service
        self.extraction_service = extraction_service
        self.source_service = source_service
        self.budget_service = budget_service

    async def investigate(
        self,
        sub_question: SubQuestion,
        budget: BudgetState,
        existing_sources: List[SourceMetadata],
        existing_finding_count: int,
        override_queries: Optional[List[str]] = None,
    ) -> ResearcherExecutionResult:
        result = ResearcherExecutionResult(sub_question_id=sub_question.id)
        tracked_sources = list(existing_sources)
        finding_counter = existing_finding_count
        queries = override_queries if override_queries else sub_question.search_queries

        for query in queries:
            can_search, deny_reason = self.budget_service.can_execute_search(
                budget, sub_question.id
            )
            if not can_search:
                result.budget_halted = True
                result.halt_reason = deny_reason
                break

            self.budget_service.record_search(budget, sub_question.id)
            search_batch = await self.search_service.search(query=query, max_results=4)
            result.tool_calls.append(
                {
                    "tool": "tavily_search",
                    "query": query,
                    "status": search_batch.status.value,
                    "hit_count": len(search_batch.hits),
                    "error": search_batch.error_detail,
                }
            )

            if search_batch.status != SourceStatus.SUCCESS:
                continue

            for hit in search_batch.hits:
                allowed_budget, b_reason = self.budget_service.check_execution_allowed(
                    budget
                )
                if not allowed_budget:
                    result.budget_halted = True
                    result.halt_reason = b_reason
                    return result

                allowed_url, canonical, domain, skip_status = (
                    self.source_service.evaluate_candidate_url(
                        hit.url, tracked_sources
                    )
                )
                if not allowed_url:
                    meta = SourceMetadata(
                        canonical_url=canonical,
                        original_url=hit.url,
                        domain=domain,
                        title=hit.title,
                        status=skip_status,
                        error_detail=f"Filtered by source policy: {skip_status.value}",
                    )
                    tracked_sources.append(meta)
                    result.new_sources.append(meta)
                    continue

                extracted = await self.extraction_service.fetch_and_extract(canonical)
                result.tool_calls.append(
                    {
                        "tool": "web_extract",
                        "url": canonical,
                        "status": extracted.status.value,
                        "chars": len(extracted.text),
                        "error": extracted.error_detail,
                    }
                )

                source_title = extracted.title or hit.title
                meta = SourceMetadata(
                    canonical_url=canonical,
                    original_url=hit.url,
                    domain=domain,
                    title=source_title,
                    status=extracted.status,
                    http_status=extracted.http_status,
                    error_detail=extracted.error_detail,
                    extracted_char_count=len(extracted.text),
                )
                tracked_sources.append(meta)
                result.new_sources.append(meta)

                if extracted.status != SourceStatus.SUCCESS:
                    continue

                # Extract grounded claims from full page text
                claims_resp = await self._extract_claims_from_text(
                    sub_question=sub_question,
                    source_url=canonical,
                    source_title=source_title,
                    page_text=extracted.text,
                )
                result.input_tokens += claims_resp[1]
                result.output_tokens += claims_resp[2]
                self.budget_service.record_llm_usage(
                    budget, claims_resp[1], claims_resp[2]
                )

                extraction_output: PageExtractionOutput = claims_resp[0]
                if not extraction_output.is_relevant:
                    continue

                now_utc = datetime.now(timezone.utc)
                for item in extraction_output.extracted_claims:
                    finding_counter += 1
                    finding = FindingRecord(
                        finding_id=f"F-{finding_counter:03d}",
                        sub_question_id=sub_question.id,
                        claim=item.claim,
                        supporting_evidence=item.supporting_evidence,
                        source_url=canonical,
                        source_title=source_title,
                        retrieved_passage=item.verbatim_passage,
                        retrieval_timestamp=now_utc,
                        source_status=SourceStatus.SUCCESS,
                    )
                    result.new_findings.append(finding)

        return result

    async def _extract_claims_from_text(
        self,
        sub_question: SubQuestion,
        source_url: str,
        source_title: str,
        page_text: str,
    ) -> tuple[PageExtractionOutput, int, int]:
        system_prompt = (
            "Extract factual claims from the provided webpage text that directly answer "
            "the target sub-question. Every claim MUST be strictly supported by the text. "
            "Include a verbatim passage quote from the text. Never extrapolate or invent facts. "
            "If the page lacks relevant factual evidence, set is_relevant=false and return an empty list."
        )
        user_prompt = (
            f"Sub-Question ({sub_question.id}): {sub_question.question}\n"
            f"Expected Evidence: {sub_question.expected_evidence}\n"
            f"Source Title: {source_title}\n"
            f"Source URL: {source_url}\n\n"
            f"--- EXTRACTED WEBPAGE TEXT ---\n{page_text}"
        )
        resp = await self.llm_service.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_schema=PageExtractionOutput,
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.parsed, resp.input_tokens, resp.output_tokens  # type: ignore[return-value]