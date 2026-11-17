from dataclasses import dataclass, field
from typing import List, Optional
import asyncio
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from app.models.findings import SourceStatus


@dataclass
class RawSearchHit:
    url: str
    title: str
    snippet: str
    score: float = 0.0


@dataclass
class SearchBatchResult:
    status: SourceStatus
    hits: List[RawSearchHit] = field(default_factory=list)
    error_detail: Optional[str] = None


class SearchService:
    """Wraps Tavily search with retries and structured failure states."""

    def __init__(self, api_key: str):
        self.client = TavilyClient(api_key=api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=6),
        reraise=True,
    )
    def _call_tavily_sync(self, query: str, max_results: int) -> dict:
        return self.client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False,
        )

    async def search(self, query: str, max_results: int = 5) -> SearchBatchResult:
        try:
            response = await asyncio.to_thread(
                self._call_tavily_sync, query, max_results
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "rate" in msg:
                return SearchBatchResult(
                    status=SourceStatus.RATE_LIMITED,
                    error_detail=f"Tavily rate limit: {exc}",
                )
            if "timeout" in msg:
                return SearchBatchResult(
                    status=SourceStatus.TIMEOUT,
                    error_detail=f"Tavily timeout: {exc}",
                )
            return SearchBatchResult(
                status=SourceStatus.EXTRACTION_FAILED,
                error_detail=f"Tavily API error: {exc}",
            )

        raw_results = response.get("results", [])
        if not raw_results:
            return SearchBatchResult(
                status=SourceStatus.NO_RESULTS,
                error_detail=f"No search hits returned for query: {query}",
            )

        hits = [
            RawSearchHit(
                url=item.get("url", ""),
                title=item.get("title", "Untitled Source"),
                snippet=item.get("content", ""),
                score=float(item.get("score", 0.0)),
            )
            for item in raw_results
            if item.get("url")
        ]
        return SearchBatchResult(status=SourceStatus.SUCCESS, hits=hits)
