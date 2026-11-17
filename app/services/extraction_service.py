from dataclasses import dataclass
from typing import Optional
import httpx
import trafilatura
from bs4 import BeautifulSoup
from app.models.findings import SourceStatus

PAYWALL_MARKERS = (
    "subscribe to read",
    "subscription required",
    "sign in to continue reading",
    "members-only content",
    "create a free account to continue",
)


@dataclass
class ExtractionResult:
    status: SourceStatus
    text: str = ""
    title: str = ""
    http_status: Optional[int] = None
    error_detail: Optional[str] = None


class ExtractionService:
    """Fetches full webpages and extracts clean main text, returning structured failure states."""

    def __init__(self, timeout_seconds: float = 10.0, max_chars: int = 6000):
        self.timeout_seconds = timeout_seconds
        self.max_chars = max_chars
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; ResearchAssistantBot/1.0; +https://example.org/bot)"
            ),
            "Accept": "text/html,application/xhtml+xml",
        }

    async def fetch_and_extract(self, url: str) -> ExtractionResult:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers=self.headers,
            ) as client:
                response = await client.get(url)
        except httpx.TimeoutException:
            return ExtractionResult(
                status=SourceStatus.TIMEOUT,
                error_detail=f"Request timed out after {self.timeout_seconds}s",
            )
        except httpx.HTTPError as exc:
            return ExtractionResult(
                status=SourceStatus.EXTRACTION_FAILED,
                error_detail=f"HTTP transport error: {type(exc).__name__}",
            )

        code = response.status_code
        if code in (401, 402, 403):
            return ExtractionResult(
                status=SourceStatus.PAYWALLED,
                http_status=code,
                error_detail=f"Access restricted (HTTP {code})",
            )
        if code == 429:
            return ExtractionResult(
                status=SourceStatus.RATE_LIMITED,
                http_status=code,
                error_detail="Target host returned HTTP 429 Too Many Requests",
            )
        if code >= 400:
            return ExtractionResult(
                status=SourceStatus.EXTRACTION_FAILED,
                http_status=code,
                error_detail=f"HTTP error status {code}",
            )

        html = response.text
        lower_sample = html[:4000].lower()
        if any(marker in lower_sample for marker in PAYWALL_MARKERS):
            return ExtractionResult(
                status=SourceStatus.PAYWALLED,
                http_status=code,
                error_detail="Detected paywall marker in page markup",
            )

        extracted = trafilatura.extract(
            html, include_comments=False, include_tables=True
        )
        soup = BeautifulSoup(html, "html.parser")
        page_title = soup.title.string.strip() if soup.title and soup.title.string else ""

        if not extracted:
            # Fallback to paragraph concatenation
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) > 40
            ]
            extracted = "\n".join(paragraphs)

        cleaned = (extracted or "").strip()
        if len(cleaned) < 120:
            return ExtractionResult(
                status=SourceStatus.EXTRACTION_FAILED,
                title=page_title,
                http_status=code,
                error_detail="Insufficient readable body text extracted (<120 chars)",
            )

        return ExtractionResult(
            status=SourceStatus.SUCCESS,
            text=cleaned[: self.max_chars],
            title=page_title,
            http_status=code,
        )