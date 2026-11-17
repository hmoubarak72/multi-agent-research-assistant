import re
from typing import Dict, List, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from app.models.findings import FindingRecord, SourceMetadata, SourceStatus

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "fbclid",
    "gclid",
    "ref",
}


class SourceService:
    """Manages canonical URL normalization, deduplication, domain caps, and citation validation."""

    def __init__(self, max_results_per_domain: int = 2):
        self.max_results_per_domain = max_results_per_domain

    @staticmethod
    def canonicalize_url(raw_url: str) -> str:
        parsed = urlparse(raw_url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path.rstrip("/") or "/"
        filtered_query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=False)
            if k.lower() not in TRACKING_PARAMS
        ]
        query = urlencode(sorted(filtered_query))
        return urlunparse((scheme, netloc, path, "", query, ""))

    @staticmethod
    def extract_domain(canonical_url: str) -> str:
        return urlparse(canonical_url).netloc.lower()

    def evaluate_candidate_url(
        self,
        raw_url: str,
        existing_sources: List[SourceMetadata],
    ) -> Tuple[bool, str, str, SourceStatus]:
        canonical = self.canonicalize_url(raw_url)
        domain = self.extract_domain(canonical)

        seen_urls = {s.canonical_url for s in existing_sources}
        if canonical in seen_urls:
            return False, canonical, domain, SourceStatus.DUPLICATE_SKIPPED

        domain_success_count = sum(
            1
            for s in existing_sources
            if s.domain == domain and s.status == SourceStatus.SUCCESS
        )
        if domain_success_count >= self.max_results_per_domain:
            return False, canonical, domain, SourceStatus.DOMAIN_CAP_SKIPPED

        return True, canonical, domain, SourceStatus.SUCCESS

    @staticmethod
    def validate_and_clean_citations(
        text: str, valid_finding_ids: Set[str]
    ) -> Tuple[str, List[str], List[str]]:
        pattern = re.compile(r"\[(F-\d+)\]")
        used_valid: List[str] = []
        stripped_invalid: List[str] = []

        def _replace(match: re.Match) -> str:
            fid = match.group(1)
            if fid in valid_finding_ids:
                if fid not in used_valid:
                    used_valid.append(fid)
                return f"[{fid}]"
            if fid not in stripped_invalid:
                stripped_invalid.append(fid)
            return ""

        cleaned_text = pattern.sub(_replace, text)
        cleaned_text = re.sub(r"\s+([.,;:])", r"\1", cleaned_text)
        cleaned_text = re.sub(r" {2,}", " ", cleaned_text)
        return cleaned_text.strip(), used_valid, stripped_invalid

    @staticmethod
    def build_reference_list(
        findings: List[FindingRecord], cited_ids: List[str]
    ) -> str:
        finding_map: Dict[str, FindingRecord] = {f.finding_id: f for f in findings}
        lines = []
        for fid in cited_ids:
            f = finding_map.get(fid)
            if f:
                ts = f.retrieval_timestamp.strftime("%Y-%m-%d %H:%M UTC")
                lines.append(
                    f"- **[{fid}]** {f.source_title} — {f.source_url} *(Retrieved: {ts})*"
                )
        return "\n".join(lines)
