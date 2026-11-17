from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SourceStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    RATE_LIMITED = "rate_limited"
    PAYWALLED = "paywalled"
    TIMEOUT = "timeout"
    EXTRACTION_FAILED = "extraction_failed"
    DOMAIN_CAP_SKIPPED = "domain_cap_skipped"
    DUPLICATE_SKIPPED = "duplicate_skipped"


class SourceMetadata(BaseModel):
    canonical_url: str
    original_url: str
    domain: str
    title: str = ""
    retrieval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: SourceStatus
    http_status: Optional[int] = None
    error_detail: Optional[str] = None
    extracted_char_count: int = 0


class FindingRecord(BaseModel):
    finding_id: str = Field(..., description="Unique deterministic ID e.g. F-001")
    sub_question_id: str
    claim: str = Field(..., min_length=10)
    supporting_evidence: str = Field(..., min_length=10)
    source_url: str
    source_title: str
    retrieved_passage: str
    retrieval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_status: SourceStatus = SourceStatus.SUCCESS


class ConflictRecord(BaseModel):
    topic: str
    finding_id_a: str
    claim_a: str
    finding_id_b: str
    claim_b: str
    description: str
