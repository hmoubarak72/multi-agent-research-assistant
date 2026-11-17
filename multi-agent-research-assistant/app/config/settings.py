from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(..., min_length=1)
    tavily_api_key: str = Field(..., min_length=1)

    api_auth_token: str = Field(default="dev-secret-token")
    rate_limit_per_minute: int = Field(default=20, ge=1)
    max_question_length: int = Field(default=2000, ge=50)

    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_run_ttl_seconds: int = Field(default=86400, ge=60)

    claude_model: str = Field(default="claude-sonnet-4-20250514")
    claude_input_cost_per_1m: float = Field(default=3.00, ge=0.0)
    claude_output_cost_per_1m: float = Field(default=15.00, ge=0.0)

    max_sub_questions: int = Field(default=5, ge=1, le=10)
    max_searches_per_sub_question: int = Field(default=3, ge=1, le=10)
    max_total_searches: int = Field(default=15, ge=1, le=50)
    max_total_tokens: int = Field(default=100_000, ge=1000)
    max_cost_usd: float = Field(default=1.00, gt=0.0)
    max_duration_seconds: int = Field(default=240, ge=10)
    max_revision_attempts: int = Field(default=1, ge=0, le=2)

    max_results_per_domain: int = Field(default=2, ge=1, le=5)
    extraction_timeout_seconds: float = Field(default=10.0, gt=0.5)
    max_extracted_chars: int = Field(default=6000, ge=500)


@lru_cache
def get_settings() -> Settings:
    return Settings()
