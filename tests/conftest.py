import pytest
import fakeredis.aioredis
from app.config.settings import Settings
from app.services.budget_service import BudgetService
from app.services.redis_service import RedisPersistenceService
from app.services.source_service import SourceService


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        anthropic_api_key="sk-ant-test-key",
        tavily_api_key="tvly-test-key",
        api_auth_token="test-bearer-token",
        redis_url="redis://localhost:6379/15",
        max_sub_questions=3,
        max_searches_per_sub_question=2,
        max_total_searches=6,
        max_total_tokens=20_000,
        max_cost_usd=0.50,
        max_duration_seconds=120,
        max_revision_attempts=1,
        max_results_per_domain=2,
    )


@pytest.fixture
async def fake_redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def redis_service(fake_redis) -> RedisPersistenceService:
    return RedisPersistenceService(redis_client=fake_redis, ttl_seconds=3600)


@pytest.fixture
def budget_service(test_settings) -> BudgetService:
    return BudgetService(settings=test_settings)


@pytest.fixture
def source_service(test_settings) -> SourceService:
    return SourceService(max_results_per_domain=test_settings.max_results_per_domain)