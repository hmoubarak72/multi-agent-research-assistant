import asyncio
import os
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_E2E") != "1",
    reason="Set RUN_LIVE_E2E=1 with real ANTHROPIC_API_KEY, TAVILY_API_KEY, and Redis to run live E2E.",
)
@pytest.mark.asyncio
async def test_live_end_to_end_research_workflow():
    app = create_app(auto_recover_on_startup=False)
    async with app.router.lifespan_context(app):
        token = app.state.settings.api_auth_token
        headers = {"Authorization": f"Bearer {token}"}
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/research",
                headers=headers,
                json={
                    "question": "What are the primary consensus mechanisms used by Ethereum and Solana?",
                    "budget_overrides": {
                        "max_sub_questions": 2,
                        "max_searches_per_sub_question": 1,
                        "max_cost_usd": 0.25,
                    },
                },
            )
            assert resp.status_code == 202
            run_id = resp.json()["run_id"]

            for _ in range(120):
                st = await client.get(f"/research/{run_id}", headers=headers)
                data = st.json()
                if data["status"] in ("completed", "budget_exceeded", "failed"):
                    assert data["status"] in ("completed", "budget_exceeded")
                    assert data["report"] is not None
                    assert len(data["report"]["cited_finding_ids"]) >= 1
                    break
                await asyncio.sleep(1.0)
