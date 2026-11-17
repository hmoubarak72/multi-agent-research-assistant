import asyncio
from unittest.mock import AsyncMock
import pytest
from httpx import ASGITransport, AsyncClient
from app.agents.researcher import ResearcherExecutionResult
from app.main import create_app
from app.models.findings import FindingRecord
from app.models.research import (
    ReportSection,
    ResearchPlan,
    ResearchReport,
    SubQuestion,
    SupervisorDecision,
    SupervisorDecisionType,
)
from app.models.schemas import RunStatus


@pytest.mark.asyncio
async def test_api_auth_rejection_and_end_to_end_run(test_settings, fake_redis):
    app = create_app(
        settings_override=test_settings,
        redis_client_override=fake_redis,
        auto_recover_on_startup=False,
    )

    async with app.router.lifespan_context(app):
        app.state.orchestrator.planner.create_plan = AsyncMock(
            return_value=(
                ResearchPlan(
                    original_question="What are the trade-offs of Redis Streams vs Pub/Sub?",
                    objective_summary="Compare Redis messaging primitives",
                    scope_boundaries="Redis 7+",
                    sub_questions=[
                        SubQuestion(
                            id="SQ-1",
                            question="How does delivery guarantee differ between Redis Streams and Pub/Sub?",
                            rationale="Reliability",
                            search_queries=["redis streams vs pubsub delivery guarantee"],
                            expected_evidence="Official Redis docs",
                        )
                    ],
                ),
                100,
                150,
            )
        )

        app.state.orchestrator.researcher.investigate = AsyncMock(
            return_value=ResearcherExecutionResult(
                sub_question_id="SQ-1",
                new_findings=[
                    FindingRecord(
                        finding_id="F-001",
                        sub_question_id="SQ-1",
                        claim="Redis Pub/Sub is fire-and-forget whereas Redis Streams persists messages with consumer groups.",
                        supporting_evidence="Redis documentation",
                        source_url="https://redis.io/docs/data-types/streams/",
                        source_title="Redis Streams",
                        retrieved_passage="Streams provide persistence and consumer groups.",
                    )
                ],
                input_tokens=200,
                output_tokens=100,
            )
        )

        app.state.orchestrator.supervisor.review_sub_question = AsyncMock(
            return_value=(
                SupervisorDecision(
                    sub_question_id="SQ-1",
                    decision=SupervisorDecisionType.APPROVE,
                    reasoning="Clear, grounded finding.",
                ),
                80,
                30,
            )
        )

        app.state.orchestrator.writer.write_report = AsyncMock(
            return_value=(
                ResearchReport(
                    title="Redis Streams vs Pub/Sub",
                    executive_summary="Redis Streams persists messages while Pub/Sub is fire-and-forget [F-001].",
                    sections=[
                        ReportSection(
                            heading="Delivery Semantics",
                            content="Consumer groups track acknowledgments in Streams [F-001].",
                        )
                    ],
                    cited_finding_ids=["F-001"],
                    rendered_markdown="# Redis Streams vs Pub/Sub\n...",
                ),
                250,
                200,
            )
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            unauth = await client.post(
                "/research",
                json={"question": "What are the trade-offs of Redis Streams vs Pub/Sub?"},
            )
            assert unauth.status_code == 401

            headers = {"Authorization": f"Bearer {test_settings.api_auth_token}"}

            create_resp = await client.post(
                "/research",
                headers=headers,
                json={"question": "What are the trade-offs of Redis Streams vs Pub/Sub?"},
            )
            assert create_resp.status_code == 202
            run_id = create_resp.json()["run_id"]

            status_payload = {}
            for _ in range(30):
                st_resp = await client.get(f"/research/{run_id}", headers=headers)
                assert st_resp.status_code == 200
                status_payload = st_resp.json()
                if status_payload["status"] == RunStatus.COMPLETED.value:
                    break
                await asyncio.sleep(0.05)

            assert status_payload["status"] == RunStatus.COMPLETED.value
            assert status_payload["report"]["cited_finding_ids"] == ["F-001"]

            trace_resp = await client.get(f"/research/{run_id}/trace", headers=headers)
            assert trace_resp.status_code == 200
            trace_data = trace_resp.json()
            assert trace_data["total_steps"] >= 4
            assert trace_data["total_estimated_cost_usd"] > 0
            agent_names = [step["agent_name"] for step in trace_data["steps"]]
            assert "PlannerAgent" in agent_names
            assert "ResearcherAgent" in agent_names
            assert "SupervisorAgent" in agent_names
            assert "WriterAgent" in agent_names
