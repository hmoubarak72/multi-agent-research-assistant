import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
import redis.asyncio as redis
from pydantic import BaseModel, Field
from app.models.findings import ConflictRecord, FindingRecord, SourceMetadata
from app.models.research import (
    ResearchPlan,
    ResearchReport,
    SubQuestionStatus,
    SupervisorDecision,
)
from app.models.schemas import BudgetState, RunStatus, TraceStep


class RunEvent(BaseModel):
    event_type: str
    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = Field(default_factory=dict)


class RunCheckpoint(BaseModel):
    run_id: str
    original_question: str
    status: RunStatus = RunStatus.QUEUED
    current_stage: str = "initialized"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    plan: Optional[ResearchPlan] = None
    sub_question_statuses: Dict[str, SubQuestionStatus] = Field(default_factory=dict)
    revisions_by_sq: Dict[str, int] = Field(default_factory=dict)
    refined_queries_by_sq: Dict[str, List[str]] = Field(default_factory=dict)
    active_sub_question_id: Optional[str] = None

    findings: List[FindingRecord] = Field(default_factory=list)
    sources: List[SourceMetadata] = Field(default_factory=list)
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    supervisor_decisions: List[SupervisorDecision] = Field(default_factory=list)

    budget: BudgetState
    trace_steps: List[TraceStep] = Field(default_factory=list)
    completed_node_keys: List[str] = Field(default_factory=list)

    report: Optional[ResearchReport] = None
    error_message: Optional[str] = None


class RedisPersistenceService:
    """
    Handles durable graph checkpoints, node-level idempotency cache,
    run discovery for crash recovery, and SSE event streaming via Redis.
    """

    def __init__(self, redis_client: redis.Redis, ttl_seconds: int = 86400):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _state_key(run_id: str) -> str:
        return f"research:run:{run_id}:state"

    @staticmethod
    def _node_cache_key(run_id: str, step_key: str) -> str:
        return f"research:run:{run_id}:node:{step_key}"

    @staticmethod
    def _events_list_key(run_id: str) -> str:
        return f"research:run:{run_id}:events"

    @staticmethod
    def _events_channel(run_id: str) -> str:
        return f"research:run:{run_id}:channel"

    @staticmethod
    def _active_runs_set() -> str:
        return "research:active_runs"

    async def save_checkpoint(self, checkpoint: RunCheckpoint) -> None:
        checkpoint.updated_at = datetime.now(timezone.utc)
        payload = checkpoint.model_dump_json()
        key = self._state_key(checkpoint.run_id)

        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(key, payload, ex=self.ttl_seconds)
            if checkpoint.status in (
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.BUDGET_EXCEEDED,
            ):
                pipe.srem(self._active_runs_set(), checkpoint.run_id)
            else:
                pipe.sadd(self._active_runs_set(), checkpoint.run_id)
            await pipe.execute()

    async def load_checkpoint(self, run_id: str) -> Optional[RunCheckpoint]:
        raw = await self.redis.get(self._state_key(run_id))
        if not raw:
            return None
        return RunCheckpoint.model_validate_json(raw)

    async def list_incomplete_run_ids(self) -> List[str]:
        members = await self.redis.smembers(self._active_runs_set())
        return [
            m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members
        ]

    async def get_cached_node_output(
        self, run_id: str, step_key: str
    ) -> Optional[Dict[str, Any]]:
        raw = await self.redis.get(self._node_cache_key(run_id, step_key))
        if not raw:
            return None
        return json.loads(raw)

    async def set_cached_node_output(
        self, run_id: str, step_key: str, output: Dict[str, Any]
    ) -> None:
        await self.redis.set(
            self._node_cache_key(run_id, step_key),
            json.dumps(output, default=str),
            ex=self.ttl_seconds,
        )

    async def emit_event(
        self, run_id: str, event_type: str, data: Optional[Dict[str, Any]] = None
    ) -> RunEvent:
        event = RunEvent(event_type=event_type, run_id=run_id, data=data or {})
        serialized = event.model_dump_json()

        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(self._events_list_key(run_id), serialized)
            pipe.expire(self._events_list_key(run_id), self.ttl_seconds)
            pipe.publish(self._events_channel(run_id), serialized)
            await pipe.execute()

        return event

    async def get_event_history(self, run_id: str) -> List[RunEvent]:
        raw_items = await self.redis.lrange(self._events_list_key(run_id), 0, -1)
        events = []
        for item in raw_items:
            text = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            events.append(RunEvent.model_validate_json(text))
        return events

    async def subscribe_events(self, run_id: str) -> AsyncIterator[RunEvent]:
        """
        Replays historical events first, then streams live Pub/Sub events
        until a terminal run event is observed.
        """
        terminal_events = {
            "report_completed",
            "execution_failed",
            "budget_exceeded",
        }

        history = await self.get_event_history(run_id)
        for ev in history:
            yield ev
            if ev.event_type in terminal_events:
                return

        pubsub = self.redis.pubsub()
        channel = self._events_channel(run_id)
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                raw = message["data"]
                text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                event = RunEvent.model_validate_json(text)
                yield event
                if event.event_type in terminal_events:
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()