import time
from typing import Any, Dict, List, Optional
from app.models.schemas import TraceStep


class TraceRecorder:
    """Builds structured, sanitized step-level traces for post-mortem debugging."""

    def __init__(self, existing_steps: Optional[List[TraceStep]] = None):
        self.steps: List[TraceStep] = list(existing_steps) if existing_steps else []

    def record_step(
        self,
        node_name: str,
        agent_name: str,
        input_summary: Dict[str, Any],
        output_summary: Dict[str, Any],
        start_time: float,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        step_cost_usd: float = 0.0,
        cached_hit: bool = False,
        error: Optional[str] = None,
    ) -> TraceStep:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step = TraceStep(
            step_number=len(self.steps) + 1,
            node_name=node_name,
            agent_name=agent_name,
            input_summary=input_summary,
            output_summary=output_summary,
            tool_calls=tool_calls or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            step_cost_usd=round(step_cost_usd, 6),
            latency_ms=latency_ms,
            cached_hit=cached_hit,
            error=error,
        )
        self.steps.append(step)
        return step
