import time
from typing import Any, Dict, List
from langgraph.graph import END, StateGraph
from app.agents.planner import PlannerAgent
from app.agents.researcher import ResearcherAgent
from app.agents.supervisor import SupervisorAgent
from app.agents.writer import WriterAgent
from app.graph.routing import (
    route_after_planner,
    route_after_researcher,
    route_after_selector,
    route_after_supervisor,
)
from app.graph.state import (
    ResearchGraphState,
    checkpoint_to_graph_state,
    graph_state_to_checkpoint,
)
from app.models.findings import ConflictRecord, FindingRecord, SourceMetadata
from app.models.research import (
    ResearchPlan,
    ResearchReport,
    SubQuestionStatus,
    SupervisorDecision,
    SupervisorDecisionType,
)
from app.models.schemas import RunStatus
from app.observability.logging import get_logger
from app.observability.tracing import TraceRecorder
from app.services.budget_service import BudgetService
from app.services.redis_service import RedisPersistenceService, RunCheckpoint

logger = get_logger(__name__)


class ResearchOrchestrator:
    """
    Compiles and executes the stateful, checkpointed LangGraph research workflow.
    Supports crash recovery and idempotent node replay from Redis.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        researcher: ResearcherAgent,
        supervisor: SupervisorAgent,
        writer: WriterAgent,
        budget_service: BudgetService,
        redis_service: RedisPersistenceService,
    ):
        self.planner = planner
        self.researcher = researcher
        self.supervisor = supervisor
        self.writer = writer
        self.budget_service = budget_service
        self.redis_service = redis_service
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(ResearchGraphState)

        builder.add_node("planner", self._planner_node)
        builder.add_node("select_sub_question", self._select_sub_question_node)
        builder.add_node("researcher", self._researcher_node)
        builder.add_node("supervisor_review", self._supervisor_review_node)
        builder.add_node("writer", self._writer_node)

        builder.set_entry_point("planner")

        builder.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "select_sub_question": "select_sub_question",
                "halt": END,
            },
        )
        builder.add_conditional_edges(
            "select_sub_question",
            route_after_selector,
            {
                "researcher": "researcher",
                "writer": "writer",
                "halt": END,
            },
        )
        builder.add_conditional_edges(
            "researcher",
            route_after_researcher,
            {
                "supervisor_review": "supervisor_review",
                "writer": "writer",
                "halt": END,
            },
        )
        builder.add_conditional_edges(
            "supervisor_review",
            route_after_supervisor,
            {
                "researcher": "researcher",
                "select_sub_question": "select_sub_question",
                "writer": "writer",
                "halt": END,
            },
        )
        builder.add_edge("writer", END)

        return builder.compile()

    async def _persist_state(
        self, state: ResearchGraphState, base_cp: RunCheckpoint
    ) -> None:
        updated_cp = graph_state_to_checkpoint(state, base_cp)
        await self.redis_service.save_checkpoint(updated_cp)

    async def _planner_node(self, state: ResearchGraphState) -> Dict[str, Any]:
        start_time = time.perf_counter()
        run_id = state["run_id"]
        step_key = "planner:v1"
        tracer = TraceRecorder(state["trace_steps"])
        budget = state["budget"]

        # Idempotency check: return cached result if node already completed
        if step_key in state["completed_node_keys"]:
            cached = await self.redis_service.get_cached_node_output(run_id, step_key)
            if cached and "plan" in cached:
                plan = ResearchPlan.model_validate(cached["plan"])
                statuses = {
                    k: SubQuestionStatus(v)
                    for k, v in cached["sub_question_statuses"].items()
                }
                tracer.record_step(
                    node_name="planner",
                    agent_name="PlannerAgent",
                    input_summary={"question": state["original_question"]},
                    output_summary={"sub_questions_count": len(plan.sub_questions)},
                    start_time=start_time,
                    cached_hit=True,
                )
                return {
                    "plan": plan,
                    "sub_question_statuses": statuses,
                    "status": RunStatus.RESEARCHING,
                    "current_stage": "planning_restored_from_cache",
                    "trace_steps": tracer.steps,
                }

        allowed, reason = self.budget_service.check_execution_allowed(budget)
        if not allowed:
            await self.redis_service.emit_event(
                run_id, "budget_exceeded", {"reason": reason}
            )
            return {
                "status": RunStatus.BUDGET_EXCEEDED,
                "error_message": reason,
                "budget": budget,
            }

        try:
            plan, in_tok, out_tok = await self.planner.create_plan(
                question=state["original_question"],
                max_sub_questions=budget.max_sub_questions,
            )
            cost = self.budget_service.record_llm_usage(budget, in_tok, out_tok)
            statuses = {sq.id: SubQuestionStatus.PENDING for sq in plan.sub_questions}

            await self.redis_service.set_cached_node_output(
                run_id,
                step_key,
                {
                    "plan": plan.model_dump(),
                    "sub_question_statuses": {k: v.value for k, v in statuses.items()},
                },
            )

            tracer.record_step(
                node_name="planner",
                agent_name="PlannerAgent",
                input_summary={"question": state["original_question"]},
                output_summary={
                    "objective": plan.objective_summary,
                    "sub_questions": [sq.id for sq in plan.sub_questions],
                },
                start_time=start_time,
                input_tokens=in_tok,
                output_tokens=out_tok,
                step_cost_usd=cost,
            )

            completed_keys = list(state["completed_node_keys"]) + [step_key]
            updates: Dict[str, Any] = {
                "plan": plan,
                "sub_question_statuses": statuses,
                "status": RunStatus.RESEARCHING,
                "current_stage": "planning_completed",
                "budget": budget,
                "trace_steps": tracer.steps,
                "completed_node_keys": completed_keys,
            }
            merged = {**state, **updates}
            base_cp = await self.redis_service.load_checkpoint(run_id)
            if base_cp:
                await self._persist_state(merged, base_cp)  # type: ignore[arg-type]

            await self.redis_service.emit_event(
                run_id,
                "planning_completed",
                {
                    "sub_questions": [
                        {"id": sq.id, "question": sq.question}
                        for sq in plan.sub_questions
                    ]
                },
            )
            return updates

        except Exception as exc:
            logger.error("planner_failed", run_id=run_id, error=str(exc))
            tracer.record_step(
                node_name="planner",
                agent_name="PlannerAgent",
                input_summary={"question": state["original_question"]},
                output_summary={},
                start_time=start_time,
                error=str(exc),
            )
            await self.redis_service.emit_event(
                run_id, "execution_failed", {"stage": "planner", "error": str(exc)}
            )
            return {
                "status": RunStatus.FAILED,
                "error_message": f"Planner failure: {exc}",
                "trace_steps": tracer.steps,
            }

    async def _select_sub_question_node(
        self, state: ResearchGraphState
    ) -> Dict[str, Any]:
        run_id = state["run_id"]
        budget = state["budget"]
        plan = state["plan"]

        allowed, reason = self.budget_service.check_execution_allowed(budget)
        if not allowed:
            # Transition to writer so partial findings still produce a cited report
            await self.redis_service.emit_event(
                run_id,
                "budget_exceeded",
                {"reason": reason, "action": "proceeding_to_writer"},
            )
            return {
                "active_sub_question_id": None,
                "status": RunStatus.BUDGET_EXCEEDED,
                "error_message": reason,
                "current_stage": "budget_capped_before_research",
                "budget": budget,
            }

        next_sq_id = None
        if plan:
            for sq in plan.sub_questions:
                st = state["sub_question_statuses"].get(sq.id)
                if st in (SubQuestionStatus.PENDING, SubQuestionStatus.NEEDS_REVISION):
                    next_sq_id = sq.id
                    break

        statuses = dict(state["sub_question_statuses"])
        if next_sq_id:
            statuses[next_sq_id] = SubQuestionStatus.IN_PROGRESS

        updates: Dict[str, Any] = {
            "active_sub_question_id": next_sq_id,
            "sub_question_statuses": statuses,
            "current_stage": (
                f"selected_{next_sq_id}" if next_sq_id else "all_sub_questions_processed"
            ),
        }
        base_cp = await self.redis_service.load_checkpoint(run_id)
        if base_cp:
            await self._persist_state({**state, **updates}, base_cp)  # type: ignore[arg-type]
        return updates

    async def _researcher_node(self, state: ResearchGraphState) -> Dict[str, Any]:
        start_time = time.perf_counter()
        run_id = state["run_id"]
        sq_id = state["active_sub_question_id"]
        plan = state["plan"]
        budget = state["budget"]
        tracer = TraceRecorder(state["trace_steps"])

        if not plan or not sq_id:
            return {"status": RunStatus.FAILED, "error_message": "Missing active sub-question"}

        sub_question = next(sq for sq in plan.sub_questions if sq.id == sq_id)
        pass_num = state["revisions_by_sq"].get(sq_id, 0) + 1
        step_key = f"researcher:{sq_id}:pass_{pass_num}"

        # Idempotent replay check
        if step_key in state["completed_node_keys"]:
            cached = await self.redis_service.get_cached_node_output(run_id, step_key)
            if cached is not None:
                restored_findings = [
                    FindingRecord.model_validate(f) for f in cached.get("findings", [])
                ]
                restored_sources = [
                    SourceMetadata.model_validate(s) for s in cached.get("sources", [])
                ]
                tracer.record_step(
                    node_name="researcher",
                    agent_name="ResearcherAgent",
                    input_summary={"sub_question_id": sq_id, "pass": pass_num},
                    output_summary={"restored_findings": len(restored_findings)},
                    start_time=start_time,
                    cached_hit=True,
                )
                return {
                    "findings": list(state["findings"]) + restored_findings,
                    "sources": list(state["sources"]) + restored_sources,
                    "current_stage": f"research_restored_{sq_id}_pass_{pass_num}",
                    "trace_steps": tracer.steps,
                }

        await self.redis_service.emit_event(
            run_id,
            "research_started",
            {"sub_question_id": sq_id, "question": sub_question.question, "pass": pass_num},
        )

        override_queries = state["refined_queries_by_sq"].get(sq_id) if pass_num > 1 else None

        exec_res = await self.researcher.investigate(
            sub_question=sub_question,
            budget=budget,
            existing_sources=state["sources"],
            existing_finding_count=len(state["findings"]),
            override_queries=override_queries,
        )

        step_cost = self.budget_service.calculate_cost(
            exec_res.input_tokens, exec_res.output_tokens
        )
        all_findings = list(state["findings"]) + exec_res.new_findings
        all_sources = list(state["sources"]) + exec_res.new_sources

        await self.redis_service.set_cached_node_output(
            run_id,
            step_key,
            {
                "findings": [f.model_dump() for f in exec_res.new_findings],
                "sources": [s.model_dump() for s in exec_res.new_sources],
            },
        )

        tracer.record_step(
            node_name="researcher",
            agent_name="ResearcherAgent",
            input_summary={
                "sub_question_id": sq_id,
                "pass": pass_num,
                "queries": override_queries or sub_question.search_queries,
            },
            output_summary={
                "new_findings_count": len(exec_res.new_findings),
                "new_sources_count": len(exec_res.new_sources),
                "budget_halted": exec_res.budget_halted,
            },
            start_time=start_time,
            tool_calls=exec_res.tool_calls,
            input_tokens=exec_res.input_tokens,
            output_tokens=exec_res.output_tokens,
            step_cost_usd=step_cost,
            error=exec_res.halt_reason,
        )

        new_status = (
            RunStatus.BUDGET_EXCEEDED
            if exec_res.budget_halted and budget.exceeded_reason
            else RunStatus.REVIEWING
        )
        completed_keys = list(state["completed_node_keys"]) + [step_key]
        updates: Dict[str, Any] = {
            "findings": all_findings,
            "sources": all_sources,
            "budget": budget,
            "status": new_status,
            "current_stage": f"research_completed_{sq_id}_pass_{pass_num}",
            "trace_steps": tracer.steps,
            "completed_node_keys": completed_keys,
            "error_message": exec_res.halt_reason or state.get("error_message"),
        }

        base_cp = await self.redis_service.load_checkpoint(run_id)
        if base_cp:
            await self._persist_state({**state, **updates}, base_cp)  # type: ignore[arg-type]

        await self.redis_service.emit_event(
            run_id,
            "findings_collected",
            {
                "sub_question_id": sq_id,
                "new_findings": len(exec_res.new_findings),
                "total_findings": len(all_findings),
            },
        )
        return updates

    async def _supervisor_review_node(
        self, state: ResearchGraphState
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        run_id = state["run_id"]
        sq_id = state["active_sub_question_id"]
        plan = state["plan"]
        budget = state["budget"]
        tracer = TraceRecorder(state["trace_steps"])

        if not plan or not sq_id:
            return {"status": RunStatus.FAILED, "error_message": "Supervisor missing active sub-question"}

        sub_question = next(sq for sq in plan.sub_questions if sq.id == sq_id)
        revs_used = state["revisions_by_sq"].get(sq_id, 0)
        step_key = f"supervisor:{sq_id}:review_{revs_used + 1}"

        decision, in_tok, out_tok = await self.supervisor.review_sub_question(
            sub_question=sub_question,
            findings=state["findings"],
            sources=state["sources"],
            revisions_used_for_sq=revs_used,
            max_revisions=budget.max_revision_attempts,
        )
        cost = self.budget_service.record_llm_usage(budget, in_tok, out_tok)

        statuses = dict(state["sub_question_statuses"])
        revisions_map = dict(state["revisions_by_sq"])
        refined_map = dict(state["refined_queries_by_sq"])

        if decision.decision == SupervisorDecisionType.REVISE:
            revisions_map[sq_id] = revs_used + 1
            statuses[sq_id] = SubQuestionStatus.NEEDS_REVISION
            if decision.refined_queries:
                refined_map[sq_id] = decision.refined_queries
        elif decision.decision == SupervisorDecisionType.APPROVE:
            statuses[sq_id] = SubQuestionStatus.COMPLETED
        else:
            statuses[sq_id] = SubQuestionStatus.FAILED

        decisions_list: List[SupervisorDecision] = list(state["supervisor_decisions"]) + [
            decision
        ]

        tracer.record_step(
            node_name="supervisor_review",
            agent_name="SupervisorAgent",
            input_summary={"sub_question_id": sq_id, "revisions_used": revs_used},
            output_summary={
                "decision": decision.decision.value,
                "reasoning": decision.reasoning,
                "refined_queries": decision.refined_queries,
            },
            start_time=start_time,
            input_tokens=in_tok,
            output_tokens=out_tok,
            step_cost_usd=cost,
        )

        completed_keys = list(state["completed_node_keys"]) + [step_key]
        updates: Dict[str, Any] = {
            "sub_question_statuses": statuses,
            "revisions_by_sq": revisions_map,
            "refined_queries_by_sq": refined_map,
            "supervisor_decisions": decisions_list,
            "budget": budget,
            "status": RunStatus.RESEARCHING,
            "current_stage": f"supervisor_reviewed_{sq_id}_{decision.decision.value}",
            "trace_steps": tracer.steps,
            "completed_node_keys": completed_keys,
        }

        base_cp = await self.redis_service.load_checkpoint(run_id)
        if base_cp:
            await self._persist_state({**state, **updates}, base_cp)  # type: ignore[arg-type]

        await self.redis_service.emit_event(
            run_id,
            "supervisor_reviewed",
            {
                "sub_question_id": sq_id,
                "decision": decision.decision.value,
                "reasoning": decision.reasoning,
            },
        )
        return updates

    async def _writer_node(self, state: ResearchGraphState) -> Dict[str, Any]:
        start_time = time.perf_counter()
        run_id = state["run_id"]
        plan = state["plan"]
        budget = state["budget"]
        tracer = TraceRecorder(state["trace_steps"])

        if not plan:
            return {"status": RunStatus.FAILED, "error_message": "Writer missing research plan"}

        await self.redis_service.emit_event(
            run_id,
            "writing_started",
            {"findings_count": len(state["findings"])},
        )

        try:
            report, in_tok, out_tok = await self.writer.write_report(
                plan=plan,
                findings=state["findings"],
                conflicts=state["conflicts"],
            )
            cost = self.budget_service.record_llm_usage(budget, in_tok, out_tok)

            tracer.record_step(
                node_name="writer",
                agent_name="WriterAgent",
                input_summary={"findings_count": len(state["findings"])},
                output_summary={
                    "cited_finding_ids": report.cited_finding_ids,
                    "invalid_citations_stripped": report.invalid_citations_stripped,
                },
                start_time=start_time,
                input_tokens=in_tok,
                output_tokens=out_tok,
                step_cost_usd=cost,
            )

            final_status = (
                RunStatus.BUDGET_EXCEEDED
                if state["status"] == RunStatus.BUDGET_EXCEEDED
                else RunStatus.COMPLETED
            )

            updates: Dict[str, Any] = {
                "report": report,
                "budget": budget,
                "status": final_status,
                "current_stage": "report_completed",
                "trace_steps": tracer.steps,
                "completed_node_keys": list(state["completed_node_keys"]) + ["writer:v1"],
            }

            base_cp = await self.redis_service.load_checkpoint(run_id)
            if base_cp:
                await self._persist_state({**state, **updates}, base_cp)  # type: ignore[arg-type]

            await self.redis_service.emit_event(
                run_id,
                "report_completed",
                {
                    "status": final_status.value,
                    "cited_count": len(report.cited_finding_ids),
                    "total_cost_usd": budget.estimated_cost_usd,
                },
            )
            return updates

        except Exception as exc:
            logger.error("writer_failed", run_id=run_id, error=str(exc))
            tracer.record_step(
                node_name="writer",
                agent_name="WriterAgent",
                input_summary={"findings_count": len(state["findings"])},
                output_summary={},
                start_time=start_time,
                error=str(exc),
            )
            await self.redis_service.emit_event(
                run_id, "execution_failed", {"stage": "writer", "error": str(exc)}
            )
            return {
                "status": RunStatus.FAILED,
                "error_message": f"Writer failure: {exc}",
                "trace_steps": tracer.steps,
            }

    async def execute_run(self, checkpoint: RunCheckpoint) -> ResearchGraphState:
        """Executes or resumes a research run from a Redis checkpoint."""
        initial_state = checkpoint_to_graph_state(checkpoint)
        final_state = await self.graph.ainvoke(initial_state)
        return final_state  # type: ignore[return-value]