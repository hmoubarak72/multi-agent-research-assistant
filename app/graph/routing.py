from app.graph.state import ResearchGraphState
from app.models.research import SubQuestionStatus
from app.models.schemas import RunStatus


def route_after_planner(state: ResearchGraphState) -> str:
    if state["status"] in (RunStatus.FAILED, RunStatus.BUDGET_EXCEEDED):
        return "halt"
    if not state.get("plan") or not state["plan"].sub_questions:
        return "halt"
    return "select_sub_question"


def route_after_selector(state: ResearchGraphState) -> str:
    if state["status"] in (RunStatus.FAILED, RunStatus.BUDGET_EXCEEDED):
        return "halt"
    if state.get("active_sub_question_id"):
        return "researcher"
    return "writer"


def route_after_researcher(state: ResearchGraphState) -> str:
    if state["status"] == RunStatus.FAILED:
        return "halt"
    # Even if budget_exceeded occurred mid-research, let supervisor or writer
    # inspect collected findings unless hard-halted
    if state["status"] == RunStatus.BUDGET_EXCEEDED:
        return "writer"
    return "supervisor_review"


def route_after_supervisor(state: ResearchGraphState) -> str:
    if state["status"] == RunStatus.FAILED:
        return "halt"
    if state["status"] == RunStatus.BUDGET_EXCEEDED:
        return "writer"

    active_id = state.get("active_sub_question_id")
    if active_id:
        sq_status = state["sub_question_statuses"].get(active_id)
        if sq_status == SubQuestionStatus.NEEDS_REVISION:
            return "researcher"

    return "select_sub_question"