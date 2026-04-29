from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.advisor import AdvisorAgent
from app.agents.analyst import AnalystAgent
from app.agents.client import ClientAgent
from app.models.schemas import AdvisorResponse, AnalystReport, AnalystTask, ClientDecision, ClientProfile


class WorkflowState(TypedDict, total=False):
    session_id: str
    client_profile: ClientProfile
    client_message: str
    raw_client_message: str
    normalized_client_message: str
    client_intent: str
    loop_count: int
    max_loops: int
    client_loop_count: int
    max_client_loops: int
    client_decision: ClientDecision
    resolved: bool
    analyst_task: AnalystTask
    analyst_report: AnalystReport
    analyst_tasks: list[AnalystTask]
    analyst_reports: list[AnalystReport]
    advisor_response: AdvisorResponse


def build_workflow(advisor: AdvisorAgent, analyst: AnalystAgent, client: ClientAgent):
    graph = StateGraph(WorkflowState)

    def define_task(state: WorkflowState) -> WorkflowState:
        state["loop_count"] = 0
        state["max_loops"] = state.get("max_loops", 1)
        state["client_loop_count"] = state.get("client_loop_count", 0)
        state["max_client_loops"] = state.get("max_client_loops", 1)
        tasks = advisor.create_tasks(
            state["client_message"],
            state["client_profile"],
            session_id=state.get("session_id"),
        )
        state["analyst_tasks"] = tasks
        state["analyst_task"] = tasks[0]
        return state

    async def analyst_step(state: WorkflowState) -> WorkflowState:
        tasks = state.get("analyst_tasks", [])
        reports = await asyncio.gather(*(analyst.run(task) for task in tasks))
        state["analyst_reports"] = reports
        if reports:
            merged_findings = []
            for rep in reports:
                merged_findings.extend(rep.findings)
            merged_summary = " | ".join(rep.summary for rep in reports if rep.summary.strip())
            aggregate = AnalystReport(
                task_id=tasks[0].task_id if tasks else "aggregate",
                findings=merged_findings,
                summary=merged_summary or "No evidence found.",
            )
            state["analyst_report"] = aggregate
        else:
            state["analyst_report"] = AnalystReport(task_id="aggregate", findings=[], summary="No evidence found.")
        return state

    async def advisor_step(state: WorkflowState) -> WorkflowState:
        state["advisor_response"] = await advisor.respond_async(
            state["client_message"],
            state["client_profile"],
            state["analyst_report"],
            client_intent=state.get("client_intent"),
        )
        return state

    def refine_task(state: WorkflowState) -> WorkflowState:
        current_loop = state.get("loop_count", 0) + 1
        state["loop_count"] = current_loop
        refined = advisor.refine_tasks(
            previous_tasks=state.get("analyst_tasks", []),
            report=state["analyst_report"],
            loop_count=current_loop,
        )
        state["analyst_tasks"] = refined
        if refined:
            state["analyst_task"] = refined[0]
        return state

    def route_after_advisor(state: WorkflowState) -> str:
        if advisor.should_continue_loop(
            report=state["analyst_report"],
            loop_count=state.get("loop_count", 0),
            max_loops=state.get("max_loops", 2),
        ):
            return "refine_task"
        return "client_review"

    def client_review(state: WorkflowState) -> WorkflowState:
        decision = client.evaluate_response(
            original_message=state["client_message"],
            advisor_message=state["advisor_response"].message,
            recommendation=state["advisor_response"].recommendation,
            analyst_report=state["analyst_report"],
            client_loop_count=state.get("client_loop_count", 0),
            max_client_loops=state.get("max_client_loops", 1),
        )
        state["client_decision"] = decision
        state["resolved"] = decision.resolved
        return state

    def prepare_follow_up(state: WorkflowState) -> WorkflowState:
        state["client_loop_count"] = state.get("client_loop_count", 0) + 1
        follow_up = state["client_decision"].follow_up_message or state["client_message"]
        state["client_message"] = follow_up
        state["loop_count"] = 0
        return state

    def route_after_client_review(state: WorkflowState) -> str:
        if state.get("resolved", False):
            return END
        return "prepare_follow_up"

    graph.add_node("define_task", define_task)
    graph.add_node("analyst_step", analyst_step)
    graph.add_node("advisor_step", advisor_step)
    graph.add_node("refine_task", refine_task)
    graph.add_node("client_review", client_review)
    graph.add_node("prepare_follow_up", prepare_follow_up)

    graph.add_edge(START, "define_task")
    graph.add_edge("define_task", "analyst_step")
    graph.add_edge("analyst_step", "advisor_step")
    graph.add_conditional_edges(
        "advisor_step",
        route_after_advisor,
        {"refine_task": "refine_task", "client_review": "client_review"},
    )
    graph.add_edge("refine_task", "analyst_step")
    graph.add_conditional_edges(
        "client_review",
        route_after_client_review,
        {"prepare_follow_up": "prepare_follow_up", END: END},
    )
    graph.add_edge("prepare_follow_up", "define_task")

    return graph.compile()
