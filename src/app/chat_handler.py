from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.agents.client import ClientAgent
from app.knowledge.chroma_store import ChromaKnowledgeStore
from app.models.schemas import ClientProfile
from app.storage.db import get_session
from app.storage.repositories import add_analyst_report, add_turn, get_or_create_session, list_turns


def _dedupe_findings(state: dict) -> list[dict[str, str]]:
    report = state.get("analyst_report")
    if not report or not hasattr(report, "findings"):
        return []
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for finding in report.findings:
        key = (finding.source, finding.snippet.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "source": finding.source,
                "snippet": finding.snippet,
                "confidence": finding.confidence,
            }
        )
    return unique


def serialize_workflow_state(state: dict) -> dict[str, Any]:
    response = state.get("advisor_response")
    report = state.get("analyst_report")
    task = state.get("analyst_task")
    decision = state.get("client_decision")
    out: dict[str, Any] = {
        "advisor": response.model_dump(mode="json") if response is not None else None,
        "analyst": {
            "summary": report.summary,
            "findings": _dedupe_findings(state),
        }
        if report is not None
        else None,
        "analyst_task": task.model_dump(mode="json") if task is not None else None,
        "client_decision": decision.model_dump() if decision is not None and hasattr(decision, "model_dump") else None,
        "client_loop_count": state.get("client_loop_count"),
        "client_intent": state.get("client_intent"),
        "normalized_client_message": state.get("normalized_client_message"),
        "resolved": state.get("resolved"),
    }
    return out


async def run_client_turn(
    *,
    session_id: str,
    user_input: str,
    client_profile: ClientProfile,
    store: ChromaKnowledgeStore,
    workflow: Any,
    prompt_client: ClientAgent,
    max_loops: int = 1,
    max_client_loops: int = 1,
) -> dict[str, Any]:
    if not user_input.strip():
        user_input = prompt_client.next_prompt()
        out_of_band_prompt = user_input
    else:
        out_of_band_prompt = None
    parsed = await prompt_client.parse_user_request(user_input, client_profile)
    normalized_message = parsed.get("normalized_message", user_input).strip() or user_input
    client_intent = parsed.get("category", "general")

    state = await workflow.ainvoke(
        {
            "session_id": session_id,
            "client_profile": client_profile,
            "client_message": normalized_message,
            "raw_client_message": user_input,
            "normalized_client_message": normalized_message,
            "client_intent": client_intent,
            "max_loops": max_loops,
            "max_client_loops": max_client_loops,
        }
    )
    report = state["analyst_report"]
    response = state["advisor_response"]
    task = state["analyst_task"]

    with get_session() as db:
        turns = list_turns(db, session_id)
        turn_index = len(turns)
        session_row = get_or_create_session(db, session_id=session_id, profile=client_profile)
        turn_row = add_turn(
            db,
            session_row=session_row,
            turn_index=turn_index,
            client_message=user_input,
            advisor_response=response,
        )
        add_analyst_report(db, turn_row=turn_row, task=task, report=report)
        db.commit()

    store.upsert_session_turn(
        session_id=session_id,
        turn_index=turn_index,
        client_message=user_input,
        advisor_response=response.message,
        analyst_summary=report.summary,
    )

    payload = serialize_workflow_state(state)
    payload["session_id"] = session_id
    payload["turn_index"] = turn_index
    payload["client_message"] = user_input
    payload["normalized_client_message"] = normalized_message
    payload["client_intent"] = client_intent
    if out_of_band_prompt is not None:
        payload["auto_client_prompt"] = out_of_band_prompt
    if "client_loop_count" in state:
        payload["agent_loops"] = {
            "rounds": state["client_loop_count"] + 1,
            "resolved": state.get("resolved", False),
        }
    return payload


def new_session_id() -> str:
    return str(uuid4())
