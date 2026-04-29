from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel

from app.agents.advisor import AdvisorAgent
from app.agents.analyst import AnalystAgent
from app.agents.client import ClientAgent
from app.chat_handler import new_session_id, run_client_turn
from app.graph.workflow import build_workflow
from app.knowledge.chroma_store import ChromaKnowledgeStore, seed_default_knowledge
from app.logging_config import configure_logging
from app.storage.db import init_db


async def run_chat() -> None:
    console = Console()
    client = ClientAgent()
    advisor = AdvisorAgent()
    client_agent = ClientAgent(profile=client.profile)
    store = ChromaKnowledgeStore()
    seed_default_knowledge(store)
    analyst = AnalystAgent(store)
    app = build_workflow(advisor, analyst, client_agent)

    session_id = new_session_id()

    console.print(Panel.fit("Multi-Agent Investment Advisor (type 'exit' to quit)"))
    console.print(f"[bold]Client profile:[/bold] {client.profile.model_dump_json(indent=2)}")

    while True:
        user_input = input("\nClient> ").strip()
        if not user_input:
            user_input = client.next_prompt()
            console.print(f"[dim]Auto client prompt:[/dim] {user_input}")
        if user_input.lower() in {"exit", "quit"}:
            break

        result = await run_client_turn(
            session_id=session_id,
            user_input=user_input,
            client_profile=client.profile,
            store=store,
            workflow=app,
            prompt_client=client,
        )
        response = result["advisor"]
        if response is None:
            continue
        report = result.get("analyst") or {}
        findings = report.get("findings") or []

        console.print(Panel(response["message"], title="Advisor"))
        console.print(Panel(response["recommendation"], title="Recommendation"))
        console.print(
            Panel(
                "\n".join(f"- {r['source']}: {r['snippet']}" for r in findings) or "- No evidence found.",
                title="Analyst Evidence",
            )
        )
        if "agent_loops" in result:
            al = result["agent_loops"]
            console.print(
                f"[dim]Agent-loop rounds: {al['rounds']}, resolved={al['resolved']}[/dim]"
            )


def main() -> None:
    configure_logging()
    init_db()
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
