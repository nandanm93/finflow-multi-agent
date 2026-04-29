import pytest

from app.agents.advisor import AdvisorAgent
pytest.importorskip("langgraph")
from app.graph.workflow import build_workflow
from app.models.schemas import AdvisorResponse, AnalystReport, EvidenceItem


class StubAnalyst:
    def __init__(self):
        self.calls = 0

    async def run(self, task):
        self.calls += 1
        return AnalystReport(
            task_id=task.task_id,
            findings=[EvidenceItem(source="stub", snippet="Diversify prudently.", confidence=0.8)],
            summary="Stub summary",
        )


@pytest.mark.asyncio
async def test_workflow_generates_advisor_response(monkeypatch):
    from app.agents.client import ClientAgent

    advisor = AdvisorAgent()
    seen_intents = []

    async def _fake_respond_async(client_message, client_profile, report, client_intent=None):
        seen_intents.append(client_intent)
        return AdvisorResponse(
            message="Use allocation rebalancing plan based on profile.",
            recommendation="Expected return ~6.50%, volatility ~8.12%, Sharpe ~0.55.",
            risk_notes=["Educational guidance only."],
        )

    monkeypatch.setattr(advisor, "respond_async", _fake_respond_async)
    analyst = StubAnalyst()
    client = ClientAgent()
    graph = build_workflow(advisor=advisor, analyst=analyst, client=client)

    out = await graph.ainvoke(
        {
            "client_profile": client.profile,
            "client_message": "How should I rebalance for moderate risk?",
            "client_intent": "general",
        }
    )
    assert out["analyst_task"].question
    assert out["analyst_report"].summary
    assert "Sharpe" in out["advisor_response"].recommendation
    assert analyst.calls >= 2
    assert seen_intents and seen_intents[0] == "general"
    assert out["resolved"] is True
