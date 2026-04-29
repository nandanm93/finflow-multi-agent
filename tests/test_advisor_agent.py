import pytest

from app.agents.advisor import AdvisorAgent
from app.agents.client import ClientAgent
from app.models.schemas import AnalystReport, EvidenceItem


def _report() -> AnalystReport:
    return AnalystReport(
        task_id="t1",
        findings=[EvidenceItem(source="kb", snippet="Diversify and rebalance periodically.", confidence=0.8)],
        summary="Diversification and rebalancing can improve risk-adjusted stability.",
    )


@pytest.mark.asyncio
async def test_advisor_respond_async_falls_back_and_keeps_metrics():
    advisor = AdvisorAgent()
    advisor._advisor_provider = "disabled"
    client = ClientAgent()

    out = await advisor.respond_async(
        client_message="What should I do next?",
        client_profile=client.profile,
        report=_report(),
        client_intent="general",
    )
    assert "Expected return" in out.recommendation
    assert "Sharpe" in out.recommendation


@pytest.mark.asyncio
async def test_advisor_respond_async_respects_client_intent():
    advisor = AdvisorAgent()
    advisor._advisor_provider = "disabled"
    client = ClientAgent()

    out = await advisor.respond_async(
        client_message="Anything",
        client_profile=client.profile,
        report=_report(),
        client_intent="legal_risks",
    )
    assert "legal" in out.recommendation.lower()
    assert any("legal" in note.lower() for note in out.risk_notes)
