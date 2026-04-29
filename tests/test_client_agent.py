import pytest

from app.agents.client import ClientAgent


@pytest.mark.asyncio
async def test_parse_user_request_uses_heuristic_without_ollama():
    agent = ClientAgent()
    agent._client_provider = "disabled"
    parsed = await agent.parse_user_request("Give legal/finance risks only", agent.profile)
    assert parsed["category"] == "legal_risks"
    assert parsed["normalized_message"] == "Give legal/finance risks only"


@pytest.mark.asyncio
async def test_parse_user_request_parses_llm_structured_output(monkeypatch):
    agent = ClientAgent()
    agent._client_provider = "ollama"

    class _Msg:
        content = "category=beginner\nnormalized=Give beginner advice for Jordan Lee."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    async def _fake_create(**kwargs):
        return _Resp()

    monkeypatch.setattr(agent._ollama_client.chat.completions, "create", _fake_create)
    parsed = await agent.parse_user_request("help me start investing", agent.profile)
    assert parsed["category"] == "beginner"
    assert parsed["normalized_message"] == "Give beginner advice for Jordan Lee."
