import pytest

from app.agents.analyst import AnalystAgent
from app.models.schemas import AnalystTask


class StubKnowledgeStore:
    def query(self, text: str, n_results: int = 3):
        return [
            {"snippet": "Diversify across sectors.", "source": "kb", "distance": 0.1},
            {"snippet": "Diversify across sectors.", "source": "kb", "distance": 0.1},
            {"snippet": "Maintain cash reserve.", "source": "kb", "distance": 0.2},
        ]


@pytest.mark.asyncio
async def test_analyst_dedupes_findings_and_builds_summary(monkeypatch):
    analyst = AnalystAgent(knowledge_store=StubKnowledgeStore())
    async def _fake_web_research(query: str):
        return []

    monkeypatch.setattr(analyst, "_web_research", _fake_web_research)
    task = AnalystTask(task_id="t1", question="How do I reduce portfolio risk?")
    report = await analyst.run(task)

    assert len(report.findings) == 2
    assert "RAG summary" in report.summary
