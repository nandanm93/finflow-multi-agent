from __future__ import annotations

import asyncio
import random

from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_config import get_logger
from app.models.schemas import AnalystReport, AssetPosition, ClientDecision, ClientProfile

logger = get_logger(__name__)


class ClientAgent:
    def __init__(self, profile: ClientProfile | None = None) -> None:
        self.settings = get_settings()
        self.profile = profile or self.generate_dummy_profile()
        self._client_provider, _, self._client_model = self.settings.client_llm.partition(":")
        self._client_provider = self._client_provider.strip().lower()
        self._client_model = self._client_model.strip() or "llama3.2:1b"
        self._ollama_client = AsyncOpenAI(
            base_url=self.settings.ollama_base_url,
            api_key="ollama",
        )

    @staticmethod
    def generate_dummy_profile() -> ClientProfile:
        return ClientProfile(
            name="Jordan Lee",
            age=38,
            risk_aversion="moderate",
            annual_income=155000,
            liquid_assets=420000,
            current_investments=[
                AssetPosition(
                    ticker="SPY",
                    allocation_pct=45,
                    expected_return=0.08,
                    volatility=0.16,
                ),
                AssetPosition(
                    ticker="BND",
                    allocation_pct=30,
                    expected_return=0.04,
                    volatility=0.06,
                ),
                AssetPosition(
                    ticker="QQQ",
                    allocation_pct=15,
                    expected_return=0.1,
                    volatility=0.22,
                ),
                AssetPosition(
                    ticker="CASH",
                    allocation_pct=10,
                    expected_return=0.02,
                    volatility=0.01,
                ),
            ],
            goals=["retire by 60", "fund child's college in 8 years", "maintain emergency buffer"],
        )

    def next_prompt(self) -> str:
        prompts = [
            "I want to improve my portfolio returns without taking extreme risk.",
            "Should I shift more into bonds given current uncertainty?",
            "Can I retire by 60 with this allocation?",
            "What should I do over the next 12 months?",
        ]
        return random.choice(prompts)

    @staticmethod
    def _heuristic_category(text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ("legal", "tax", "compliance", "regulation", "risk only")):
            return "legal_risks"
        if any(token in lowered for token in ("beginner", "basic", "new to", "simple", "first time")):
            return "beginner"
        if any(token in lowered for token in ("compare", "vs", "versus", "difference")):
            return "compare"
        return "general"

    async def parse_user_request(self, raw_message: str, profile: ClientProfile) -> dict[str, str]:
        category = self._heuristic_category(raw_message)
        if self._client_provider != "ollama":
            return {"category": category, "normalized_message": raw_message.strip() or self.next_prompt()}
        prompt = (
            "Classify and normalize this client request for an investment advisor workflow.\n"
            "Return exactly two lines:\n"
            "category=<one of: beginner, legal_risks, compare, general>\n"
            "normalized=<clean one-sentence request preserving client intent>\n\n"
            f"Client profile risk={profile.risk_aversion}; goals={', '.join(profile.goals)}\n"
            f"Request: {raw_message}\n"
        )
        try:
            response = await asyncio.wait_for(
                self._ollama_client.chat.completions.create(
                    model=self._client_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=120,
                ),
                timeout=self.settings.llm_timeout_seconds,
            )
            text = (response.choices[0].message.content or "").strip()
            out_category = category
            normalized = raw_message.strip() or self.next_prompt()
            for line in text.splitlines():
                if line.lower().startswith("category="):
                    parsed = line.split("=", 1)[1].strip().lower()
                    if parsed in {"beginner", "legal_risks", "compare", "general"}:
                        out_category = parsed
                if line.lower().startswith("normalized="):
                    candidate = line.split("=", 1)[1].strip()
                    if candidate:
                        normalized = candidate
            return {"category": out_category, "normalized_message": normalized}
        except Exception as exc:
            logger.warning("Client request parsing LLM failed; using heuristic fallback. err=%s", exc)
            return {"category": category, "normalized_message": raw_message.strip() or self.next_prompt()}

    def evaluate_response(
        self,
        original_message: str,
        advisor_message: str,
        recommendation: str,
        analyst_report: AnalystReport,
        client_loop_count: int,
        max_client_loops: int,
    ) -> ClientDecision:
        is_legal_risk_prompt = any(
            token in original_message.lower()
            for token in ("legal", "tax", "compliance", "regulation", "risk only")
        )
        if client_loop_count >= max_client_loops:
            return ClientDecision(
                resolved=True,
                rationale="Reached max client-agent loop count.",
            )
        if len(analyst_report.findings) < 2:
            return ClientDecision(
                resolved=False,
                follow_up_message="Can you provide more evidence and concrete allocation steps?",
                rationale="Evidence density is low.",
            )
        if (not is_legal_risk_prompt) and ("Expected return" not in recommendation or "Sharpe" not in recommendation):
            return ClientDecision(
                resolved=False,
                follow_up_message="Please include clearer risk-adjusted metrics for this advice.",
                rationale="Risk-return metrics missing from recommendation.",
            )
        if "rebalanc" not in advisor_message.lower() and "allocation" not in advisor_message.lower():
            return ClientDecision(
                resolved=False,
                follow_up_message=(
                    f"For my question '{original_message}', what exact rebalancing actions should I take?"
                ),
                rationale="Advisor response lacks concrete action framing.",
            )
        return ClientDecision(resolved=True, rationale="Advice appears actionable and sufficiently supported.")
