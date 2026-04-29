from __future__ import annotations

import asyncio
import re
from uuid import uuid4

from openai import AsyncOpenAI

from app.config import get_settings
from app.finance.metrics import portfolio_diagnostics
from app.logging_config import get_logger
from app.models.schemas import AdvisorResponse, AnalystReport, AnalystTask, ClientProfile

logger = get_logger(__name__)


class AdvisorAgent:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._advisor_provider, _, self._advisor_model = self.settings.advisor_llm.partition(":")
        self._advisor_provider = self._advisor_provider.strip().lower()
        self._advisor_model = self._advisor_model.strip() or "llama3.2:1b"
        self._ollama_client = AsyncOpenAI(
            base_url=self.settings.ollama_base_url,
            api_key="ollama",
        )

    @staticmethod
    def _detect_intent(client_message: str) -> str:
        text = client_message.lower()
        legal_risk_terms = (
            "legal",
            "regulation",
            "regulatory",
            "tax",
            "compliance",
            "lawsuit",
            "liability",
            "risk only",
        )
        beginner_terms = ("beginner", "new to", "starting", "simple", "basic", "first time")
        compare_terms = ("compare", "vs", "versus", "difference")

        if any(term in text for term in legal_risk_terms):
            return "legal_risks"
        if any(term in text for term in beginner_terms):
            return "beginner"
        if any(term in text for term in compare_terms):
            return "compare"
        return "general"

    def create_task(
        self, client_message: str, client_profile: ClientProfile, session_id: str | None = None
    ) -> AnalystTask:
        return AnalystTask(
            task_id=str(uuid4()),
            question=(
                f"Client message: {client_message}. "
                f"Profile age={client_profile.age}, risk={client_profile.risk_aversion}, "
                f"goals={', '.join(client_profile.goals)}. Return concise market and allocation guidance."
            ),
            required_depth="standard",
            session_id=session_id,
        )

    def create_tasks(
        self, client_message: str, client_profile: ClientProfile, session_id: str | None = None
    ) -> list[AnalystTask]:
        base_context = (
            f"Client message: {client_message}. "
            f"Profile age={client_profile.age}, risk={client_profile.risk_aversion}, "
            f"goals={', '.join(client_profile.goals)}."
        )
        return [
            AnalystTask(
                task_id=str(uuid4()),
                question=f"{base_context} Task type: research. Gather concise market and portfolio evidence.",
                required_depth="standard",
                session_id=session_id,
            ),
            AnalystTask(
                task_id=str(uuid4()),
                question=(
                    f"{base_context} Task type: recommendation. Propose practical rebalancing actions with caveats."
                ),
                required_depth="standard",
                session_id=session_id,
            ),
        ]

    @staticmethod
    def _default_risk_notes(intent: str) -> list[str]:
        if intent == "legal_risks":
            return [
                "Regulatory and tax rules vary by jurisdiction and account type.",
                "This is educational content, not legal or tax advice.",
                "Confirm final actions with a licensed tax/legal professional.",
            ]
        if intent == "beginner":
            return [
                "Start with small changes and track results over time.",
                "Avoid reacting to short-term market noise.",
                "Consider taxes and fees before making allocation changes.",
            ]
        if intent == "compare":
            return [
                "Choose based on timeline, drawdown tolerance, and liquidity needs.",
                "Reassess if your goals or income stability changes.",
                "Use staged rebalancing rather than one-time large shifts.",
            ]
        return [
            "This is educational guidance, not personalized fiduciary advice.",
            "Market conditions and rates can change expected outcomes.",
            "Rebalance gradually and validate tax impact before execution.",
        ]

    @staticmethod
    def _rule_based_recommendation(
        intent: str, objective_hint: str, metrics: dict[str, float]
    ) -> str:
        if intent == "legal_risks":
            return (
                f"For your question on '{objective_hint}', focus only on legal/financial risk controls: "
                "verify suitability documentation, model tax impact before trades, review concentration limits, "
                "and confirm disclosures and record-keeping obligations."
            )
        if intent == "beginner":
            return (
                "Beginner path: (1) keep an emergency buffer, (2) diversify with broad funds, "
                "(3) rebalance on a schedule, and (4) increase risk only gradually. "
                f"Current portfolio baseline: expected return ~{metrics['expected_return']:.2%}, "
                f"volatility ~{metrics['volatility']:.2%}."
            )
        if intent == "compare":
            return (
                "Comparison view: a conservative tilt lowers downside volatility, while a growth tilt may "
                "raise expected return but increases drawdown risk. "
                f"Baseline Sharpe estimate: ~{metrics['sharpe_ratio']:.2f}."
            )
        return (
            f"For your question on '{objective_hint}', use a balanced reallocation with risk-aware tilt. "
            f"Expected return ~{metrics['expected_return']:.2%}, "
            f"volatility ~{metrics['volatility']:.2%}, "
            f"Sharpe ~{metrics['sharpe_ratio']:.2f}."
        )

    @staticmethod
    def _profile_aware_legal_risks(client_profile: ClientProfile) -> str:
        equity_weight = sum(
            p.allocation_pct for p in client_profile.current_investments if p.ticker.upper() in {"SPY", "QQQ"}
        )
        concentration = ", ".join(
            f"{p.ticker}:{p.allocation_pct:.0f}%"
            for p in sorted(client_profile.current_investments, key=lambda x: x.allocation_pct, reverse=True)[:3]
        )
        retirement_goal = any("retire" in g.lower() for g in client_profile.goals)
        college_goal = any("college" in g.lower() for g in client_profile.goals)
        return (
            "Profile-specific legal/finance risks to review: "
            f"(1) Suitability/documentation risk: with risk profile '{client_profile.risk_aversion}' and top holdings "
            f"({concentration}), ensure recommendations are documented as suitable and consistent with stated goals. "
            f"(2) Tax-lot/capital-gains risk: any rebalance from current equity exposure (~{equity_weight:.0f}%) may "
            "trigger taxable events; model short-term vs long-term gains before execution. "
            f"(3) Goal-liability timing risk: goals include retirement={retirement_goal} and college={college_goal}; "
            "misaligned duration between assets and near/medium-term liabilities can create governance and planning risk. "
            f"(4) Liquidity/buffer risk: with liquid assets around ${client_profile.liquid_assets:,.0f}, keep emergency "
            "fund segregation documented so invested assets are not used for short-horizon needs. "
            "(5) Disclosure/process risk: record assumptions, constraints, and rebalancing policy to reduce compliance "
            "and client-communication risk."
        )

    async def _llm_draft_recommendation(
        self,
        *,
        intent: str,
        client_message: str,
        client_profile: ClientProfile,
        report: AnalystReport,
        metrics: dict[str, float],
    ) -> str:
        evidence = "\n".join(f"- {f.source}: {f.snippet[:180]}" for f in report.findings[:3]) or "- none"
        prompt = (
            "You are a financial advisor assistant. Draft a concise recommendation paragraph for the client.\n"
            "Constraints:\n"
            "- Keep it to 3-5 sentences.\n"
            "- Match intent exactly (beginner / legal_risks / compare / general).\n"
            "- Include these exact metrics once: "
            f"expected_return={metrics['expected_return']:.2%}, "
            f"volatility={metrics['volatility']:.2%}, "
            f"sharpe={metrics['sharpe_ratio']:.2f}.\n"
            "- Do not invent numbers.\n"
            "- For legal_risks intent, avoid giving allocation percentages.\n\n"
            f"Intent: {intent}\n"
            f"Client question: {client_message}\n"
            f"Risk profile: {client_profile.risk_aversion}\n"
            f"Goals: {', '.join(client_profile.goals)}\n"
            f"Analyst summary: {report.summary[:700]}\n"
            f"Top evidence:\n{evidence}\n"
        )
        response = await asyncio.wait_for(
            self._ollama_client.chat.completions.create(
                model=self._advisor_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=220,
            ),
            timeout=self.settings.llm_timeout_seconds,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _force_canonical_metrics_line(recommendation: str, metrics: dict[str, float]) -> str:
        metrics_line = (
            f"Expected return ~{metrics['expected_return']:.2%}, "
            f"volatility ~{metrics['volatility']:.2%}, "
            f"Sharpe ~{metrics['sharpe_ratio']:.2f}."
        )
        # Remove any model-invented metrics sentence before appending canonical values.
        cleaned = re.sub(
            r"Expected return\s*~[^.]*volatility\s*~[^.]*Sharpe\s*~[^.]*\.?",
            "",
            recommendation,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        if cleaned:
            return f"{cleaned} {metrics_line}".strip()
        return metrics_line

    async def respond_async(
        self,
        client_message: str,
        client_profile: ClientProfile,
        report: AnalystReport,
        client_intent: str | None = None,
    ) -> AdvisorResponse:
        metrics = portfolio_diagnostics(client_profile.current_investments)
        objective_hint = client_message.strip().rstrip("?")
        evidence_line = report.findings[0].snippet if report.findings else "Limited external evidence."
        intent = client_intent or self._detect_intent(client_message)
        fallback = self._rule_based_recommendation(intent, objective_hint, metrics)
        try:
            if intent == "legal_risks":
                recommendation = self._profile_aware_legal_risks(client_profile)
            elif self._advisor_provider == "ollama":
                recommendation = await self._llm_draft_recommendation(
                    intent=intent,
                    client_message=client_message,
                    client_profile=client_profile,
                    report=report,
                    metrics=metrics,
                )
            else:
                recommendation = fallback
        except Exception:
            logger.warning("Advisor LLM drafting failed; using fallback recommendation.")
            recommendation = fallback
        if not recommendation:
            recommendation = fallback
        if intent != "legal_risks":
            recommendation = self._force_canonical_metrics_line(recommendation, metrics)
        message = (
            f"Given your {client_profile.risk_aversion} risk profile and goals ({', '.join(client_profile.goals)}), "
            f"the analyst's RAG summary indicates: {report.summary} "
            f"Key supporting point: {evidence_line}"
        )
        return AdvisorResponse(
            message=message,
            recommendation=recommendation,
            risk_notes=self._default_risk_notes(intent),
        )

    def respond(
        self, client_message: str, client_profile: ClientProfile, report: AnalystReport
    ) -> AdvisorResponse:
        metrics = portfolio_diagnostics(client_profile.current_investments)
        objective_hint = client_message.strip().rstrip("?")
        intent = self._detect_intent(client_message)
        recommendation = self._rule_based_recommendation(intent, objective_hint, metrics)
        message = (
            f"Given your {client_profile.risk_aversion} risk profile and goals ({', '.join(client_profile.goals)}), "
            f"the analyst's RAG summary indicates: {report.summary}"
        )
        return AdvisorResponse(
            message=message,
            recommendation=recommendation,
            risk_notes=self._default_risk_notes(intent),
        )

    def should_continue_loop(self, report: AnalystReport, loop_count: int, max_loops: int = 2) -> bool:
        if loop_count >= max_loops:
            return False
        if len(report.findings) < 2:
            return True
        avg_confidence = sum(item.confidence for item in report.findings) / len(report.findings)
        return avg_confidence < 0.6

    def refine_task(
        self, previous_task: AnalystTask, report: AnalystReport, loop_count: int
    ) -> AnalystTask:
        follow_up = (
            f"{previous_task.question} "
            f"Refinement round {loop_count + 1}: Address gaps from prior summary: {report.summary}"
        )
        return AnalystTask(
            task_id=str(uuid4()),
            question=follow_up,
            required_depth="deep" if loop_count > 0 else "standard",
            session_id=previous_task.session_id,
        )

    def refine_tasks(
        self, previous_tasks: list[AnalystTask], report: AnalystReport, loop_count: int
    ) -> list[AnalystTask]:
        refined: list[AnalystTask] = []
        for task in previous_tasks:
            refined.append(
                AnalystTask(
                    task_id=str(uuid4()),
                    question=(
                        f"{task.question} "
                        f"Refinement round {loop_count + 1}: Address gaps from prior summary: {report.summary}"
                    ),
                    required_depth="deep" if loop_count > 0 else "standard",
                    session_id=task.session_id,
                )
            )
        return refined
