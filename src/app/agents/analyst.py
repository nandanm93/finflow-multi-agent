from __future__ import annotations

import asyncio
from collections.abc import Sequence

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from app.config import get_settings
from app.knowledge.chroma_store import ChromaKnowledgeStore
from app.logging_config import get_logger
from app.models.schemas import AnalystReport, AnalystTask, EvidenceItem

logger = get_logger(__name__)


class AnalystAgent:
    def __init__(self, knowledge_store: ChromaKnowledgeStore) -> None:
        self.settings = get_settings()
        self.knowledge_store = knowledge_store
        self.anthropic_client = AsyncAnthropic(api_key=self.settings.anthropic_api_key or None)
        if self.settings.openai_api_key:
            self._openai_client: AsyncOpenAI | None = AsyncOpenAI(api_key=self.settings.openai_api_key)
        else:
            self._openai_client = None
        self._ollama_client = AsyncOpenAI(
            base_url=self.settings.ollama_base_url,
            api_key="ollama",
        )
        self._genai_client = None
        if self.settings.gemini_api_key:
            from google import genai

            self._genai_client = genai.Client(api_key=self.settings.gemini_api_key)

    def _llm_candidates(self, purpose: str) -> list[str]:
        if purpose == "web":
            configured_specs = self.settings.analyst_web_llms
            legacy_models = self.settings.analyst_web_models
        else:
            configured_specs = self.settings.analyst_summary_llms
            legacy_models = self.settings.analyst_summary_models
        if configured_specs:
            return configured_specs
        if legacy_models:
            return [f"anthropic:{name}" for name in legacy_models]
        return [f"anthropic:{self.settings.claude_model}"]

    def _has_available_provider(self, purpose: str) -> bool:
        for llm_spec in self._llm_candidates(purpose):
            provider, _, _ = llm_spec.partition(":")
            provider = provider.strip().lower()
            if self._provider_is_available(provider):
                return True
        return False

    def _provider_is_available(self, provider: str) -> bool:
        if provider == "anthropic":
            return bool(self.settings.anthropic_api_key)
        if provider == "openai":
            return bool(self.settings.openai_api_key)
        if provider == "gemini":
            return bool(self.settings.gemini_api_key)
        if provider == "ollama":
            return True
        return False

    async def _call_anthropic(self, model: str, messages: Sequence[dict], max_tokens: int, tools: list[dict] | None):
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": list(messages)}
        if tools:
            kwargs["tools"] = tools
        response = await asyncio.wait_for(
            self.anthropic_client.messages.create(**kwargs),
            timeout=self.settings.llm_timeout_seconds,
        )
        text_blocks = [getattr(block, "text", "") for block in response.content]
        return " ".join(t.strip() for t in text_blocks if t.strip())

    async def _call_openai(self, model: str, messages: Sequence[dict], max_tokens: int):
        if self._openai_client is None:
            raise RuntimeError("OpenAI client is not configured (set OPENAI_API_KEY).")
        response = await asyncio.wait_for(
            self._openai_client.chat.completions.create(
                model=model,
                messages=list(messages),
                max_tokens=max_tokens,
            ),
            timeout=self.settings.llm_timeout_seconds,
        )
        return response.choices[0].message.content or ""

    async def _call_gemini(self, model: str, messages: Sequence[dict]) -> str:
        if self._genai_client is None:
            raise RuntimeError("Gemini client is not configured (set GEMINI_API_KEY).")
        prompt = "\n".join(str(m.get("content", "")) for m in messages)
        response = await asyncio.wait_for(
            self._genai_client.aio.models.generate_content(
                model=model,
                contents=prompt,
            ),
            timeout=self.settings.llm_timeout_seconds,
        )
        return getattr(response, "text", "") or ""

    async def _call_ollama(self, model: str, messages: Sequence[dict], max_tokens: int) -> str:
        response = await asyncio.wait_for(
            self._ollama_client.chat.completions.create(
                model=model,
                messages=list(messages),
                max_tokens=max_tokens,
            ),
            timeout=self.settings.llm_timeout_seconds,
        )
        return response.choices[0].message.content or ""

    async def _messages_create_with_fallback(
        self,
        purpose: str,
        *,
        max_tokens: int,
        messages: Sequence[dict],
        tools: list[dict] | None = None,
    ):
        last_error: Exception | None = None
        for llm_spec in self._llm_candidates(purpose):
            provider, _, model_name = llm_spec.partition(":")
            provider = provider.strip().lower()
            model_name = model_name.strip()
            if not self._provider_is_available(provider):
                continue
            try:
                if provider == "anthropic":
                    text = await self._call_anthropic(model_name, messages, max_tokens, tools)
                elif provider == "openai":
                    text = await self._call_openai(model_name, messages, max_tokens)
                elif provider == "gemini":
                    text = await self._call_gemini(model_name, messages)
                elif provider == "ollama":
                    text = await self._call_ollama(model_name, messages, max_tokens)
                else:
                    raise ValueError(f"Unsupported LLM provider '{provider}' in spec '{llm_spec}'.")
                return text
            except Exception as exc:  # pragma: no cover - provider/runtime dependent
                logger.warning("Analyst LLM provider failed: provider=%s model=%s err=%s", provider, model_name, exc)
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("No model candidates configured for analyst LLM call.")

    async def _web_research(self, query: str) -> list[EvidenceItem]:
        if not self._has_available_provider("web"):
            return []
        response = await self._messages_create_with_fallback(
            "web",
            max_tokens=500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[
                {
                    "role": "user",
                    "content": f"Find concise evidence for this investment question: {query}",
                }
            ],
        )
        output = []
        if response.strip():
            output.append(EvidenceItem(source="web_search", snippet=response[:400], confidence=0.65))
        return output[:3]

    def _kb_research(self, query: str) -> list[EvidenceItem]:
        rows = self.knowledge_store.query(query, n_results=3)
        return [
            EvidenceItem(
                source=row["source"],
                snippet=row["snippet"],
                confidence=max(0.2, min(0.9, 1.0 - row["distance"])),
            )
            for row in rows
        ]

    def _memory_research(self, session_id: str | None, query: str) -> list[EvidenceItem]:
        if not session_id:
            return []
        rows = self.knowledge_store.query_session_memory(session_id=session_id, text=query, n_results=2)
        return [
            EvidenceItem(
                source=row["source"],
                snippet=row["snippet"],
                confidence=max(
                    0.2,
                    min(
                        0.95,
                        (1.0 - row["raw_distance"]) * 0.75 + float(row.get("recency_boost", 0.0)) * 0.25,
                    ),
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _dedupe_findings(findings: list[EvidenceItem]) -> list[EvidenceItem]:
        deduped: list[EvidenceItem] = []
        seen: set[tuple[str, str]] = set()
        for item in findings:
            key = (item.source, item.snippet.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    async def _rag_summary(self, question: str, findings: list[EvidenceItem]) -> str:
        if not findings:
            return "No evidence found."
        context = "\n".join(f"- ({f.source}) {f.snippet}" for f in findings[:5])
        if not self._has_available_provider("summary"):
            return f"RAG summary (offline): {context[:350]}"
        response = await self._messages_create_with_fallback(
            "summary",
            max_tokens=220,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are an investment research analyst. "
                        "Synthesize the evidence into 2-3 concise sentences with clear caveats.\n\n"
                        f"Question:\n{question}\n\n"
                        f"Retrieved Context:\n{context}"
                    ),
                }
            ],
        )
        summary = response.strip()
        return summary or "No evidence found."

    async def run(self, task: AnalystTask) -> AnalystReport:
        kb = self._kb_research(task.question)
        memory = self._memory_research(task.session_id, task.question)
        web = await self._web_research(task.question)
        findings = self._dedupe_findings((memory + kb + web))[:5]
        summary = await self._rag_summary(task.question, findings)
        return AnalystReport(task_id=task.task_id, findings=findings, summary=summary)
