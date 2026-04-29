from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import get_settings
from app.knowledge.embeddings import embed_texts


class ChromaKnowledgeStore:
    def __init__(self, collection_name: str = "investment_knowledge") -> None:
        settings = get_settings()
        Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection: Collection = self.client.get_or_create_collection(collection_name)
        self.memory_collection: Collection = self.client.get_or_create_collection("session_memory")

    def upsert_documents(self, docs: list[str], source: str) -> None:
        if not docs:
            return
        embeddings = embed_texts(docs)
        ids = [hashlib.sha256(f"{source}:{doc}".encode("utf-8")).hexdigest() for doc in docs]
        metadatas = [{"source": source} for _ in docs]
        self.collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)

    def query(self, text: str, n_results: int = 3) -> list[dict]:
        result = self.collection.query(query_embeddings=embed_texts([text]), n_results=n_results)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows: list[dict] = []
        for doc, meta, dist in zip(docs, metas, distances):
            rows.append(
                {
                    "snippet": doc,
                    "source": (meta or {}).get("source", "knowledge_store"),
                    "distance": float(dist) if dist is not None else 0.0,
                }
            )
        return rows

    def upsert_session_turn(
        self,
        session_id: str,
        turn_index: int,
        client_message: str,
        advisor_response: str,
        analyst_summary: str,
    ) -> None:
        doc = (
            f"Turn {turn_index}\n"
            f"Client: {client_message}\n"
            f"Advisor: {advisor_response}\n"
            f"AnalystSummary: {analyst_summary}"
        )
        embedding = embed_texts([doc])[0]
        turn_id = hashlib.sha256(f"{session_id}:{turn_index}".encode("utf-8")).hexdigest()
        self.memory_collection.upsert(
            ids=[turn_id],
            documents=[doc],
            embeddings=[embedding],
            metadatas=[{"source": "session_memory", "session_id": session_id, "turn_index": turn_index}],
        )

    def query_session_memory(self, session_id: str, text: str, n_results: int = 2) -> list[dict]:
        fetch_k = max(n_results * 3, n_results)
        result = self.memory_collection.query(
            query_embeddings=embed_texts([text]),
            n_results=fetch_k,
            where={"session_id": session_id},
        )
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        max_turn_index = 0
        for meta in metas:
            if meta and isinstance(meta.get("turn_index"), int):
                max_turn_index = max(max_turn_index, int(meta["turn_index"]))
        rows: list[dict] = []
        for doc, meta, dist in zip(docs, metas, distances):
            turn_index = int((meta or {}).get("turn_index", 0))
            recency_boost = (turn_index + 1) / (max_turn_index + 1 if max_turn_index >= 0 else 1)
            adjusted_distance = float(dist) - 0.15 * recency_boost if dist is not None else 0.0
            rows.append(
                {
                    "snippet": doc,
                    "source": (meta or {}).get("source", "session_memory"),
                    "distance": adjusted_distance,
                    "raw_distance": float(dist) if dist is not None else 0.0,
                    "turn_index": turn_index,
                    "recency_boost": recency_boost,
                }
            )
        rows.sort(key=lambda row: row["distance"])
        return rows[:n_results]


def seed_default_knowledge(store: ChromaKnowledgeStore, path: str = "data/knowledge/market_primer.txt") -> None:
    if store.collection.count() > 0:
        return
    text = Path(path).read_text(encoding="utf-8")
    docs = [line.strip() for line in text.splitlines() if line.strip()]
    store.upsert_documents(docs=docs, source="market_primer")
