from app.knowledge.chroma_store import ChromaKnowledgeStore


def test_chroma_query_returns_seeded_docs(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    import app.knowledge.chroma_store as chroma_store

    monkeypatch.setattr(
        chroma_store,
        "embed_texts",
        lambda texts: [[float(i + 1), float(i + 2), float(i + 3)] for i, _ in enumerate(texts)],
    )

    store = ChromaKnowledgeStore(collection_name="test_collection")
    store.upsert_documents(
        ["Diversification lowers single-stock risk.", "Bonds can reduce volatility."],
        source="unit_test",
    )
    rows = store.query("How can I reduce portfolio volatility?", n_results=2)
    assert rows
    assert rows[0]["source"] == "unit_test"
    assert "snippet" in rows[0]


def test_chroma_upsert_is_idempotent_with_deterministic_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    import app.knowledge.chroma_store as chroma_store

    monkeypatch.setattr(
        chroma_store,
        "embed_texts",
        lambda texts: [[0.1, 0.2, 0.3] for _ in texts],
    )

    store = ChromaKnowledgeStore(collection_name="idempotent_collection")
    docs = ["A stable allocation can reduce drawdown.", "Rebalance on schedule."]
    store.upsert_documents(docs, source="unit_test")
    count_first = store.collection.count()
    store.upsert_documents(docs, source="unit_test")
    count_second = store.collection.count()

    assert count_first == 2
    assert count_second == 2


def test_session_memory_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    import app.knowledge.chroma_store as chroma_store

    monkeypatch.setattr(
        chroma_store,
        "embed_texts",
        lambda texts: [[0.7, 0.8, 0.9] for _ in texts],
    )

    store = ChromaKnowledgeStore(collection_name="memory_collection")
    session_id = "session-123"
    store.upsert_session_turn(
        session_id=session_id,
        turn_index=0,
        client_message="I want to retire early.",
        advisor_response="Increase long-term equity exposure moderately.",
        analyst_summary="Focus on horizon-aligned risk.",
    )
    results = store.query_session_memory(session_id=session_id, text="retire early", n_results=1)
    assert len(results) == 1
    assert results[0]["source"] == "session_memory"
    assert "Client: I want to retire early." in results[0]["snippet"]


def test_session_memory_prefers_more_recent_turns(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))

    import app.knowledge.chroma_store as chroma_store

    monkeypatch.setattr(
        chroma_store,
        "embed_texts",
        lambda texts: [[0.4, 0.5, 0.6] for _ in texts],
    )

    store = ChromaKnowledgeStore(collection_name="memory_recency_collection")
    session_id = "session-456"
    store.upsert_session_turn(
        session_id=session_id,
        turn_index=0,
        client_message="I want lower volatility.",
        advisor_response="Increase bonds.",
        analyst_summary="Initial conservative direction.",
    )
    store.upsert_session_turn(
        session_id=session_id,
        turn_index=3,
        client_message="I can accept slightly more equity now.",
        advisor_response="Reintroduce selective equity exposure.",
        analyst_summary="Updated risk tolerance.",
    )
    results = store.query_session_memory(session_id=session_id, text="equity and volatility", n_results=2)
    assert len(results) == 2
    assert results[0]["turn_index"] >= results[1]["turn_index"]
