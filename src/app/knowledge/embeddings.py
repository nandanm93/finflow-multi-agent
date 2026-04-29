from __future__ import annotations

from functools import lru_cache

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    return SentenceTransformer(settings.embedding_model, device="cpu")


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()
