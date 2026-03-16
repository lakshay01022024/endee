"""
Embedding service using Sentence Transformers.

Wraps the `all-MiniLM-L6-v2` model (384-dimensional embeddings)
in a singleton pattern for efficient reuse across requests.
"""

import logging
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load and cache the embedding model."""
    global _model
    if _model is None:
        settings = get_settings()
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        logger.info(
            f"Model loaded — dimension: {_model.get_sentence_embedding_dimension()}"
        )
    return _model


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding vectors."""
    return _get_model().get_sentence_embedding_dimension()


def embed_text(text: str) -> list[float]:
    """
    Embed a single text string into a dense vector.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Embed a batch of texts into dense vectors.

    Args:
        texts: List of input texts to embed.
        batch_size: Number of texts to process in each batch.

    Returns:
        A list of embedding vectors.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
    )
    return [emb.tolist() for emb in embeddings]
