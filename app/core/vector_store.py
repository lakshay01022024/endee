"""
Endee Vector Database client wrapper.

Provides a clean interface to the Endee Python SDK for
creating indexes, upserting vectors, and running similarity queries.
"""

import logging
from endee import Endee, Precision
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_client: Endee | None = None
_index = None


def _get_client() -> Endee:
    """Lazy-load and cache the Endee client."""
    global _client
    if _client is None:
        settings = get_settings()
        token = settings.endee_auth_token if settings.endee_auth_token else ""
        _client = Endee(token)
        _client.set_base_url(settings.endee_base_url)
        logger.info(f"Endee client connected → {settings.endee_base_url}")
    return _client


def init_index(dimension: int = 384) -> None:
    """
    Create the knowledge_base index in Endee if it doesn't already exist.

    Args:
        dimension: Vector dimensionality (default 384 for all-MiniLM-L6-v2).
    """
    settings = get_settings()
    client = _get_client()

    try:
        client.create_index(
            name=settings.index_name,
            dimension=dimension,
            space_type="cosine",
            precision=Precision.INT8,
        )
        logger.info(
            f"Created Endee index '{settings.index_name}' "
            f"(dim={dimension}, cosine, INT8)"
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "already exists" in error_msg or "conflict" in error_msg:
            logger.info(f"Index '{settings.index_name}' already exists — reusing.")
        else:
            logger.error(f"Failed to create index: {e}")
            raise


def _get_index():
    """Get and cache the Endee index object."""
    global _index
    if _index is None:
        settings = get_settings()
        client = _get_client()
        _index = client.get_index(name=settings.index_name)
    return _index


def upsert_vectors(items: list[dict]) -> None:
    """
    Upsert vectors into the Endee index.

    Args:
        items: List of dicts with keys: "id" (str), "vector" (list[float]),
               "meta" (dict with metadata fields).

    Example:
        upsert_vectors([{
            "id": "doc1_chunk0",
            "vector": [0.1, 0.2, ...],
            "meta": {"title": "My Doc", "source": "upload", "chunk_index": 0}
        }])
    """
    index = _get_index()
    index.upsert(items)
    logger.info(f"Upserted {len(items)} vectors into Endee")


def search(
    vector: list[float],
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """
    Run a similarity search on the Endee index.

    Args:
        vector: Query embedding vector.
        top_k: Number of nearest neighbors to return.
        filters: Optional metadata filters.

    Returns:
        List of result dicts with "id", "score", and "meta" keys.
    """
    index = _get_index()

    results = index.query(
        vector=vector,
        top_k=top_k,
    )

    formatted = []
    for item in results:
        formatted.append({
            "id": item.get("id", ""),
            "score": item.get("similarity", item.get("score", 0.0)),
            "meta": item.get("meta", {}),
        })

    logger.info(f"Search returned {len(formatted)} results (top_k={top_k})")
    return formatted


def get_index_stats() -> dict:
    """
    Get basic index statistics for health checks.

    Returns:
        Dict with index name and status info.
    """
    settings = get_settings()
    try:
        _get_client()
        _get_index()
        return {
            "index_name": settings.index_name,
            "status": "connected",
            "endee_url": settings.endee_base_url,
        }
    except Exception as e:
        return {
            "index_name": settings.index_name,
            "status": "error",
            "error": str(e),
            "endee_url": settings.endee_base_url,
        }
