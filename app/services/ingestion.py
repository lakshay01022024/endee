"""
Document ingestion pipeline.

Handles text chunking, embedding generation, and vector upsert
into the Endee vector database.
"""

import hashlib
import logging
from app.core.embeddings import embed_batch
from app.core.vector_store import upsert_vectors
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks using a sliding window.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size * 0.3:
                    end = start + last_sep + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def _generate_chunk_id(title: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a document chunk."""
    raw = f"{title}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def ingest_document(
    title: str,
    content: str,
    source: str = "manual",
    category: str = "general",
) -> dict:
    """
    Full ingestion pipeline: chunk → embed → upsert to Endee.

    Args:
        title: Document title.
        content: Full document text.
        source: Source identifier.
        category: Category label.

    Returns:
        Dict with ingestion results (chunks_created, document_ids).
    """
    logger.info(f"Ingesting document: '{title}' ({len(content)} chars)")

    # Step 1: Chunk the text
    chunks = chunk_text(content)
    logger.info(f"Created {len(chunks)} chunks")

    # Step 2: Generate embeddings for all chunks
    embeddings = embed_batch(chunks)
    logger.info(f"Generated {len(embeddings)} embeddings")

    # Step 3: Prepare upsert items with metadata
    items = []
    doc_ids = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc_id = _generate_chunk_id(title, i)
        doc_ids.append(doc_id)
        items.append({
            "id": doc_id,
            "vector": embedding,
            "meta": {
                "title": title,
                "content": chunk,
                "source": source,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        })

    # Step 4: Upsert into Endee
    upsert_vectors(items)
    logger.info(f"Successfully ingested '{title}' → {len(items)} vectors in Endee")

    return {
        "title": title,
        "chunks_created": len(chunks),
        "document_ids": doc_ids,
    }
