#!/usr/bin/env python3
"""
Seed script — Load sample documents into the Endee knowledge base.

Usage:
    python scripts/seed_data.py

Requires:
    - Endee server running (docker compose up -d)
    - Python dependencies installed (pip install -r requirements.txt)
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import get_settings
from app.core.vector_store import init_index
from app.core.embeddings import get_embedding_dimension
from app.services.ingestion import ingest_document


def main():
    """Load sample documents into the Endee knowledge base."""
    settings = get_settings()

    print("=" * 60)
    print("  Endee AI Knowledge Base — Seed Script")
    print("=" * 60)
    print()

    # Step 1: Initialize Endee index
    print(f"[1/3] Initializing Endee index '{settings.index_name}'...")
    dim = get_embedding_dimension()
    print(f"       Embedding dimension: {dim}")

    try:
        init_index(dimension=dim)
        print(f"       ✓ Index ready")
    except Exception as e:
        print(f"       ✗ Failed to initialize index: {e}")
        print(f"       Make sure Endee is running: docker compose up -d")
        sys.exit(1)

    # Step 2: Load sample documents
    data_file = Path(__file__).parent.parent / "data" / "sample_documents.json"
    print(f"\n[2/3] Loading documents from {data_file.name}...")

    if not data_file.exists():
        print(f"       ✗ File not found: {data_file}")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"       Found {len(documents)} documents")

    # Step 3: Ingest each document
    print(f"\n[3/3] Ingesting documents into Endee...\n")
    total_chunks = 0
    start_time = time.time()

    for i, doc in enumerate(documents, 1):
        title = doc["title"]
        content = doc["content"]
        source = doc.get("source", "sample")
        category = doc.get("category", "general")

        print(f"  [{i:2d}/{len(documents)}] {title[:50]}...", end=" ")

        try:
            result = ingest_document(
                title=title,
                content=content,
                source=source,
                category=category,
            )
            chunks = result["chunks_created"]
            total_chunks += chunks
            print(f"✓ ({chunks} chunks)")
        except Exception as e:
            print(f"✗ Error: {e}")

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print(f"  Done! Ingested {len(documents)} documents ({total_chunks} chunks)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Index: {settings.index_name}")
    print(f"  Endee: {settings.endee_base_url}")
    print("=" * 60)


if __name__ == "__main__":
    main()
