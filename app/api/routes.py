"""
API routes for the Endee AI Knowledge Base.

Endpoints:
  GET  /api/health   — Health check
  POST /api/ingest   — Ingest a document
  POST /api/search   — Semantic search
  POST /api/ask      — RAG-powered question answering
"""

import logging
from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceDocument,
)
from app.services.ingestion import ingest_document
from app.services import rag_engine
from app.core.vector_store import get_index_stats
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Knowledge Base"])


# ── Health Check ───────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health of the API, Endee connection, and loaded models.",
)
async def health_check():
    """Verify API, Endee database, and model status."""
    settings = get_settings()
    endee_stats = get_index_stats()

    return HealthResponse(
        status="healthy" if endee_stats.get("status") == "connected" else "degraded",
        endee=endee_stats,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )


# ── Document Ingestion ─────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Document",
    description="Ingest a document into the knowledge base. The document is chunked, "
    "embedded, and stored as vectors in Endee for later retrieval.",
)
async def ingest(request: IngestRequest):
    """Ingest a document: chunk → embed → store in Endee."""
    try:
        result = ingest_document(
            title=request.title,
            content=request.content,
            source=request.source,
            category=request.category,
        )
        return IngestResponse(
            message=f"Document '{request.title}' ingested successfully",
            title=result["title"],
            chunks_created=result["chunks_created"],
            document_ids=result["document_ids"],
        )
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ── Semantic Search ────────────────────────────────


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Search",
    description="Search the knowledge base using natural language. Returns the most "
    "semantically similar document chunks from Endee.",
)
async def semantic_search(request: SearchRequest):
    """Embed the query and search Endee for similar document chunks."""
    try:
        results = rag_engine.retrieve(query=request.query, top_k=request.top_k)

        sources = []
        for doc in results:
            meta = doc.get("meta", {})
            sources.append(
                SourceDocument(
                    id=doc.get("id", ""),
                    title=meta.get("title", ""),
                    content=meta.get("content", ""),
                    source=meta.get("source", ""),
                    category=meta.get("category", ""),
                    score=doc.get("score", 0.0),
                    chunk_index=meta.get("chunk_index", 0),
                )
            )

        return SearchResponse(
            query=request.query,
            results=sources,
            total_results=len(sources),
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ── RAG Question Answering ─────────────────────────


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask AI (RAG)",
    description="Ask a question and get an AI-generated answer grounded in the "
    "knowledge base. Uses Retrieval-Augmented Generation with Endee.",
)
async def ask_question(request: AskRequest):
    """Full RAG pipeline: retrieve from Endee → generate answer with LLM."""
    try:
        result = rag_engine.query(question=request.question, top_k=request.top_k)

        sources = [
            SourceDocument(**src) for src in result["sources"]
        ]

        return AskResponse(
            question=result["question"],
            answer=result["answer"],
            sources=sources,
            model_used=result["model_used"],
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Question answering failed: {str(e)}"
        )
