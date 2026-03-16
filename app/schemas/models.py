"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    title: str = Field(..., description="Document title", min_length=1, max_length=500)
    content: str = Field(..., description="Full document text content", min_length=10)
    source: str = Field(
        default="manual", description="Document source (e.g., 'upload', 'web', 'api')"
    )
    category: str = Field(
        default="general",
        description="Document category (e.g., 'ai', 'ml', 'database')",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Introduction to Vector Databases",
                    "content": "Vector databases are specialized database systems designed to store, index, and query high-dimensional vector embeddings...",
                    "source": "manual",
                    "category": "database",
                }
            ]
        }
    }


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str = Field(
        ..., description="Natural language search query", min_length=1, max_length=1000
    )
    top_k: int = Field(default=5, description="Number of results to return", ge=1, le=50)
    category: str | None = Field(
        default=None, description="Optional category filter"
    )


class AskRequest(BaseModel):
    """Request body for RAG-powered question answering."""

    question: str = Field(
        ..., description="Natural language question", min_length=1, max_length=1000
    )
    top_k: int = Field(
        default=5, description="Number of context documents to retrieve", ge=1, le=20
    )


# ── Response Models ────────────────────────────────


class SourceDocument(BaseModel):
    """A retrieved source document with similarity score."""

    id: str = Field(..., description="Document chunk ID")
    title: str = Field(default="", description="Original document title")
    content: str = Field(default="", description="Chunk text content")
    source: str = Field(default="", description="Document source")
    category: str = Field(default="", description="Document category")
    score: float = Field(..., description="Cosine similarity score (0 to 1)")
    chunk_index: int = Field(default=0, description="Chunk position in document")


class SearchResponse(BaseModel):
    """Response for semantic search queries."""

    query: str
    results: list[SourceDocument]
    total_results: int


class AskResponse(BaseModel):
    """Response for RAG-powered question answering."""

    question: str
    answer: str
    sources: list[SourceDocument]
    model_used: str


class IngestResponse(BaseModel):
    """Response after document ingestion."""

    message: str
    title: str
    chunks_created: int
    document_ids: list[str]


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str
    endee: dict
    embedding_model: str
    llm_model: str
