"""
Endee AI Knowledge Base — FastAPI Application Entry Point.

A production-style RAG (Retrieval-Augmented Generation) system
powered by the Endee Vector Database for intelligent document
search and question answering.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router
from app.core.config import get_settings
from app.core.vector_store import init_index
from app.core.embeddings import get_embedding_dimension

# ── Logging ────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("  Endee AI Knowledge Base — Starting Up")
    logger.info("=" * 60)

    # Initialize embedding model and get dimension
    dim = get_embedding_dimension()
    logger.info(f"Embedding model: {settings.embedding_model} (dim={dim})")

    # Initialize Endee index
    try:
        init_index(dimension=dim)
        logger.info(f"Endee index '{settings.index_name}' ready")
    except Exception as e:
        logger.warning(
            f"Could not initialize Endee index: {e}. "
            f"Ensure Endee is running at {settings.endee_base_url}"
        )

    logger.info(f"LLM: {settings.llm_model} (provider: {settings.llm_provider})")
    logger.info("=" * 60)
    logger.info(f"  API:       http://{settings.app_host}:{settings.app_port}")
    logger.info(f"  Docs:      http://{settings.app_host}:{settings.app_port}/docs")
    logger.info(f"  Endee:     {settings.endee_base_url}")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down Endee AI Knowledge Base")


# ── FastAPI App ────────────────────────────────────

app = FastAPI(
    title="Endee AI Knowledge Base",
    description=(
        "A production-style RAG (Retrieval-Augmented Generation) system powered by "
        "the **Endee Vector Database**. Ingest documents, perform semantic search, "
        "and get AI-generated answers grounded in your knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS Middleware ────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ─────────────────────────────────────

app.include_router(router)

# ── Static Frontend ────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the frontend UI."""
        return FileResponse(str(FRONTEND_DIR / "index.html"))
