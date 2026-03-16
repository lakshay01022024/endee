<div align="center">

# 🧠 Endee AI Knowledge Base

### RAG-Powered Semantic Search System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Endee](https://img.shields.io/badge/Endee-Vector_DB-6366f1)](https://github.com/endee-io/endee)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-style **Retrieval-Augmented Generation (RAG)** system that uses the **Endee Vector Database** for intelligent document storage, semantic search, and AI-powered question answering.

[Getting Started](#-getting-started) · [Architecture](#-architecture) · [API Reference](#-api-reference) · [How Endee is Used](#-how-endee-is-used)

</div>

---

## 📋 Problem Statement

Organizations accumulate vast amounts of knowledge across documents, wikis, and internal resources. Traditional keyword search fails to surface relevant information when users phrase queries differently from the source text. Teams need a system that:

- **Understands meaning, not just keywords** — finds relevant documents even when exact terms don't match
- **Answers questions directly** — synthesizes information from multiple sources into concise answers
- **Scales with their data** — efficiently handles growing knowledge bases without performance degradation
- **Runs on their infrastructure** — no vendor lock-in for the vector storage layer

This project solves these problems by combining **Endee's high-performance vector search** with a **RAG pipeline** that retrieves relevant context and generates grounded answers.

---

## 🏗 Architecture

```
┌─────────────────┐      ┌──────────────────────────────────────┐
│   Frontend UI   │─────▶│  FastAPI Backend  (port 8000)        │
│  (HTML/JS/CSS)  │      │                                      │
└─────────────────┘      │  ┌────────────┐  ┌───────────────┐  │
                         │  │ /api/ingest │  │ /api/search   │  │
                         │  │             │  │ /api/ask      │  │
                         │  └──────┬──────┘  └───────┬───────┘  │
                         │         │                 │          │
                         │  ┌──────▼─────────────────▼───────┐  │
                         │  │     Embedding Service          │  │
                         │  │  (all-MiniLM-L6-v2, dim=384)   │  │
                         │  └──────┬─────────────────┬───────┘  │
                         │         │                 │          │
                         │  ┌──────▼──────┐  ┌──────▼───────┐  │
                         │  │ Ingestion   │  │ RAG Engine   │  │
                         │  │ Pipeline    │  │ (flan-t5)    │  │
                         │  └──────┬──────┘  └──────┬───────┘  │
                         └─────────┼────────────────┼──────────┘
                                   │                │
                         ┌─────────▼────────────────▼──────────┐
                         │     Endee Vector Database            │
                         │     (port 8080, cosine, INT8)        │
                         └─────────────────────────────────────┘
```

### Data Flow

| Stage | What Happens |
|---|---|
| **Ingest** | Document → chunk (512 chars, 50 overlap) → embed with Sentence Transformers → upsert vectors + metadata into Endee |
| **Search** | Query → embed → Endee cosine similarity search → return ranked chunks with scores |
| **Ask (RAG)** | Query → embed → retrieve top-k from Endee → build context prompt → generate answer with LLM |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) (Docker, Python SDK) |
| **Backend** | FastAPI + Uvicorn |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **LLM** | HuggingFace (`google/flan-t5-base`) or OpenAI (optional) |
| **Frontend** | Vanilla HTML/CSS/JS (dark glassmorphism theme) |
| **Infrastructure** | Docker Compose |

---

## 🔍 How Endee is Used

Endee serves as the **core vector storage and retrieval engine** for the entire system. Here's a step-by-step breakdown:

### 1. Initialization
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create a 384-dimensional index with cosine similarity
client.create_index(
    name="knowledge_base",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8  # Quantized for speed + memory efficiency
)
```

### 2. Document Ingestion (Upsert)
```python
index = client.get_index(name="knowledge_base")

# Each document chunk is stored as a vector with rich metadata
index.upsert([{
    "id": "doc1_chunk0",
    "vector": [0.023, -0.041, ...],  # 384-dim embedding
    "meta": {
        "title": "Understanding RAG",
        "content": "RAG combines retrieval with generation...",
        "source": "manual",
        "category": "ai",
        "chunk_index": 0
    }
}])
```

### 3. Semantic Search (Query)
```python
# Embed the user's query and search Endee
query_vector = embedding_model.encode("How do vector databases work?")

results = index.query(
    vector=query_vector.tolist(),
    top_k=5  # Return 5 most similar chunks
)
# Returns: [{"id": "...", "similarity": 0.89, "meta": {...}}, ...]
```

### 4. RAG Answer Generation
The retrieved documents from Endee are combined into a prompt and fed to an LLM:
```
Context: [Retrieved chunks from Endee with similarity scores]
Question: How do vector databases work?
Answer: → LLM generates a grounded response
```

---

## 🚀 Getting Started

### Prerequisites
- **Docker** (for Endee server)
- **Python 3.10+**
- **pip**

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/endee-ai-knowledge-base.git
cd endee-ai-knowledge-base
```

### 2. Start Endee Vector Database
```bash
docker compose up -d
```
Verify: open [http://localhost:8080](http://localhost:8080) — you should see the Endee dashboard.

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env if needed (defaults work out of the box)
```

### 5. Seed the Knowledge Base
```bash
python scripts/seed_data.py
```
This loads 15 sample AI/ML documents into Endee (~30 vector chunks).

### 6. Start the Backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 7. Open the App
- **Frontend UI:** [http://localhost:8000](http://localhost:8000)
- **Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Reference

### `GET /api/health` — Health Check
```bash
curl http://localhost:8000/api/health
```

### `POST /api/ingest` — Ingest Document
```bash
curl -X POST http://localhost:8000/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Introduction to Neural Networks",
    "content": "Neural networks are computational models inspired by...",
    "source": "manual",
    "category": "ai"
  }'
```

### `POST /api/search` — Semantic Search
```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "How does cosine similarity work?", "top_k": 5}'
```

### `POST /api/ask` — RAG Question Answering
```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between semantic and keyword search?", "top_k": 5}'
```

---

## 📁 Project Structure

```
endee/
├── docker-compose.yml          # Endee server config
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── README.md
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── embeddings.py       # Sentence Transformers wrapper
│   │   └── vector_store.py     # Endee SDK wrapper
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── services/
│   │   ├── ingestion.py        # Chunking + embedding pipeline
│   │   └── rag_engine.py       # Retrieve + generate pipeline
│   └── schemas/
│       └── models.py           # Pydantic request/response models
├── data/
│   └── sample_documents.json   # Sample knowledge base
├── scripts/
│   └── seed_data.py            # Bulk data loader
└── frontend/
    ├── index.html              # Single-page application
    ├── style.css               # Dark glassmorphism theme
    └── app.js                  # Client-side logic
```

---

## 🔧 Production Enhancements

To make this project even more production-ready, consider:

| Enhancement | Details |
|---|---|
| **Authentication** | Add API key middleware to FastAPI; use Endee's `NDD_AUTH_TOKEN` |
| **Rate Limiting** | Use `slowapi` to protect endpoints from abuse |
| **Caching** | Cache embeddings for repeated queries with Redis |
| **Monitoring** | Add Prometheus metrics + Grafana dashboards |
| **CI/CD** | GitHub Actions for linting, testing, and Docker image builds |
| **Testing** | Add pytest test suite with `httpx.AsyncClient` |
| **File Upload** | Accept PDF/DOCX uploads with `python-docx` and `PyPDF2` |
| **Streaming** | Stream RAG responses via WebSocket/SSE for better UX |
| **Multi-tenancy** | Separate Endee indexes per user/organization |
| **Deployment** | Deploy to AWS/GCP with Docker Compose or Kubernetes |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

Built with ❤️ using [Endee Vector Database](https://github.com/endee-io/endee)

</div>
