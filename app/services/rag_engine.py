"""
RAG (Retrieval-Augmented Generation) engine.

Combines Endee vector retrieval with LLM-based answer generation
to provide grounded, context-aware responses to user questions.
"""

import logging
from app.core.embeddings import embed_text
from app.core.vector_store import search
from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ── LLM Singleton ──────────────────────────────────

_llm_pipeline = None


def _get_llm():
    """Lazy-load the LLM pipeline."""
    global _llm_pipeline
    if _llm_pipeline is None:
        settings = get_settings()

        if settings.llm_provider == "openai" and settings.openai_api_key:
            logger.info("Using OpenAI for answer generation")
            _llm_pipeline = _create_openai_client(settings)
        else:
            logger.info(f"Loading local LLM: {settings.llm_model}")
            _llm_pipeline = _create_local_pipeline(settings)

    return _llm_pipeline


def _create_local_pipeline(settings):
    """Create a local HuggingFace text generation pipeline."""
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.llm_model)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    logger.info(f"Local LLM loaded: {settings.llm_model}")
    return {"type": "local", "pipeline": pipe, "model_name": settings.llm_model}


def _create_openai_client(settings):
    """Create an OpenAI client wrapper."""
    return {
        "type": "openai",
        "api_key": settings.openai_api_key,
        "model_name": settings.openai_model,
    }


# ── Retrieval ──────────────────────────────────────


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve the most relevant document chunks from Endee.

    Args:
        query: Natural language query.
        top_k: Number of results to return.

    Returns:
        List of result dicts with id, score, and meta.
    """
    logger.info(f"Retrieving context for: '{query[:80]}...'")

    # Embed the query
    query_vector = embed_text(query)

    # Search Endee
    results = search(vector=query_vector, top_k=top_k)

    logger.info(f"Retrieved {len(results)} context documents")
    return results


# ── Generation ─────────────────────────────────────


def _build_prompt(question: str, context_docs: list[dict]) -> str:
    """Build a RAG prompt with retrieved context."""
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        meta = doc.get("meta", {})
        title = meta.get("title", "Unknown")
        content = meta.get("content", "")
        context_parts.append(f"[Source {i}: {title}]\n{content}")

    context_text = "\n\n".join(context_parts)

    prompt = (
        f"Answer the following question based on the provided context. "
        f"If the context doesn't contain enough information, say so clearly. "
        f"Be concise and accurate.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return prompt


def generate_answer(question: str, context_docs: list[dict]) -> dict:
    """
    Generate an answer using context documents and an LLM.

    Args:
        question: User's question.
        context_docs: Retrieved context documents from Endee.

    Returns:
        Dict with "answer" and "model_used" keys.
    """
    llm = _get_llm()
    prompt = _build_prompt(question, context_docs)

    if llm["type"] == "local":
        result = llm["pipeline"](prompt)
        answer = result[0]["generated_text"].strip()
    elif llm["type"] == "openai":
        answer = _call_openai(llm, prompt)
    else:
        answer = "LLM not configured."

    return {"answer": answer, "model_used": llm["model_name"]}


def _call_openai(llm: dict, prompt: str) -> str:
    """Call OpenAI API for answer generation."""
    try:
        import openai

        client = openai.OpenAI(api_key=llm["api_key"])
        response = client.chat.completions.create(
            model=llm["model_name"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant. Answer questions based on the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error generating answer: {str(e)}"


# ── Full RAG Pipeline ──────────────────────────────


def query(question: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline: retrieve context from Endee → generate answer.

    Args:
        question: Natural language question.
        top_k: Number of context documents to retrieve.

    Returns:
        Dict with "question", "answer", "sources", and "model_used".
    """
    logger.info(f"RAG query: '{question[:80]}...'")

    # Step 1: Retrieve relevant context from Endee
    context_docs = retrieve(question, top_k=top_k)

    # Step 2: Generate answer using LLM + context
    result = generate_answer(question, context_docs)

    # Step 3: Format source documents
    sources = []
    for doc in context_docs:
        meta = doc.get("meta", {})
        sources.append({
            "id": doc.get("id", ""),
            "title": meta.get("title", ""),
            "content": meta.get("content", ""),
            "source": meta.get("source", ""),
            "category": meta.get("category", ""),
            "score": doc.get("score", 0.0),
            "chunk_index": meta.get("chunk_index", 0),
        })

    return {
        "question": question,
        "answer": result["answer"],
        "sources": sources,
        "model_used": result["model_used"],
    }
