#!/usr/bin/env python3
"""
iPOSpays RAG Chatbot API
========================
FastAPI server that answers questions using the Qdrant vector store
populated by data_pipeline.py.

Usage:
    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-gemini-api-key"
    python chatbot_api.py

Endpoints:
    POST /chat  {"question": "How do I integrate Google Pay?"}
    GET  /health
"""

import os
import sys
import logging
from datetime import datetime, timezone

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Configuration ──────────────────────────────────────────────
QDRANT_URL = "https://67b235d3-96eb-45b1-aaa2-0a2156e4ffe1.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.EEtDd1f32jA9expXqWZsC02peDIbwRIRNV2pVQydoR0"
COLLECTION_NAME = "ipospays-knowledge"

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768
CHAT_MODEL = "models/gemini-2.5-flash"

TOP_K = 5  # Number of similar chunks to retrieve
SCORE_THRESHOLD = 0.3  # Minimum similarity score

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chatbot")

# ── Initialize ─────────────────────────────────────────────────
app = FastAPI(title="iPOSpays Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    log.error("GOOGLE_API_KEY environment variable not set!")
    sys.exit(1)

genai.configure(api_key=api_key)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
chat_model = genai.GenerativeModel(CHAT_MODEL)


# ── Models ─────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str
    top_k: int = TOP_K


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    query_time_ms: int


class HealthResponse(BaseModel):
    status: str
    collection: str
    vectors_count: int


# ── Embedding ──────────────────────────────────────────────────


def embed_query(text: str) -> list[float]:
    """Embed a query with forced dimensions for retrieval."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBEDDING_DIMENSIONS,
    )
    embedding = result["embedding"]
    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Query embedding has {len(embedding)} dims, expected {EMBEDDING_DIMENSIONS}")
    return embedding


# ── Retrieval ──────────────────────────────────────────────────


def retrieve_context(query_embedding: list[float], top_k: int) -> list[dict]:
    """Search Qdrant for relevant chunks."""
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        score_threshold=SCORE_THRESHOLD,
    )

    chunks = []
    for hit in response.points:
        chunks.append({
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", ""),
            "title": hit.payload.get("title", ""),
            "score": round(hit.score, 4),
            "chunk_index": hit.payload.get("chunk_index", 0),
        })
    return chunks


# ── Generation ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for iPOSpays, a payment processing platform.
Answer questions based ONLY on the provided context. If the context doesn't contain
enough information to answer, say "I don't have enough information about that in my
knowledge base."

Rules:
- Be concise and accurate
- Cite sources when possible (mention the page URL)
- If the question is about API integration, include relevant details
- Don't make up information not in the context
- Format responses clearly with bullet points or numbered steps when appropriate
"""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Generate answer using Gemini with retrieved context."""
    if not context_chunks:
        return "I don't have enough information about that in my knowledge base."

    # Build context string
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"--- Source {i}: {chunk['title']} ({chunk['source']}) ---\n{chunk['text']}"
        )
    context_str = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_str}

QUESTION: {question}

ANSWER:"""

    try:
        response = chat_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        return response.text
    except Exception as e:
        log.error(f"Gemini API error: {e}")
        return "Sorry, I'm having trouble generating a response right now. Please try again in a moment."


# ── Endpoints ──────────────────────────────────────────────────


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using RAG."""
    import time

    start = time.time()

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    log.info(f"Question: {request.question}")

    # 1. Embed query
    query_embedding = embed_query(request.question)

    # 2. Retrieve context
    chunks = retrieve_context(query_embedding, request.top_k)
    log.info(f"  Retrieved {len(chunks)} chunks")

    # 3. Generate answer
    answer = generate_answer(request.question, chunks)

    elapsed_ms = int((time.time() - start) * 1000)
    log.info(f"  Answered in {elapsed_ms}ms")

    return ChatResponse(
        answer=answer,
        sources=[
            {"url": c["source"], "title": c["title"], "score": c["score"]}
            for c in chunks
        ],
        query_time_ms=elapsed_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with collection info."""
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return HealthResponse(
            status="healthy",
            collection=COLLECTION_NAME,
            vectors_count=info.points_count,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/")
async def root():
    return {
        "name": "iPOSpays Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Ask a question",
            "GET /health": "Health check",
        },
    }


if __name__ == "__main__":
    log.info("Starting iPOSpays Chatbot API on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
