#!/usr/bin/env python3
"""
iPOSpays RAG Data Pipeline
==========================
Scrapes all iPOSpays URLs, chunks the text, embeds with Google Gemini,
and stores vectors in Qdrant Cloud.

Usage:
    pip install -r requirements.txt
    export GOOGLE_API_KEY="your-gemini-api-key"
    python data_pipeline.py

Features:
    - Scrapes 469+ URLs with retry logic
    - Strips HTML to clean text
    - Chunks text with overlap (1600 chars, 200 overlap)
    - Embeds with gemini-embedding-001 (forced to 768 dims for consistency)
    - Stores in Qdrant Cloud with metadata
    - Change detection via content hashing (skips unchanged pages)
    - Batch processing with rate limit handling
    - Progress tracking and resume capability
"""

import os
import sys
import json
import time
import hashlib
import re
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

# ── Configuration ──────────────────────────────────────────────
QDRANT_URL = "https://67b235d3-96eb-45b1-aaa2-0a2156e4ffe1.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.EEtDd1f32jA9expXqWZsC02peDIbwRIRNV2pVQydoR0"
COLLECTION_NAME = "ipospays-knowledge"

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # Force consistent dimensions
EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT"

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
MIN_TEXT_LENGTH = 200  # Skip pages with less text

URLS_FILE = "ipospays_all_urls.txt"
STATE_FILE = "pipeline_state.json"  # Tracks hashes for change detection

BATCH_SIZE = 5  # Qdrant upsert batch size
EMBED_BATCH_SIZE = 5  # Embed 5 chunks at a time (Gemini supports batch)
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_PAGES = 0.5  # Seconds between page fetches
DELAY_BETWEEN_EMBEDS = 1.0  # Seconds between embedding API calls
MAX_RETRIES = 5  # More retries for rate limits

# URLs to skip (auth, payment, email-like)
SKIP_PATTERNS = [
    "auth.ipospays.com",
    "payment.ipospays.com",
    "@",  # email-like URLs
]

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── HTML Cleaning ──────────────────────────────────────────────


def extract_text_from_html(html: str) -> tuple[str, str]:
    """Extract clean text and title from HTML content."""
    soup = BeautifulSoup(html, "html.parser")

    # Get title
    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    # Remove unwanted elements
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "noscript", "iframe"]):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text, title


# ── Text Chunking ──────────────────────────────────────────────


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, trying to break at sentence boundaries."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to break at a sentence boundary
            for sep in ["\n\n", ".\n", ". ", "\n", " "]:
                boundary = text.rfind(sep, start + chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + len(sep)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


# ── Content Hashing ────────────────────────────────────────────


def content_hash(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── State Management ───────────────────────────────────────────


def load_state() -> dict:
    """Load pipeline state (content hashes for change detection)."""
    if Path(STATE_FILE).exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"hashes": {}, "last_run": None, "stats": {}}


def save_state(state: dict):
    """Save pipeline state."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── URL Filtering ──────────────────────────────────────────────


def load_urls() -> list[str]:
    """Load and filter URLs from file."""
    if not Path(URLS_FILE).exists():
        log.error(f"URLs file not found: {URLS_FILE}")
        sys.exit(1)

    with open(URLS_FILE, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    filtered = []
    for url in urls:
        if not url.startswith("https://"):
            continue
        if any(pattern in url for pattern in SKIP_PATTERNS):
            continue
        filtered.append(url)

    log.info(f"Loaded {len(filtered)} URLs (filtered from {len(urls)} total)")
    return filtered


# ── Web Scraping ───────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
})


def fetch_page(url: str) -> Optional[str]:
    """Fetch a web page with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            elif resp.status_code in (429, 503):
                wait = min(2 ** attempt * 5, 60)
                log.warning(f"  Rate limited on {url}, waiting {wait}s...")
                time.sleep(wait)
            else:
                log.warning(f"  HTTP {resp.status_code} for {url}")
                return None
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                log.warning(f"  Failed to fetch {url}: {e}")
                return None
    return None


# ── Embedding ──────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini with forced dimensions."""
    for attempt in range(MAX_RETRIES):
        try:
            # For single text, pass as string; for multiple, pass as list
            content = texts[0] if len(texts) == 1 else texts
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=content,
                task_type=EMBEDDING_TASK_TYPE,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            )
            raw = result["embedding"]

            # Normalize: single text returns flat list, multiple returns list of lists
            if len(texts) == 1 and isinstance(raw[0], (int, float)):
                embeddings = [raw]
            else:
                embeddings = raw

            # Validate all embeddings have correct dimensions
            for i, emb in enumerate(embeddings):
                if len(emb) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"Embedding {i} has {len(emb)} dims, expected {EMBEDDING_DIMENSIONS}"
                    )

            return embeddings

        except Exception as e:
            err_str = str(e).lower()
            is_network = any(k in err_str for k in ["503", "timeout", "connect", "deadline", "unavailable"])
            is_rate = any(k in err_str for k in ["429", "quota", "resource", "rate"])

            if is_network:
                wait = min(2 ** attempt * 30, 300)  # 30s, 60s, 120s, 240s, 300s
                log.warning(f"  Network error (retry {attempt+1}/{MAX_RETRIES}), waiting {wait}s: {str(e)[:80]}")
                time.sleep(wait)
            elif is_rate:
                wait = min(2 ** attempt * 15, 180)
                log.warning(f"  Rate limit (retry {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                log.warning(f"  Embedding error (retry {attempt+1}): {e}")
                time.sleep(2 ** attempt * 3)
            else:
                log.error(f"  Embedding failed after {MAX_RETRIES} retries: {e}")
                return None  # Return None explicitly instead of raising


# ── Qdrant Operations ─────────────────────────────────────────


def setup_qdrant() -> QdrantClient:
    """Initialize Qdrant client and ensure collection exists."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)

    if not exists:
        log.info(f"Creating collection '{COLLECTION_NAME}' with {EMBEDDING_DIMENSIONS} dims...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=Distance.COSINE,
            ),
        )
    else:
        # Verify dimensions match
        info = client.get_collection(COLLECTION_NAME)
        existing_size = info.config.params.vectors.size
        if existing_size != EMBEDDING_DIMENSIONS:
            log.warning(
                f"Collection has {existing_size} dims, need {EMBEDDING_DIMENSIONS}. Recreating..."
            )
            client.delete_collection(COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )

    # Create payload indexes for efficient filtering
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="content_hash",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # Indexes may already exist

    log.info(f"Qdrant collection '{COLLECTION_NAME}' ready ({EMBEDDING_DIMENSIONS} dims)")
    return client


def delete_old_vectors(client: QdrantClient, url: str, current_hash: str):
    """Delete old vectors for a URL (ones with different content hash)."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=url))],
                must_not=[FieldCondition(key="content_hash", match=MatchValue(value=current_hash))],
            ),
        )
    except Exception as e:
        log.debug(f"  Delete old vectors for {url}: {e}")


def upsert_vectors(
    client: QdrantClient,
    chunks: list[str],
    embeddings: list[list[float]],
    url: str,
    title: str,
    content_hash_val: str,
):
    """Upsert chunk vectors into Qdrant."""
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Create deterministic point ID from URL + chunk index
        point_id = hashlib.md5(f"{url}:{content_hash_val}:{i}".encode()).hexdigest()
        # Qdrant expects UUID-like or integer IDs
        # Convert first 32 hex chars to UUID format
        uuid_str = f"{point_id[:8]}-{point_id[8:12]}-{point_id[12:16]}-{point_id[16:20]}-{point_id[20:32]}"

        points.append(
            PointStruct(
                id=uuid_str,
                vector=embedding,
                payload={
                    "source": url,
                    "title": title,
                    "content_hash": content_hash_val,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        )

    # Upsert in batches
    for batch_start in range(0, len(points), BATCH_SIZE):
        batch = points[batch_start : batch_start + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)


# ── Main Pipeline ──────────────────────────────────────────────


def run_pipeline():
    """Run the full data pipeline."""
    log.info("=" * 60)
    log.info("iPOSpays RAG Data Pipeline")
    log.info("=" * 60)

    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log.error("GOOGLE_API_KEY environment variable not set!")
        log.error("Run: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)

    genai.configure(api_key=api_key)

    # Load URLs and state
    urls = load_urls()
    state = load_state()

    # Setup Qdrant
    qdrant = setup_qdrant()

    # Stats
    stats = {
        "total_urls": len(urls),
        "processed": 0,
        "changed": 0,
        "unchanged": 0,
        "skipped_short": 0,
        "skipped_fetch_failed": 0,
        "chunks_embedded": 0,
        "errors": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    log.info(f"\nProcessing {len(urls)} URLs...\n")

    for idx, url in enumerate(urls, 1):
        log.info(f"[{idx}/{len(urls)}] {url}")
        stats["processed"] += 1

        # 1. Fetch page
        html = fetch_page(url)
        if not html:
            log.info(f"  ⊘ Fetch failed, skipping")
            stats["skipped_fetch_failed"] += 1
            continue

        # 2. Extract text
        text, title = extract_text_from_html(html)
        if len(text) < MIN_TEXT_LENGTH:
            log.info(f"  ⊘ Too short ({len(text)} chars), skipping")
            stats["skipped_short"] += 1
            continue

        # 3. Check for changes
        current_hash = content_hash(text)
        previous_hash = state["hashes"].get(url)

        if current_hash == previous_hash:
            log.info(f"  ⊘ Unchanged, skipping")
            stats["unchanged"] += 1
            continue

        log.info(f"  ✓ Changed ({len(text)} chars, title: {title[:50]})")
        stats["changed"] += 1

        # 4. Chunk text
        chunks = chunk_text(text)
        log.info(f"  → {len(chunks)} chunks")

        # 5. Embed chunks in batches of EMBED_BATCH_SIZE
        all_embeddings = []
        embed_failed = False
        for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            embeddings = embed_texts(batch)
            if embeddings is None:
                log.warning(f"  ⚠ Embedding failed at batch {batch_start//EMBED_BATCH_SIZE + 1}, saving {len(all_embeddings)} partial embeddings")
                embed_failed = True
                break
            all_embeddings.extend(embeddings)
            batch_num = batch_start // EMBED_BATCH_SIZE + 1
            total_batches = (len(chunks) - 1) // EMBED_BATCH_SIZE + 1
            log.info(
                f"  → Embedded batch {batch_num}/{total_batches}"
                f" ({len(all_embeddings)}/{len(chunks)} chunks, {EMBEDDING_DIMENSIONS}d)"
            )

            if batch_start + EMBED_BATCH_SIZE < len(chunks):
                time.sleep(DELAY_BETWEEN_EMBEDS)

        # Store whatever we have (partial or full)
        if len(all_embeddings) == 0:
            log.error(f"  ✗ No embeddings produced, skipping")
            stats["errors"] += 1
            continue

        # Use only the chunks that were successfully embedded
        stored_chunks = chunks[:len(all_embeddings)]

        # 6. Delete old vectors for this URL
        delete_old_vectors(qdrant, url, current_hash)

        # 7. Upsert vectors (partial or full)
        try:
            upsert_vectors(qdrant, stored_chunks, all_embeddings, url, title, current_hash)
            if embed_failed:
                log.warning(f"  → Stored {len(stored_chunks)}/{len(chunks)} vectors (partial) in Qdrant")
            else:
                log.info(f"  → Stored {len(stored_chunks)} vectors in Qdrant")
            stats["chunks_embedded"] += len(stored_chunks)
        except Exception as e:
            log.error(f"  ✗ Qdrant upsert failed: {e}")
            stats["errors"] += 1
            continue

        # 8. Update state (even partial — re-run will re-process if content changes)
        state["hashes"][url] = current_hash
        save_state(state)

        # Polite delay
        time.sleep(DELAY_BETWEEN_PAGES)

    # Final summary
    stats["finished_at"] = datetime.now(timezone.utc).isoformat()
    state["last_run"] = stats
    save_state(state)

    log.info("\n" + "=" * 60)
    log.info("Pipeline Complete!")
    log.info("=" * 60)
    log.info(f"  URLs processed:    {stats['processed']}")
    log.info(f"  Changed/new:       {stats['changed']}")
    log.info(f"  Unchanged:         {stats['unchanged']}")
    log.info(f"  Skipped (short):   {stats['skipped_short']}")
    log.info(f"  Skipped (fetch):   {stats['skipped_fetch_failed']}")
    log.info(f"  Chunks embedded:   {stats['chunks_embedded']}")
    log.info(f"  Errors:            {stats['errors']}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
