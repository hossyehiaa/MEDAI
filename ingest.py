"""
ingest.py — Main ingestion pipeline for the medAI RAG system.

Workflow:
  1. Load ``data/cleaned_output.json``
  2. Chunk sections + tables  →  structured Chunk objects
  3. Generate embeddings      →  dense vectors (all-MiniLM-L6-v2)
  4. Upsert into ChromaDB     →  persistent cosine-similarity store
  5. Save ``data/chunks.json``
  6. Run 3 sample retrieval queries

Usage:
    python ingest.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.settings import (
    CLEANED_OUTPUT_PATH,
    CHUNKS_OUTPUT_PATH,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    EMBEDDING_MODEL,
)
from src.ingestion.chunker import load_cleaned_data, chunk_cleaned_data, save_chunks
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-38s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger("ingest")


def _separator(char: str = "─", width: int = 80) -> str:
    return char * width


def main() -> None:
    t0 = time.time()

    print(f"\n{'=' * 80}")
    print("  medAI — Document Ingestion Pipeline")
    print(f"{'=' * 80}\n")

    # ── 0. Print configuration ───────────────────────────────────────
    print("⚙️  Configuration:")
    print(f"     Cleaned input  : {CLEANED_OUTPUT_PATH}")
    print(f"     Chunk size     : {CHUNK_SIZE} tokens (overlap {CHUNK_OVERLAP})")
    print(f"     Min chunk      : {MIN_CHUNK_CHARS} chars")
    print(f"     Embedding      : {EMBEDDING_MODEL}")
    print(f"     ChromaDB       : {VECTOR_DB_PATH}")
    print(f"     Collection     : {COLLECTION_NAME}")
    print()

    # ── 1. Load cleaned data ─────────────────────────────────────────
    print(f"{_separator('─')}")
    print("📂  Step 1: Loading cleaned data …")
    data = load_cleaned_data()
    n_sections = len(data.get("sections", []))
    n_tables = len(data.get("tables", []))
    print(f"     Loaded {n_sections} sections + {n_tables} tables")
    print()

    # ── 2. Chunk ─────────────────────────────────────────────────────
    print(f"{_separator('─')}")
    print("✂️   Step 2: Chunking sections & tables …")
    chunks = chunk_cleaned_data(data, include_tables=True)

    # Statistics
    text_chunks = [c for c in chunks if not c.is_table]
    table_chunks = [c for c in chunks if c.is_table]
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
    topic_counts = Counter(c.topic for c in chunks)
    doc_counts = Counter(c.document_name for c in chunks)
    screening_count = sum(1 for c in chunks if c.has_screening_tools)

    print(f"     Text chunks    : {len(text_chunks)}")
    print(f"     Table chunks   : {len(table_chunks)}")
    print(f"     Total chunks   : {len(chunks)}")
    print(f"     Avg tokens     : {avg_tokens:.0f}")
    print(f"     Screening      : {screening_count} chunks mention screening tools")
    print(f"     By document    :")
    for doc, cnt in doc_counts.most_common():
        print(f"       • {doc}: {cnt}")
    print(f"     By topic       :")
    for topic, cnt in topic_counts.most_common():
        print(f"       • {topic}: {cnt}")
    print()

    # ── 3. Generate embeddings ───────────────────────────────────────
    print(f"{_separator('─')}")
    print(f"🧠  Step 3: Generating embeddings ({EMBEDDING_MODEL}) …")
    embedder = Embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts, show_progress=True)
    print(f"     Generated {len(embeddings)} embeddings ({embedder.dimension}-dim)")
    print()

    # ── 4. Upsert into ChromaDB ──────────────────────────────────────
    print(f"{_separator('─')}")
    print("💾  Step 4: Upserting into ChromaDB …")
    store = VectorStore()
    store.reset()  # clean slate on re-ingest
    chunk_dicts = [c.to_dict() for c in chunks]
    store.add_chunks(chunk_dicts, embeddings)
    print(f"     ChromaDB count : {store.count()} documents")
    print()

    # ── 5. Save chunks.json ──────────────────────────────────────────
    print(f"{_separator('─')}")
    print("📄  Step 5: Saving chunks.json …")
    out_path = save_chunks(chunks)
    print(f"     Saved to       : {out_path}")
    print()

    # ── 6. Sample retrieval queries ──────────────────────────────────
    print(f"{_separator('═')}")
    print("🔍  Step 6: Sample Retrieval Queries")
    print(_separator("═"))

    sample_queries = [
        "Should pregnant women be screened for depression?",
        "What screening tools are recommended for depression?",
        "What is the USPSTF grade for adult depression screening?",
    ]

    for query in sample_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  {'·' * 72}")

        query_vec = embedder.embed_single(query)
        hits = store.search(query_embedding=query_vec, top_k=3)

        for rank, hit in enumerate(hits, 1):
            doc = hit.get("document_name", "?")
            section = hit.get("section_name", "?")
            pages = f"p.{hit.get('start_page', '?')}-{hit.get('end_page', '?')}"
            dist = hit.get("distance", 0)
            text_preview = hit.get("text", "")[:200].replace("\n", " ")
            print(f"    [{rank}] dist={dist:.4f}  │  {doc}  │  §{section}  │  {pages}")
            print(f"        {text_preview} …")

    # ── Done ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  ✅ Ingestion complete in {elapsed:.1f}s")
    print(f"     {len(chunks)} chunks  →  {store.count()} vectors in ChromaDB")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
