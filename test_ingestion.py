"""
test_ingestion.py — Validation suite for the ingestion pipeline output.

Checks:
  1. File existence (chunks.json, vector_db/)
  2. Chunk quality (min chars, metadata fields, token counts)
  3. Vector DB count matches chunks.json
  4. Sample retrieval queries return relevant results

Usage:
    python test_ingestion.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.settings import (
    CHUNKS_OUTPUT_PATH,
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    MIN_CHUNK_CHARS,
)
from src.ingestion.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger("test_ingestion")


def _sep(char: str = "─", width: int = 80) -> str:
    return char * width


def _pass(msg: str) -> None:
    print(f"  ✅ PASS: {msg}")


def _fail(msg: str) -> None:
    print(f"  ❌ FAIL: {msg}")


def run_tests() -> bool:
    print(f"\n{'=' * 80}")
    print("  medAI — Ingestion Pipeline Validation")
    print(f"{'=' * 80}\n")

    all_passed = True

    # ──────────────────────────────────────────────────────────────────
    # TEST 1: File Existence
    # ──────────────────────────────────────────────────────────────────
    print(f"{_sep('═')}")
    print("  TEST 1: File Existence")
    print(_sep("─"))

    chunks_path = CHUNKS_OUTPUT_PATH
    db_path = Path(VECTOR_DB_PATH)

    if chunks_path.exists():
        size_mb = chunks_path.stat().st_size / (1024 * 1024)
        _pass(f"chunks.json exists ({size_mb:.2f} MB)")
    else:
        _fail(f"chunks.json not found at {chunks_path}")
        all_passed = False

    if db_path.exists() and any(db_path.iterdir()):
        _pass(f"vector_db/ exists at {db_path}")
    else:
        _fail(f"vector_db/ not found or empty at {db_path}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────
    # TEST 2: Chunk Quality
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{_sep('═')}")
    print("  TEST 2: Chunk Quality")
    print(_sep("─"))

    if not chunks_path.exists():
        _fail("Cannot test chunk quality — chunks.json missing.")
        all_passed = False
    else:
        with open(chunks_path, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)

        _pass(f"Loaded {len(chunks)} chunks")

        # Required metadata fields
        required_fields = [
            "chunk_id", "document_name", "section_name",
            "start_page", "end_page", "text", "token_count",
            "char_count", "topic", "has_screening_tools", "is_table",
        ]
        missing_fields_count = 0
        for i, c in enumerate(chunks):
            for field in required_fields:
                if field not in c:
                    missing_fields_count += 1
                    if missing_fields_count <= 3:
                        _fail(f"Chunk #{i} missing field '{field}'")

        if missing_fields_count == 0:
            _pass("All chunks have required metadata fields")
        else:
            all_passed = False

        # Min char check
        short_chunks = [c for c in chunks if len(c.get("text", "")) < MIN_CHUNK_CHARS]
        if not short_chunks:
            _pass(f"No chunks shorter than {MIN_CHUNK_CHARS} chars")
        else:
            _fail(f"{len(short_chunks)} chunks shorter than {MIN_CHUNK_CHARS} chars")
            all_passed = False

        # Token counts > 0
        zero_token = [c for c in chunks if c.get("token_count", 0) <= 0]
        if not zero_token:
            _pass("All chunks have positive token counts")
        else:
            _fail(f"{len(zero_token)} chunks have token_count ≤ 0")
            all_passed = False

        # Document coverage
        doc_names = set(c.get("document_name") for c in chunks)
        if len(doc_names) >= 3:
            _pass(f"All 3 source documents represented: {len(doc_names)}")
        else:
            _fail(f"Only {len(doc_names)} documents in chunks (expected 3)")
            all_passed = False

        # Screening tool chunks exist
        screening = [c for c in chunks if c.get("has_screening_tools")]
        if screening:
            _pass(f"{len(screening)} chunks have screening tool flags")
        else:
            _fail("No chunks flagged with screening tools")
            all_passed = False

        # Table chunks exist
        table_chunks = [c for c in chunks if c.get("is_table")]
        if table_chunks:
            _pass(f"{len(table_chunks)} table chunks included")
        else:
            _fail("No table chunks found")
            all_passed = False

        # Stats
        avg_tokens = sum(c.get("token_count", 0) for c in chunks) / max(len(chunks), 1)
        max_tokens = max(c.get("token_count", 0) for c in chunks)
        min_tokens = min(c.get("token_count", 0) for c in chunks)
        print(f"\n  📊 Token stats: min={min_tokens}, avg={avg_tokens:.0f}, max={max_tokens}")

    # ──────────────────────────────────────────────────────────────────
    # TEST 3: Vector DB Count
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{_sep('═')}")
    print("  TEST 3: Vector DB Count")
    print(_sep("─"))

    try:
        store = VectorStore()
        db_count = store.count()

        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as fh:
                expected = len(json.load(fh))
            if db_count == expected:
                _pass(f"ChromaDB count ({db_count}) matches chunks.json ({expected})")
            else:
                _fail(f"ChromaDB count ({db_count}) ≠ chunks.json ({expected})")
                all_passed = False
        else:
            print(f"  ℹ️  ChromaDB count: {db_count} (cannot compare — chunks.json missing)")
    except Exception as e:
        _fail(f"Could not connect to ChromaDB: {e}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────
    # TEST 4: Sample Retrieval Queries
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{_sep('═')}")
    print("  TEST 4: Sample Retrieval Queries")
    print(_sep("─"))

    sample_queries = [
        {
            "query": "Should pregnant women be screened for depression?",
            "expect_keywords": ["pregnant", "postpartum", "depression", "screen"],
        },
        {
            "query": "What screening tools are recommended?",
            "expect_keywords": ["PHQ", "screen", "instrument", "tool"],
        },
        {
            "query": "What is the USPSTF grade for adult depression screening?",
            "expect_keywords": ["grade", "B", "recommend", "depression"],
        },
    ]

    try:
        embedder = Embedder()

        for sq in sample_queries:
            query = sq["query"]
            expect = sq["expect_keywords"]
            print(f"\n  🔍 Query: \"{query}\"")

            vec = embedder.embed_single(query)
            hits = store.search(query_embedding=vec, top_k=3)

            if not hits:
                _fail("No results returned")
                all_passed = False
                continue

            _pass(f"Got {len(hits)} results")

            # Check if any result contains at least one expected keyword
            all_text = " ".join(h.get("text", "") for h in hits).lower()
            found_keywords = [kw for kw in expect if kw.lower() in all_text]
            if found_keywords:
                _pass(f"Results contain expected keywords: {found_keywords}")
            else:
                _fail(f"None of expected keywords found: {expect}")
                all_passed = False

            # Print top result preview
            top = hits[0]
            dist = top.get("distance", 0)
            section = top.get("section_name", "?")
            preview = top.get("text", "")[:150].replace("\n", " ")
            print(f"     Top result (dist={dist:.4f}): §{section} → {preview} …")

    except Exception as e:
        _fail(f"Retrieval test failed: {e}")
        all_passed = False

    # ──────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ──────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    if all_passed:
        print("  🎉 OVERALL: ALL TESTS PASSED")
    else:
        print("  ⚠️  OVERALL: SOME TESTS FAILED — review output above")
    print(f"{'=' * 80}\n")

    return all_passed


if __name__ == "__main__":
    passed = run_tests()
    sys.exit(0 if passed else 1)
