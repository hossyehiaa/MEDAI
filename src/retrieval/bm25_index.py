"""
BM25 Lexical Index — Fast keyword-based retrieval using BM25Okapi.

Constructs an in-memory lexical index over all document chunks in ``data/chunks.json``.
Uses lowercase tokenization with punctuation stripping for medical terminology matching.

Usage:
    from src.retrieval.bm25_index import BM25Index

    index = BM25Index()
    results = index.search("PHQ-9 depression screening cutoff", k=15)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import CHUNKS_OUTPUT_PATH, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

# Tokenization pattern: alphanumeric words + preserved hyphens inside terms (e.g., phq-9, ces-d)
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize_text(text: str) -> list[str]:
    """
    Tokenize *text* for BM25 indexing and querying.

    Converts to lowercase and extracts words/alphanumeric tokens,
    preserving medical hyphens (e.g., 'phq-9', 'c-ssrs').
    """
    return _TOKEN_PATTERN.findall(text.lower())


# ──────────────────────────────────────────────────────────────────────
# Singleton Cache
# ──────────────────────────────────────────────────────────────────────
_BM25_INSTANCE: BM25Index | None = None


def get_bm25_index(chunks_path: str | Path = CHUNKS_OUTPUT_PATH) -> BM25Index:
    """Return the cached singleton :class:`BM25Index` instance."""
    global _BM25_INSTANCE
    if _BM25_INSTANCE is None:
        _BM25_INSTANCE = BM25Index(chunks_path=chunks_path)
    return _BM25_INSTANCE


class BM25Index:
    """In-memory BM25Okapi index over document chunks."""

    def __init__(self, chunks_path: str | Path = CHUNKS_OUTPUT_PATH) -> None:
        """
        Parameters
        ----------
        chunks_path : str | Path
            Path to ``chunks.json``.
        """
        self.chunks_path = Path(chunks_path)
        self.chunks: list[dict[str, Any]] = []
        self.corpus_tokens: list[list[str]] = []
        self.index: BM25Okapi | None = None

        self._build_index()

    def _build_index(self) -> None:
        """Load chunks and build the BM25Okapi index."""
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at '{self.chunks_path}'. Run ingest.py first.")

        logger.info("Loading chunks from '%s' for BM25 indexing …", self.chunks_path)
        with open(self.chunks_path, "r", encoding="utf-8") as fh:
            self.chunks = json.load(fh)

        self.corpus_tokens = [tokenize_text(c.get("text", "")) for c in self.chunks]
        self.index = BM25Okapi(self.corpus_tokens)
        logger.info("BM25 index built successfully over %d chunks.", len(self.chunks))

    def search(self, query: str, k: int = TOP_K_RETRIEVAL) -> list[dict[str, Any]]:
        """
        Search the BM25 index with a query string.

        Parameters
        ----------
        query : str
            Clinical query string.
        k : int, default=15
            Number of top results to return.

        Returns
        -------
        list[dict]
            Top-k results sorted descending by BM25 score. Each dict includes
            ``chunk_id``, ``bm25_score``, ``bm25_rank``, ``text``, and metadata.
        """
        if not self.index or not self.chunks:
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        doc_scores = self.index.get_scores(tokenized_query)

        # Get top-k indices sorted descending by score
        ranked_indices = sorted(
            range(len(doc_scores)),
            key=lambda idx: doc_scores[idx],
            reverse=True,
        )[:k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices, 1):
            score = float(doc_scores[idx])
            chunk_data = dict(self.chunks[idx])
            chunk_data["bm25_score"] = score
            chunk_data["bm25_rank"] = rank
            results.append(chunk_data)

        return results

    def count(self) -> int:
        """Return total indexed document count."""
        return len(self.chunks)
