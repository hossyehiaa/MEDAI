"""
Hybrid Search — Combines Dense Vector Similarity (ChromaDB) and Sparse Lexical
Matching (BM25) via Reciprocal Rank Fusion (RRF).

Reciprocal Rank Fusion Formula:
    RRF_Score(d) = \\sum_{m \\in M} \\frac{1}{k + \\text{rank}_m(d)}
where M = {semantic, bm25} and k = 60 by default.

Usage:
    from src.retrieval.hybrid_search import HybridSearcher

    searcher = HybridSearcher()
    candidates = searcher.search("depression screening in pregnancy", top_k=15)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import TOP_K_RETRIEVAL, RRF_K
from src.ingestion.embedder import Embedder
from src.retrieval.bm25_index import BM25Index, get_bm25_index
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Orchestrates hybrid retrieval (ChromaDB dense + BM25 sparse) with RRF."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
        embedder: Embedder | None = None,
        rrf_k: int = RRF_K,
    ) -> None:
        """
        Parameters
        ----------
        vector_store : VectorStore, optional
            ChromaDB vector store client.
        bm25_index : BM25Index, optional
            BM25Okapi lexical index instance.
        embedder : Embedder, optional
            Dense text embedder for query encoding.
        rrf_k : int, default=60
            Reciprocal Rank Fusion smoothing constant.
        """
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or get_bm25_index()
        self.embedder = embedder or Embedder()
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid search and fuse results using Reciprocal Rank Fusion.

        Parameters
        ----------
        query : str
            Clinical query string.
        top_k : int, default=15
            Number of top candidates to retrieve from each retriever and return.

        Returns
        -------
        list[dict]
            Fused candidate list sorted descending by ``rrf_score``.
            Each dict contains ``chunk_id``, ``rrf_score``, ``semantic_rank``,
            ``bm25_rank``, ``semantic_distance``, ``bm25_score``, ``text``, and metadata.
        """
        if not query or not query.strip():
            return []

        # 1. Semantic Dense Search (ChromaDB)
        query_embedding = self.embedder.embed_single(query)
        semantic_hits = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # 2. Lexical Sparse Search (BM25)
        bm25_hits = self.bm25_index.search(
            query=query,
            k=top_k,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        # Map chunk_id -> merged info
        fused_map: dict[str, dict[str, Any]] = {}

        # Process Semantic Hits
        for rank, hit in enumerate(semantic_hits, 1):
            cid = hit["chunk_id"]
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            if cid not in fused_map:
                fused_map[cid] = {
                    "chunk_id": cid,
                    "text": hit.get("text", ""),
                    "document_name": hit.get("document_name", ""),
                    "section_name": hit.get("section_name", ""),
                    "start_page": hit.get("start_page", 1),
                    "end_page": hit.get("end_page", 1),
                    "grades": hit.get("grades", ""),
                    "topic": hit.get("topic", "general"),
                    "has_screening_tools": hit.get("has_screening_tools", False),
                    "is_table": hit.get("is_table", False),
                    "semantic_rank": rank,
                    "bm25_rank": None,
                    "semantic_distance": hit.get("distance", None),
                    "bm25_score": 0.0,
                    "rrf_score": rrf_contrib,
                }
            else:
                fused_map[cid]["semantic_rank"] = rank
                fused_map[cid]["semantic_distance"] = hit.get("distance", None)
                fused_map[cid]["rrf_score"] += rrf_contrib

        # Process BM25 Hits
        for rank, hit in enumerate(bm25_hits, 1):
            cid = hit["chunk_id"]
            rrf_contrib = 1.0 / (self.rrf_k + rank)
            if cid not in fused_map:
                fused_map[cid] = {
                    "chunk_id": cid,
                    "text": hit.get("text", ""),
                    "document_name": hit.get("document_name", ""),
                    "section_name": hit.get("section_name", ""),
                    "start_page": hit.get("start_page", 1),
                    "end_page": hit.get("end_page", 1),
                    "grades": hit.get("grades", []),
                    "topic": hit.get("topic", "general"),
                    "has_screening_tools": hit.get("has_screening_tools", False),
                    "is_table": hit.get("is_table", False),
                    "semantic_rank": None,
                    "bm25_rank": rank,
                    "semantic_distance": None,
                    "bm25_score": hit.get("bm25_score", 0.0),
                    "rrf_score": rrf_contrib,
                }
            else:
                fused_map[cid]["bm25_rank"] = rank
                fused_map[cid]["bm25_score"] = hit.get("bm25_score", 0.0)
                fused_map[cid]["rrf_score"] += rrf_contrib

        # Sort descending by fused RRF score
        ranked_candidates = sorted(
            fused_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:top_k]

        logger.info(
            "Hybrid Search for '%s': %d semantic + %d BM25 -> %d fused candidates (RRF_K=%d)",
            query,
            len(semantic_hits),
            len(bm25_hits),
            len(ranked_candidates),
            self.rrf_k,
        )

        return ranked_candidates
