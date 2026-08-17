"""
Retrieval Manager — High-level orchestrator for Clinical Retrieval Layer.

Workflow:
  1. Clinical Query Input
  2. Hybrid Retrieval (ChromaDB Dense + BM25 Sparse fused via RRF k=60)
  3. Force-Inclusion of Population Tools (top-2 GDS for 65+, top-2 EPDS for perinatal)
  4. Cross-Encoder Deep Re-ranking (ms-marco-MiniLM-L-6-v2)
  5. Population Boosts:
     - Perinatal Boost (1.25x for EPDS/Edinburgh/postpartum chunks on perinatal queries)
     - Older Adults Boost (1.10x for GDS/Geriatric/65+ chunks containing screening terms on older adults queries)
  6. Section Prior Boost (Recommendation=1.30, General=1.10, References=0.50)
  7. Greedy Top-3 Diversity Rule (max 1 per DOCUMENT; fallback if <3 unique docs)
  8. Comprehensive audit logging to ``logs/retrieval.log`` (raw + boosted scores + forced inclusions)

Usage:
    from src.retrieval.retrieval_manager import RetrievalManager

    manager = RetrievalManager()
    result = manager.retrieve("Should postpartum women be screened for depression?")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import (
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
    RETRIEVAL_LOG_PATH,
    RRF_K,
    RERANKER_MODEL,
    EMBEDDING_MODEL,
    SECTION_PRIORS,
    CONFIDENCE_THRESHOLD,
    PERINATAL_QUERY_KEYWORDS,
    PERINATAL_CHUNK_KEYWORDS,
    PERINATAL_BOOST,
    OLDER_ADULTS_QUERY_KEYWORDS,
    OLDER_ADULTS_CHUNK_KEYWORDS,
    OLDER_ADULTS_BOOST,
)
from src.retrieval.bm25_index import tokenize_text
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


def _is_perinatal_query(query: str) -> bool:
    """Check if the query relates to perinatal/postpartum screening."""
    q_lower = query.lower()
    return any(kw.lower() in q_lower for kw in PERINATAL_QUERY_KEYWORDS)


def _chunk_matches_perinatal(chunk: dict[str, Any]) -> bool:
    """Check if a chunk contains perinatal-relevant content."""
    text = chunk.get("text", "").lower()
    return any(kw.lower() in text for kw in PERINATAL_CHUNK_KEYWORDS)


def _is_older_adults_query(query: str) -> bool:
    """Check if the query relates to older adults/geriatric depression screening."""
    q_lower = query.lower()
    return any(kw.lower() in q_lower for kw in OLDER_ADULTS_QUERY_KEYWORDS)


def _chunk_matches_older_adults(chunk: dict[str, Any]) -> bool:
    """
    Check if a chunk contains older-adults-relevant content WITH screening context.
    Requires at least one population term AND at least one screening term.
    """
    text = chunk.get("text", "").lower()
    has_pop = any(kw.lower() in text for kw in OLDER_ADULTS_CHUNK_KEYWORDS)
    has_screening = any(term in text for term in ["screen", "screening", "depression", "depressive", "mdd"])
    return has_pop and has_screening


class RetrievalManager:
    """End-to-end clinical retrieval orchestrator with population boosting, force-inclusions, and diversity rules."""

    def __init__(
        self,
        hybrid_searcher: HybridSearcher | None = None,
        reranker: Reranker | None = None,
        log_path: str | Path = RETRIEVAL_LOG_PATH,
    ) -> None:
        """
        Parameters
        ----------
        hybrid_searcher : HybridSearcher, optional
            Hybrid search instance.
        reranker : Reranker, optional
            CrossEncoder reranker instance.
        log_path : str | Path, default="logs/retrieval.log"
            Destination file for retrieval audit logs.
        """
        self.hybrid_searcher = hybrid_searcher or HybridSearcher()
        self.reranker = reranker or Reranker()
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_top_tool_chunks(self, query: str, tool_names: list[str], top_n: int = 2) -> list[dict[str, Any]]:
        """Find and rank candidate chunks tagged with specific screening tools."""
        all_chunks = self.hybrid_searcher.bm25_index.chunks
        matching = [
            c for c in all_chunks
            if any(t in c.get("screening_tools", []) for t in tool_names)
            or any(t.lower() in c.get("text", "").lower() for t in tool_names)
        ]
        if not matching:
            return []

        q_tokens = set(tokenize_text(query))

        def _tool_score(c: dict[str, Any]) -> tuple[int, int]:
            text = c.get("text", "")
            c_tokens = set(tokenize_text(text))
            token_overlap = len(q_tokens.intersection(c_tokens))
            tool_exact_matches = sum(1 for t in tool_names if t.lower() in text.lower())
            return (token_overlap, tool_exact_matches)

        matching.sort(key=_tool_score, reverse=True)
        return matching[:top_n]

    def retrieve(
        self,
        query: str,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        top_k_final: int = TOP_K_FINAL,
    ) -> dict[str, Any]:
        """
        Execute full multi-stage clinical retrieval pipeline.

        Parameters
        ----------
        query : str
            Clinical query string.
        top_k_retrieval : int, default=15
            Candidate pool size from hybrid search.
        top_k_final : int, default=3
            Final top passages after cross-encoder re-ranking & diversity selection.

        Returns
        -------
        dict
            Retrieval bundle containing query, final_chunks, avg_confidence, metrics, forced inclusions.
        """
        t0 = time.perf_counter()

        # Step 1: Hybrid Retrieval (ChromaDB Dense + BM25 Sparse with RRF)
        t_hybrid_start = time.perf_counter()
        hybrid_candidates = self.hybrid_searcher.search(
            query=query,
            top_k=top_k_retrieval,
        )
        hybrid_time_ms = round((time.perf_counter() - t_hybrid_start) * 1000, 2)

        # Step 2: Population Check & Tool Force-Inclusion
        is_perinatal = _is_perinatal_query(query)
        is_older_adults = _is_older_adults_query(query)
        forced_inclusions: list[str] = []
        candidate_pool = list(hybrid_candidates)
        existing_candidate_ids = {c.get("chunk_id") for c in candidate_pool}

        if is_older_adults:
            top_gds = self._get_top_tool_chunks(query, ["GDS", "Geriatric Depression Scale"], top_n=2)
            for g_chunk in top_gds:
                cid = g_chunk.get("chunk_id", "")
                if cid not in existing_candidate_ids:
                    candidate_pool.append(dict(g_chunk))
                    existing_candidate_ids.add(cid)
                forced_inclusions.append(cid)

        if is_perinatal:
            top_epds = self._get_top_tool_chunks(query, ["EPDS", "Edinburgh"], top_n=2)
            for e_chunk in top_epds:
                cid = e_chunk.get("chunk_id", "")
                if cid not in existing_candidate_ids:
                    candidate_pool.append(dict(e_chunk))
                    existing_candidate_ids.add(cid)
                forced_inclusions.append(cid)

        # Step 3: Cross-Encoder Re-ranking (all candidate pool including forced inclusions)
        t_rerank_start = time.perf_counter()
        reranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidate_pool,
            top_k=len(candidate_pool),
        )

        # Step 4: Apply Population-Specific Boosts (Perinatal & Older Adults)
        perinatal_boosted_ids: list[str] = []
        older_adults_boosted_ids: list[str] = []

        for cand in reranked_candidates:
            cand["perinatal_boosted"] = False
            cand["older_adults_boosted"] = False
            raw_conf = cand.get("confidence", 0.0)

            # Perinatal boost (1.25x)
            if is_perinatal and _chunk_matches_perinatal(cand):
                cand["confidence"] = min(raw_conf * PERINATAL_BOOST, 1.0)
                cand["perinatal_boosted"] = True
                perinatal_boosted_ids.append(cand.get("chunk_id", ""))

            # Older Adults boost (1.10x with tightened screening term constraint)
            if is_older_adults and _chunk_matches_older_adults(cand):
                current_conf = cand.get("confidence", raw_conf)
                cand["confidence"] = min(current_conf * OLDER_ADULTS_BOOST, 1.0)
                cand["older_adults_boosted"] = True
                older_adults_boosted_ids.append(cand.get("chunk_id", ""))

        # Step 5: Apply Section Prior Boost (with AAFP recommendation table demotion to 0.40x)
        boosted_candidates: list[dict[str, Any]] = []
        for cand in reranked_candidates:
            c = dict(cand)
            text_lower = c.get("text", "").lower()
            # Defect 3: Demote AAFP recommendation table header pattern polluting Q02 and Q13
            if "| condition |" in text_lower and "| organization" in text_lower and "| recommendation" in text_lower:
                prior = 0.40
            else:
                section = c.get("section_name", "")
                prior = SECTION_PRIORS.get(section, 1.0)
            conf = c.get("confidence", 0.0)
            boosted_score = conf * prior
            c["section_prior"] = round(prior, 4)
            c["boosted_score"] = round(boosted_score, 4)
            boosted_candidates.append(c)

        # Sort candidates descending by boosted_score
        boosted_candidates.sort(key=lambda x: x["boosted_score"], reverse=True)

        # Step 6: Greedy Diversity Selection — Max 1 per DOCUMENT in top-3
        final_chunks: list[dict[str, Any]] = []
        dropped_duplicates: list[dict[str, Any]] = []
        seen_docs: set[str] = set()

        for cand in boosted_candidates:
            doc_name = cand.get("document_name", "")
            if doc_name not in seen_docs:
                seen_docs.add(doc_name)
                final_chunks.append(cand)
                if len(final_chunks) == top_k_final:
                    break
            else:
                dropped_duplicates.append(cand)

        # Fallback: If fewer than 3 unique documents have candidates,
        # fill remaining slots from dropped duplicates (best boosted score first)
        if len(final_chunks) < top_k_final and dropped_duplicates:
            for cand in dropped_duplicates:
                if cand.get("confidence", 0.0) >= 0.5 or not final_chunks:
                    final_chunks.append(cand)
                    if len(final_chunks) == top_k_final:
                        break

        # If still under top_k_final, backfill with next available candidate
        if len(final_chunks) < top_k_final:
            for cand in boosted_candidates:
                if cand not in final_chunks:
                    final_chunks.append(cand)
                    if len(final_chunks) == top_k_final:
                        break

        # Assign final 1-indexed ranks
        for rank_idx, chunk in enumerate(final_chunks, 1):
            chunk["final_rank"] = rank_idx

        rerank_time_ms = round((time.perf_counter() - t_rerank_start) * 1000, 2)
        total_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Step 7: Compute Confidence Summary & Diversity Checks
        confidences = [c.get("confidence", 0.0) for c in final_chunks]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        top1_confidence = final_chunks[0].get("confidence", 0.0) if final_chunks else 0.0
        unique_docs_count = len(set(c.get("document_name", "") for c in final_chunks))
        has_diversity_warning = unique_docs_count < min(2, len(final_chunks))

        # Step 8: Write Audit Log Entry
        self._write_retrieval_log(
            query=query,
            hybrid_candidates=hybrid_candidates,
            boosted_candidates=boosted_candidates,
            final_chunks=final_chunks,
            dropped_duplicates=dropped_duplicates,
            avg_confidence=avg_confidence,
            top1_confidence=top1_confidence,
            total_time_ms=total_time_ms,
            hybrid_time_ms=hybrid_time_ms,
            rerank_time_ms=rerank_time_ms,
            is_perinatal=is_perinatal,
            perinatal_boosted_ids=perinatal_boosted_ids,
            is_older_adults=is_older_adults,
            older_adults_boosted_ids=older_adults_boosted_ids,
            forced_inclusions=forced_inclusions,
        )

        return {
            "query": query,
            "final_chunks": final_chunks,
            "avg_confidence": avg_confidence,
            "top1_confidence": top1_confidence,
            "is_in_scope": top1_confidence >= CONFIDENCE_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "is_perinatal_query": is_perinatal,
            "is_older_adults_query": is_older_adults,
            "forced_inclusions": forced_inclusions,
            "unique_documents_count": unique_docs_count,
            "diversity_warning": has_diversity_warning,
            "hybrid_candidates_count": len(hybrid_candidates),
            "hybrid_candidates": hybrid_candidates,
            "retrieval_time_ms": total_time_ms,
            "latency_breakdown_ms": {
                "hybrid": hybrid_time_ms,
                "rerank": rerank_time_ms,
            },
            "models": {
                "embedding": EMBEDDING_MODEL,
                "reranker": RERANKER_MODEL,
                "rrf_k": RRF_K,
            },
        }

    def _write_retrieval_log(
        self,
        query: str,
        hybrid_candidates: list[dict[str, Any]],
        boosted_candidates: list[dict[str, Any]],
        final_chunks: list[dict[str, Any]],
        dropped_duplicates: list[dict[str, Any]],
        avg_confidence: float,
        top1_confidence: float,
        total_time_ms: float,
        hybrid_time_ms: float,
        rerank_time_ms: float,
        is_perinatal: bool = False,
        perinatal_boosted_ids: list[str] | None = None,
        is_older_adults: bool = False,
        older_adults_boosted_ids: list[str] | None = None,
        forced_inclusions: list[str] | None = None,
    ) -> None:
        """Append a structured JSON line entry to logs/retrieval.log."""
        try:
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "query": query,
                "is_perinatal_query": is_perinatal,
                "perinatal_boosted_chunk_ids": perinatal_boosted_ids or [],
                "is_older_adults_query": is_older_adults,
                "older_adults_boost_applied": bool(older_adults_boosted_ids),
                "older_adults_boosted_chunk_ids": older_adults_boosted_ids or [],
                "forced_inclusions": forced_inclusions or [],
                "metrics": {
                    "total_time_ms": total_time_ms,
                    "hybrid_time_ms": hybrid_time_ms,
                    "rerank_time_ms": rerank_time_ms,
                    "avg_confidence": avg_confidence,
                    "top1_confidence": top1_confidence,
                    "confidence_threshold": CONFIDENCE_THRESHOLD,
                    "hybrid_candidates_count": len(hybrid_candidates),
                    "final_chunks_count": len(final_chunks),
                    "dropped_duplicates_count": len(dropped_duplicates),
                },
                "hybrid_candidates": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "document": c.get("document_name"),
                        "section": c.get("section_name"),
                        "pages": f"p.{c.get('start_page')}-{c.get('end_page')}",
                        "semantic_rank": c.get("semantic_rank"),
                        "semantic_distance": c.get("semantic_distance"),
                        "bm25_rank": c.get("bm25_rank"),
                        "bm25_score": c.get("bm25_score"),
                        "rrf_score": round(c.get("rrf_score", 0.0), 6),
                    }
                    for c in hybrid_candidates
                ],
                "final_chunks": [
                    {
                        "final_rank": c.get("final_rank"),
                        "chunk_id": c.get("chunk_id"),
                        "document": c.get("document_name"),
                        "section": c.get("section_name"),
                        "pages": f"p.{c.get('start_page')}-{c.get('end_page')}",
                        "raw_reranker_score": c.get("reranker_score"),
                        "raw_confidence": c.get("confidence"),
                        "section_prior": c.get("section_prior"),
                        "boosted_score": c.get("boosted_score"),
                        "perinatal_boosted": c.get("perinatal_boosted", False),
                        "older_adults_boosted": c.get("older_adults_boosted", False),
                        "has_screening_tools": c.get("has_screening_tools"),
                        "screening_tools": c.get("screening_tools"),
                    }
                    for c in final_chunks
                ],
            }

            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        except Exception as exc:
            logger.warning("Failed to write retrieval log to '%s': %s", self.log_path, exc)
