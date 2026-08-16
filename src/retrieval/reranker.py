"""
Cross-Encoder Reranker — Deep semantic re-ranking for clinical precision.

Uses a pre-trained sentence-transformers ``CrossEncoder`` to perform joint
attention between the clinical query and candidate chunk passages.
Computes cross-encoder relevance logits and calibrated sigmoid confidence metrics.

Usage:
    from src.retrieval.reranker import Reranker

    reranker = Reranker()
    reranked = reranker.rerank(query, candidates, top_k=15)
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import RERANKER_MODEL, TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Singleton Model Cache
# ──────────────────────────────────────────────────────────────────────
_CROSS_ENCODER_CACHE: dict[str, CrossEncoder] = {}


def _resolve_model_path(model_name: str) -> str:
    """Check for local cached snapshot or return standard model identifier."""
    repo_dir_name = "models--" + model_name.replace("/", "--")
    cache_base = Path.home() / ".cache" / "huggingface" / "hub" / repo_dir_name / "snapshots"
    if cache_base.exists():
        snapshots = list(cache_base.iterdir())
        if snapshots:
            local_snap = str(snapshots[0])
            logger.info("Found local cached CrossEncoder snapshot at: %s", local_snap)
            return local_snap
    return model_name


def _get_cross_encoder(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    """Return a cached CrossEncoder instance with fast local loading."""
    if model_name not in _CROSS_ENCODER_CACHE:
        target_path = _resolve_model_path(model_name)
        logger.info("Loading CrossEncoder model '%s' …", target_path)
        try:
            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(target_path, local_files_only=True)
        except Exception:
            _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(target_path)
        logger.info("CrossEncoder '%s' loaded successfully.", model_name)
    return _CROSS_ENCODER_CACHE[model_name]


def _sigmoid(logit: float) -> float:
    """Map real-valued cross-encoder logit to a calibrated (0, 1) confidence score."""
    try:
        return 1.0 / (1.0 + math.exp(-logit))
    except OverflowError:
        return 1.0 if logit > 0 else 0.0


class Reranker:
    """Cross-encoder based candidate passage reranker."""

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace cross-encoder model identifier.
        """
        self.model_name = model_name
        self.model = _get_cross_encoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = TOP_K_RETRIEVAL,
    ) -> list[dict[str, Any]]:
        """
        Score (query, chunk_text) pairs and return candidates sorted descending by cross-encoder score.

        Parameters
        ----------
        query : str
            Clinical query string.
        candidates : list[dict]
            Candidate chunks from hybrid retrieval.
        top_k : int, default=15
            Number of scored candidates to return.

        Returns
        -------
        list[dict]
            Chunks sorted descending by cross-encoder score.
            Each chunk dictionary includes ``reranker_score`` and ``confidence``.
        """
        if not candidates or not query:
            return []

        # Prepare (query, passage) pairs
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Predict cross-encoder relevance logits
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores to candidates
        scored_candidates: list[dict[str, Any]] = []
        for cand, score in zip(candidates, raw_scores):
            cand_copy = dict(cand)
            score_val = float(score)
            cand_copy["reranker_score"] = round(score_val, 4)
            cand_copy["confidence"] = round(_sigmoid(score_val), 4)
            scored_candidates.append(cand_copy)

        # Sort descending by cross-encoder score
        reranked = sorted(
            scored_candidates,
            key=lambda x: x["reranker_score"],
            reverse=True,
        )[:top_k]

        return reranked
