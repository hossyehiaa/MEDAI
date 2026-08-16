"""
Embedder — Generates dense vector embeddings for document chunks using
a locally-running ``sentence-transformers`` model.

Key features:
  • Batched inference with configurable batch_size for GPU/CPU efficiency.
  • ``tqdm`` progress bar for visibility on large corpora.
  • Singleton model caching so the model is loaded only once per process.

Usage:
    from src.ingestion.embedder import Embedder

    embedder = Embedder()
    vectors = embedder.embed_texts(["What is PHQ-9?", "Depression screening"])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Singleton cache
# ──────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance (loaded once)."""
    if model_name not in _MODEL_CACHE:
        logger.info("Loading embedding model '%s' …", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        logger.info("Model loaded: %s", model_name)
    return _MODEL_CACHE[model_name]


# ──────────────────────────────────────────────────────────────────────
# Embedder
# ──────────────────────────────────────────────────────────────────────

class Embedder:
    """Batch-oriented text embedder backed by sentence-transformers."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name / local path.
        batch_size : int
            Number of texts to encode in a single forward pass.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = _get_model(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension: int = self.model.get_embedding_dimension()
        else:
            self.dimension: int = self.model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    def embed_texts(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of text strings.

        Parameters
        ----------
        texts : list[str]
            The input texts to embed.
        show_progress : bool
            Whether to display a ``tqdm`` progress bar.

        Returns
        -------
        list[list[float]]
            One embedding vector per input text.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=f"Embedding ({self.model_name})",
                unit="batch",
            )

        for start in iterator:
            batch = texts[start : start + self.batch_size]
            vectors = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # sentence-transformers returns np.ndarray
            if isinstance(vectors, np.ndarray):
                all_embeddings.extend(vectors.tolist())
            else:
                all_embeddings.extend([v.tolist() for v in vectors])

        logger.info(
            "Embedded %d texts → %d-dim vectors (model=%s).",
            len(texts),
            self.dimension,
            self.model_name,
        )
        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text string (convenience wrapper)."""
        return self.embed_texts([text], show_progress=False)[0]
