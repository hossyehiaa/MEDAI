"""
Vector Store — Persistent ChromaDB wrapper for medical document retrieval.

Stores chunks with rich metadata (document name, section, pages, grades,
topic tags, screening-tool flags) and supports cosine-similarity search.

Important ChromaDB constraint: metadata values must be ``str | int | float | bool``.
Lists (like ``grades``, ``screening_tools``) are comma-joined into strings.

Usage:
    from src.retrieval.vector_store import VectorStore

    store = VectorStore()
    store.add_chunks(chunks, embeddings)
    results = store.search("depression screening", top_k=5)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import VECTOR_DB_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)


def _flatten_metadata(chunk_dict: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """
    Convert a chunk dict into ChromaDB-safe metadata.

    ChromaDB only accepts str/int/float/bool values.
    Lists are joined with commas; ``None`` values are dropped.
    """
    meta: dict[str, str | int | float | bool] = {}
    for key, value in chunk_dict.items():
        if key in ("text", "chunk_id"):
            continue  # stored separately in ChromaDB
        if value is None:
            continue
        if isinstance(value, list):
            meta[key] = ", ".join(str(v) for v in value) if value else ""
        elif isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = str(value)
    return meta


class VectorStore:
    """Persistent ChromaDB collection with cosine similarity search."""

    def __init__(
        self,
        persist_dir: str | Path = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        """
        Parameters
        ----------
        persist_dir : str | Path
            Directory for ChromaDB on-disk storage.
        collection_name : str
            Name of the collection.
        """
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready  |  %d existing documents  |  path=%s",
            collection_name,
            self.collection.count(),
            persist_dir,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        batch_size: int = 500,
    ) -> None:
        """
        Upsert chunks with pre-computed embeddings into ChromaDB.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts (must contain ``chunk_id`` and ``text`` keys).
        embeddings : list[list[float]]
            Corresponding embedding vectors (same length as *chunks*).
        batch_size : int
            Max items per ChromaDB upsert call.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length."
            )

        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            self.collection.upsert(
                ids=[c["chunk_id"] for c in batch_chunks],
                documents=[c["text"] for c in batch_chunks],
                embeddings=batch_embeddings,
                metadatas=[_flatten_metadata(c) for c in batch_chunks],
            )

        logger.info("Upserted %d chunk(s) into ChromaDB.", total)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str | None = None,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k most similar chunks.

        Either *query_text* or *query_embedding* must be provided.
        If *query_text* is used, ChromaDB's default embedding function runs
        (which may differ from our sentence-transformers model — prefer
        passing pre-computed embeddings for consistency).

        Parameters
        ----------
        query_text : str, optional
            Raw query string (uses ChromaDB's built-in embedding).
        query_embedding : list[float], optional
            Pre-computed embedding vector (preferred).
        top_k : int
            Number of results.
        where : dict, optional
            ChromaDB metadata filter.

        Returns
        -------
        list[dict]
            Result dicts with ``chunk_id``, ``text``, ``distance``, and metadata.
        """
        kwargs: dict[str, Any] = {"n_results": top_k}
        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding]
        elif query_text is not None:
            kwargs["query_texts"] = [query_text]
        else:
            raise ValueError("Provide either query_text or query_embedding.")

        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        hits: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i in range(len(results["ids"][0])):
            hit: dict[str, Any] = {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i] if results["documents"] else "",
                "distance": results["distances"][0][i] if results["distances"] else None,
            }
            if results["metadatas"] and results["metadatas"][0]:
                hit.update(results["metadatas"][0][i])
            hits.append(hit)

        return hits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        name = self.collection.name
        meta = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata=meta,
        )
        logger.info("Reset collection '%s'.", name)
