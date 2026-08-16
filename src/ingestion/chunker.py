"""
Section-Aware Chunker — Splits cleaned medical document sections into
retrieval-ready chunks with rich metadata propagation and exact page citations.

Key design decisions:
  • Consecutive sections sharing (document_name, section_name) are merged
    before chunking with [[PAGE:<n>]] sentinel lines.
  • Chunks extract precise start_page and end_page from sentinel markers,
    then strip sentinels from stored text (e.g. References cites p.181, not p.95-708).
  • Tables from the parsed data are converted into text chunks tagged
    with ``is_table=True`` and their matched screening tools.
  • Chunks shorter than ``MIN_CHUNK_CHARS`` are discarded.
  • Token counting uses ``tiktoken`` (cl100k_base) for accurate sizing.
  • Screening tool regex detector ensures has_screening_tools <=> non-empty screening_tools.

Usage:
    from src.ingestion.chunker import chunk_cleaned_data, load_cleaned_data

    data = load_cleaned_data()
    chunks = chunk_cleaned_data(data)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import settings from the central config
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import (
    CLEANED_OUTPUT_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    CHUNK_SEPARATORS,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────────────────────────────
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Return the number of tokens in *text* (cl100k_base)."""
    return len(_ENCODING.encode(text, disallowed_special=()))


# ──────────────────────────────────────────────────────────────────────
# Screening Tools Regex Patterns
# ──────────────────────────────────────────────────────────────────────
_SCREENING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PHQ-2", re.compile(r"\bPHQ-?2\b", re.IGNORECASE)),
    ("PHQ-9", re.compile(r"\bPHQ-?9\b", re.IGNORECASE)),
    ("PHQ-10", re.compile(r"\bPHQ-?10\b", re.IGNORECASE)),
    ("EPDS", re.compile(r"\bEPDS\b", re.IGNORECASE)),
    ("Edinburgh", re.compile(r"\bEdinburgh\b", re.IGNORECASE)),
    ("BDI", re.compile(r"\bBDI\b", re.IGNORECASE)),
    ("CES-D", re.compile(r"\bCES-?D\b", re.IGNORECASE)),
    ("GDS", re.compile(r"\bGDS\b", re.IGNORECASE)),
    ("C-SSRS", re.compile(r"\bC-?SSRS\b", re.IGNORECASE)),
    ("ASQ", re.compile(r"\bASQ\b", re.IGNORECASE)),
]


def _detect_screening_tools(text: str) -> list[str]:
    """Return canonical screening-tool names found in *text* via regex."""
    found: list[str] = []
    for name, pattern in _SCREENING_PATTERNS:
        if pattern.search(text):
            found.append(name)
    return sorted(list(set(found)))


# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A retrieval-ready chunk with full metadata."""

    chunk_id: str
    document_name: str
    section_name: str
    start_page: int
    end_page: int
    text: str
    token_count: int
    char_count: int
    grades: list[str] = field(default_factory=list)
    topic: str = ""
    has_screening_tools: bool = False
    screening_tools: list[str] = field(default_factory=list)
    is_table: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_chunk_id(document_name: str, section_name: str, index: int, text: str) -> str:
    """Deterministic short hash for a chunk."""
    payload = f"{document_name}::{section_name}::{index}::{text[:128]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _detect_topic(text: str, section_name: str) -> str:
    """Infer a lightweight topic tag from keywords in *text* and *section_name*."""
    combined = f"{section_name} {text}".lower()
    if any(k in combined for k in ("screening", "screen")):
        return "screening"
    if any(k in combined for k in ("treatment", "intervention", "therapy", "cbt", "ssri", "antidepressant")):
        return "treatment"
    if any(k in combined for k in ("accuracy", "sensitivity", "specificity", "test accuracy")):
        return "test_accuracy"
    if any(k in combined for k in ("harm", "adverse", "side effect")):
        return "harms"
    if any(k in combined for k in ("recommendation", "grade")):
        return "recommendation"
    if any(k in combined for k in ("pregnant", "postpartum", "perinatal")):
        return "perinatal"
    if any(k in combined for k in ("suicide", "self-harm", "suicidal")):
        return "suicide_risk"
    if "reference" in combined:
        return "references"
    return "general"


# ──────────────────────────────────────────────────────────────────────
# Section Grouping with Page Sentinels
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _MergedSection:
    """Consecutive sections merged by (document_name, section_name)."""

    document_name: str
    section_name: str
    start_page: int
    end_page: int
    text: str
    grades: list[str] = field(default_factory=list)


def _group_sections(sections: list[dict[str, Any]]) -> list[_MergedSection]:
    """
    Merge consecutive sections sharing the same (document_name, section_name),
    injecting [[PAGE:<n>]] sentinel lines for exact sub-page tracking.
    """
    groups: list[_MergedSection] = []
    prev_key: tuple[str, str] | None = None
    current: _MergedSection | None = None

    for sec in sections:
        page_num = sec["page_number"]
        key = (sec["document_name"], sec["section_name"])
        page_sentinel_text = f"[[PAGE:{page_num}]]\n{sec['text_content']}"

        if key == prev_key and current is not None:
            current.text += f"\n\n{page_sentinel_text}"
            current.end_page = page_num
            for g in sec.get("detected_grades", []):
                if g not in current.grades:
                    current.grades.append(g)
        else:
            if current is not None:
                groups.append(current)
            current = _MergedSection(
                document_name=sec["document_name"],
                section_name=sec["section_name"],
                start_page=page_num,
                end_page=page_num,
                text=page_sentinel_text,
                grades=list(sec.get("detected_grades", [])),
            )
            prev_key = key

    if current is not None:
        groups.append(current)

    logger.info(
        "Grouped %d sections into %d merged section(s) with page sentinels.",
        len(sections),
        len(groups),
    )
    return groups


# ──────────────────────────────────────────────────────────────────────
# Table → Text Conversion
# ──────────────────────────────────────────────────────────────────────

def _table_to_text(table: dict[str, Any]) -> str:
    """Render a parsed table as a readable text block."""
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    lines: list[str] = []

    if headers:
        lines.append("| " + " | ".join(str(h) for h in headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Core Chunking with Exact Page Citation Resolution
# ──────────────────────────────────────────────────────────────────────

def _create_splitter() -> RecursiveCharacterTextSplitter:
    """Create a token-aware text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_count,
        separators=CHUNK_SEPARATORS,
    )


def chunk_sections(merged_sections: list[_MergedSection]) -> list[Chunk]:
    """
    Split merged sections into chunks using RecursiveCharacterTextSplitter.
    Resolves exact per-page boundaries from [[PAGE:<n>]] sentinels and strips them.

    Parameters
    ----------
    merged_sections : list[_MergedSection]
        Output of :func:`_group_sections`.

    Returns
    -------
    list[Chunk]
        Chunks with exact start_page and end_page metadata.
    """
    splitter = _create_splitter()
    all_chunks: list[Chunk] = []
    skipped = 0

    for group in merged_sections:
        raw_splits = splitter.split_text(group.text)
        current_running_page = group.start_page

        for idx, fragment in enumerate(raw_splits):
            # Extract page markers
            markers = [int(p) for p in re.findall(r"\[\[PAGE:(\d+)\]\]", fragment)]

            if markers:
                # Check if text exists before the first marker in this fragment
                prefix_before_first_marker = fragment.split("[[PAGE:")[0].strip()
                chunk_start_page = current_running_page if prefix_before_first_marker else markers[0]
                chunk_end_page = markers[-1]
                current_running_page = markers[-1]
            else:
                chunk_start_page = current_running_page
                chunk_end_page = current_running_page

            # Strip sentinels from the final chunk text
            clean_fragment = re.sub(r"\[\[PAGE:\d+\]\]\n?", "", fragment).strip()

            if len(clean_fragment) < MIN_CHUNK_CHARS:
                skipped += 1
                continue

            tools = _detect_screening_tools(clean_fragment)
            has_tools = len(tools) > 0

            chunk = Chunk(
                chunk_id=_make_chunk_id(group.document_name, group.section_name, idx, clean_fragment),
                document_name=group.document_name,
                section_name=group.section_name,
                start_page=chunk_start_page,
                end_page=chunk_end_page,
                text=clean_fragment,
                token_count=_token_count(clean_fragment),
                char_count=len(clean_fragment),
                grades=list(group.grades),
                topic=_detect_topic(clean_fragment, group.section_name),
                has_screening_tools=has_tools,
                screening_tools=tools,
                is_table=False,
            )
            all_chunks.append(chunk)

    logger.info(
        "Produced %d text chunk(s) from %d group(s) | %d skipped (< %d chars).",
        len(all_chunks),
        len(merged_sections),
        skipped,
        MIN_CHUNK_CHARS,
    )
    return all_chunks


def chunk_tables(tables: list[dict[str, Any]]) -> list[Chunk]:
    """
    Convert parsed tables into individual chunks with robust screening tool tags.
    """
    table_chunks: list[Chunk] = []
    skipped = 0

    for tbl in tables:
        if tbl.get("num_rows", 0) < 1:
            skipped += 1
            continue

        text = _table_to_text(tbl)
        if len(text.strip()) < MIN_CHUNK_CHARS:
            skipped += 1
            continue

        # Combine parsed table metadata tools with regex detected tools
        raw_tools = tbl.get("matched_screening_tools", [])
        regex_tools = _detect_screening_tools(text)
        combined_tools = sorted(list(set(raw_tools + regex_tools)))
        has_tools = len(combined_tools) > 0 or tbl.get("is_screening_table", False)
        if has_tools and not combined_tools:
            combined_tools = ["Screening Tool"]

        chunk = Chunk(
            chunk_id=_make_chunk_id(
                tbl["document_name"],
                f"table_p{tbl['page_number']}_t{tbl['table_index']}",
                0,
                text,
            ),
            document_name=tbl["document_name"],
            section_name="Table",
            start_page=tbl["page_number"],
            end_page=tbl["page_number"],
            text=text,
            token_count=_token_count(text),
            char_count=len(text),
            grades=[],
            topic="table",
            has_screening_tools=has_tools,
            screening_tools=combined_tools if has_tools else [],
            is_table=True,
        )
        table_chunks.append(chunk)

    logger.info(
        "Produced %d table chunk(s) | %d skipped (empty or < %d chars).",
        len(table_chunks),
        skipped,
        MIN_CHUNK_CHARS,
    )
    return table_chunks


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def load_cleaned_data(path: str | Path | None = None) -> dict[str, Any]:
    """Load the cleaned JSON dataset."""
    path = Path(path) if path else CLEANED_OUTPUT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Cleaned output not found at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def chunk_cleaned_data(
    data: dict[str, Any],
    include_tables: bool = True,
) -> list[Chunk]:
    """
    Full chunking pipeline: group sections with page sentinels → split → add tables.

    Parameters
    ----------
    data : dict
        The loaded cleaned output (with ``sections`` and ``tables`` keys).
    include_tables : bool
        Whether to include table chunks.

    Returns
    -------
    list[Chunk]
        All chunks ready for embedding.
    """
    sections = data.get("sections", [])
    tables = data.get("tables", [])

    # 1. Group consecutive same-doc / same-section entries with page sentinels
    merged = _group_sections(sections)

    # 2. Chunk text sections with exact page citations
    text_chunks = chunk_sections(merged)

    # 3. Chunk tables
    table_chunks: list[Chunk] = []
    if include_tables:
        table_chunks = chunk_tables(tables)

    all_chunks = text_chunks + table_chunks
    logger.info("Total chunks: %d (text=%d, tables=%d)", len(all_chunks), len(text_chunks), len(table_chunks))
    return all_chunks


def save_chunks(chunks: list[Chunk], path: str | Path | None = None) -> Path:
    """Persist chunks as JSON."""
    from configs.settings import CHUNKS_OUTPUT_PATH

    path = Path(path) if path else CHUNKS_OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([c.to_dict() for c in chunks], fh, indent=2, ensure_ascii=False)
    logger.info("Saved %d chunks to %s", len(chunks), path)
    return path
