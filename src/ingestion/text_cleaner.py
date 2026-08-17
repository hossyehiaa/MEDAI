"""
Text Cleaner — Sanitizes and normalizes extracted text before chunking.

Performs document-level and section-level text cleaning:
1. Removes standalone page-number lines (Arabic, Roman, or prefixed).
2. Removes repeated running headers and footers (>30% page frequency in doc).
3. Removes standalone URL lines (http, https, www).
4. Repairs hyphenated word line breaks ("screen-\\ning" -> "screening").
5. Collapses multiple whitespace, strips trailing/leading spaces, collapses 3+ newlines into 2.
6. Preserves all medical terminology, scores, clinical recommendations, and data tables.

Usage:
    from src.ingestion.text_cleaner import clean_parsed_file

    clean_parsed_file("data/parsed_output.json", "data/cleaned_output.json")
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Regular Expression Patterns
# ──────────────────────────────────────────────────────────────────────

# Standalone page numbers:
# - Arabic digits: "12", "2085", "1"
# - Lowercase Roman numerals (or >= 2 chars): "ii", "iii", "iv", "ix", "x"
# - Prefixed page numbers: "S12", "e12", "A-12", "Page 12", "p. 12"
PAGE_NUMBER_PATTERN: re.Pattern[str] = re.compile(
    r"^(?:"
    r"\d+"
    r"|[ivxlcdm]+"
    r"|[A-Za-z][\-_]?\d+"
    r"|[Pp]age\s*\d+"
    r"|[Pp]\.?\s*\d+"
    r")$"
)

# Standalone URL lines
URL_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?:https?://|www\.)\S+\s*$",
    re.IGNORECASE,
)

# Hyphenated line breaks: "screen-\ning" -> "screening"
HYPHENATED_LINEBREAK_PATTERN: re.Pattern[str] = re.compile(
    r"(\b[A-Za-z]+)-[ \t]*\r?\n[ \t]*([A-Za-z]+\b)"
)

# Protected clinical / data acronyms that must not be removed as repeated headers
PROTECTED_DATA_TOKENS: set[str] = {
    "nr",      # Not Reported (evidence tables)
    "na",      # Not Applicable
    "n/a",
    "b",       # Recommendation Grade B
    "i",       # Recommendation Grade I
    "a",       # Recommendation Grade A
    "c",       # Recommendation Grade C
    "d",       # Recommendation Grade D
    "mdd",
    "phq-9",
    "phq-2",
    "epds",
}


# ──────────────────────────────────────────────────────────────────────
# Header & Footer Detection
# ──────────────────────────────────────────────────────────────────────

def find_repeated_headers_footers(
    sections: list[dict[str, Any]],
    threshold: float = 0.30,
) -> set[str]:
    """
    Identify lines that appear on more than *threshold* of pages in a document.

    Parameters
    ----------
    sections : list[dict]
        All sections belonging to a single document.
    threshold : float, default=0.30
        Minimum page proportion threshold to classify as repeated header/footer.

    Returns
    -------
    set[str]
        Set of stripped line strings to filter out.
    """
    pages = {s.get("page_number", 1) for s in sections}
    total_pages = len(pages)
    if total_pages == 0:
        return set()

    line_pages: dict[str, set[int]] = defaultdict(set)
    for sec in sections:
        page_num = sec.get("page_number", 1)
        for raw_line in sec.get("text_content", "").split("\n"):
            stripped = raw_line.strip()
            if stripped and stripped.lower() not in PROTECTED_DATA_TOKENS:
                line_pages[stripped].add(page_num)

    repeated: set[str] = set()
    for line, page_set in line_pages.items():
        # Must appear on at least 2 distinct pages and exceed threshold
        if len(page_set) >= 2 and (len(page_set) / total_pages > threshold):
            repeated.add(line)

    logger.debug(
        "Detected %d repeated header/footer line(s) across %d pages (threshold=%.0f%%)",
        len(repeated),
        total_pages,
        threshold * 100,
    )
    return repeated


# ──────────────────────────────────────────────────────────────────────
# Section Text Cleaning
# ──────────────────────────────────────────────────────────────────────

def clean_section_text(
    text: str,
    repeated_headers: set[str] | None = None,
) -> str:
    """
    Apply all 6 cleaning rules to a single section's text.

    Parameters
    ----------
    text : str
        Raw text content of the section.
    repeated_headers : set[str], optional
        Set of document-level repeated headers/footers to drop.

    Returns
    -------
    str
        Cleaned text.
    """
    if not text:
        return ""

    repeated_headers = repeated_headers or set()

    # Rule 0 (Defect 9): Remove zero-width spaces and normalize non-breaking spaces
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)
    text = text.replace("\u00a0", " ")

    # Rule 4: Fix hyphenated line breaks ("screen-\ning" -> "screening")
    text = HYPHENATED_LINEBREAK_PATTERN.sub(r"\1\2", text)

    # Process line-by-line for rules 1, 2, 3, and per-line spacing (rule 5)
    cleaned_lines: list[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()

        if not stripped:
            cleaned_lines.append("")
            continue

        # Rule 1: Remove standalone page numbers
        if PAGE_NUMBER_PATTERN.fullmatch(stripped):
            continue

        # Rule 2: Remove repeated running headers and footers
        if stripped in repeated_headers:
            continue

        # Rule 3: Remove standalone URLs
        if URL_PATTERN.fullmatch(stripped):
            continue

        # Rule 5: Collapse multiple horizontal spaces within the line
        collapsed_line = re.sub(r"[^\S\r\n]+", " ", stripped)
        collapsed_line = re.sub(r"[ ]{2,}", " ", collapsed_line)
        cleaned_lines.append(collapsed_line)

    # Rule 5: Recombine and collapse 3+ consecutive newlines into 2
    joined = "\n".join(cleaned_lines)
    collapsed_newlines = re.sub(r"\n{3,}", "\n\n", joined)

    return collapsed_newlines.strip()


# ──────────────────────────────────────────────────────────────────────
# Dataset Cleaning Pipeline
# ──────────────────────────────────────────────────────────────────────

def clean_parsed_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Clean the entire parsed dataset, preserving its exact top-level schema.

    Parameters
    ----------
    data : dict
        Parsed output containing ``summary``, ``sections``, and ``tables``.

    Returns
    -------
    dict
        Cleaned output with identical schema and updated section ``text_content``.
    """
    raw_sections: list[dict[str, Any]] = data.get("sections", [])
    raw_tables: list[dict[str, Any]] = data.get("tables", [])

    # Group sections by document for document-level header frequency analysis
    doc_sections_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sec in raw_sections:
        doc_sections_map[sec.get("document_name", "unknown")].append(sec)

    # Compute repeated headers per document
    doc_headers_map: dict[str, set[str]] = {}
    for doc_name, doc_secs in doc_sections_map.items():
        doc_headers_map[doc_name] = find_repeated_headers_footers(doc_secs, threshold=0.30)

    # Clean each section
    cleaned_sections: list[dict[str, Any]] = []
    for sec in raw_sections:
        doc_name = sec.get("document_name", "unknown")
        headers = doc_headers_map.get(doc_name, set())

        cleaned_sec = dict(sec)
        cleaned_sec["text_content"] = clean_section_text(
            text=sec.get("text_content", ""),
            repeated_headers=headers,
        )
        cleaned_sections.append(cleaned_sec)

    # Construct cleaned output with exact same structure
    cleaned_output: dict[str, Any] = {
        "summary": dict(data.get("summary", {})),
        "sections": cleaned_sections,
        "tables": raw_tables,
    }

    return cleaned_output


def clean_parsed_file(
    input_path: str | Path,
    output_path: str | Path = "data/cleaned_output.json",
) -> Path:
    """
    Read parsed JSON from *input_path*, clean sections, and save to *output_path*.

    Parameters
    ----------
    input_path : str | Path
        Path to the parsed JSON file (e.g. ``parsed_output.json``).
    output_path : str | Path
        Path where cleaned JSON will be written.

    Returns
    -------
    Path
        Resolved path to the saved file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Reading parsed data from '%s' …", input_path)
    with open(input_path, "r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    cleaned_data = clean_parsed_data(raw_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(cleaned_data, fh, indent=2, ensure_ascii=False)

    logger.info("Saved cleaned dataset to '%s'", output_path)
    return output_path
