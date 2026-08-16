"""
PDF Parser — Extracts structured, section-aware text from USPSTF PDF documents.

Uses PyMuPDF (fitz) to parse pages, detect known USPSTF section headers,
extract USPSTF recommendation grades, and return structured records.

Usage:
    from src.ingestion.pdf_parser import parse_pdf, parse_all_pdfs

    sections = parse_pdf("raw_documents/some_report.pdf")
    all_sections = parse_all_pdfs("raw_documents/")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Known USPSTF section headers (order matters — matched top-down)
# ──────────────────────────────────────────────────────────────────────
KNOWN_SECTIONS: list[str] = [
    "What does the USPSTF recommend",
    "Patient Population Under Consideration",
    "Update of Previous Recommendations",
    "Treatment/Interventions",
    "Research Needs and Gaps",
    "Recommendations of Others",
    "Practice Considerations",
    "Clinical Considerations",
    "Supporting Evidence",
    "USPSTF Assessment",
    "Screening Tests",
    "Recommendation",
    "Rationale",
    "Importance",
    "References",
]

# Pre-compile a single regex that matches any known header.
# Each header is escaped and made case-insensitive.  We look for headers
# that start at the beginning of a line (after optional whitespace) and
# are optionally followed by a colon, period, or newline.
_HEADER_PATTERN: re.Pattern[str] = re.compile(
    r"^[ \t]*("
    + "|".join(re.escape(h) for h in KNOWN_SECTIONS)
    + r")[ \t]*[:\.\-]?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

# USPSTF letter grades
_GRADE_PATTERN: re.Pattern[str] = re.compile(
    r"""
    (?:                           # non-capturing group for context
        \bgrade\s+                # "grade A", "grade B" …
      | \brecommendation\s+grade\s*[:\s]+  # "recommendation grade: B"
      | \b[""'(]\s*              # grade in quotes or parens
    )
    ([ABCDI])                     # capture the grade letter
    (?:\s+(?:recommendation|statement))?  # optional trailing word
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pattern to detect Table-of-Contents pages (lines like "Section ..... 12")
_TOC_LINE_PATTERN: re.Pattern[str] = re.compile(
    r"\.{4,}\s*\d+\s*$", re.MULTILINE
)

# Minimum text length to consider a page non-empty
_MIN_PAGE_TEXT_LENGTH = 20

# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ParsedSection:
    """A single section extracted from a USPSTF PDF."""

    document_name: str
    page_number: int
    section_name: str
    text_content: str
    detected_grades: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _is_toc_page(text: str) -> bool:
    """Return *True* if the page looks like a Table of Contents."""
    toc_hits = _TOC_LINE_PATTERN.findall(text)
    # If ≥3 TOC-like lines, treat the whole page as TOC
    return len(toc_hits) >= 3


def _detect_grades(text: str) -> list[str]:
    """Extract unique USPSTF letter grades from *text*."""
    matches = _GRADE_PATTERN.findall(text)
    # Normalise to uppercase, deduplicate, preserve discovery order
    seen: set[str] = set()
    grades: list[str] = []
    for m in matches:
        g = m.upper()
        if g not in seen:
            seen.add(g)
            grades.append(g)
    return grades


def _split_into_sections(
    page_text: str,
    document_name: str,
    page_number: int,
    carry_section: str,
) -> tuple[list[ParsedSection], str]:
    """
    Split a single page's text into sections based on known headers.

    Parameters
    ----------
    page_text : str
        Raw text of the page.
    document_name : str
        Stem of the source PDF filename.
    page_number : int
        1-indexed page number.
    carry_section : str
        The section name carried forward from the previous page
        (handles headers that span page boundaries).

    Returns
    -------
    tuple[list[ParsedSection], str]
        A list of parsed sections for this page, and the name of the
        last active section (to carry into the next page).
    """
    sections: list[ParsedSection] = []
    current_section = carry_section

    # Find all header matches with their positions
    matches = list(_HEADER_PATTERN.finditer(page_text))

    if not matches:
        # Entire page belongs to the carried-forward section
        content = page_text.strip()
        if content:
            sections.append(
                ParsedSection(
                    document_name=document_name,
                    page_number=page_number,
                    section_name=current_section,
                    text_content=content,
                    detected_grades=_detect_grades(content),
                )
            )
        return sections, current_section

    # Text before the first header belongs to the carry section
    pre_header_text = page_text[: matches[0].start()].strip()
    if pre_header_text:
        sections.append(
            ParsedSection(
                document_name=document_name,
                page_number=page_number,
                section_name=current_section,
                text_content=pre_header_text,
                detected_grades=_detect_grades(pre_header_text),
            )
        )

    # Process each header and the text that follows it
    for i, match in enumerate(matches):
        header_name = match.group(1).strip()
        # Normalise header to title case for consistency
        header_name = _normalise_header(header_name)
        current_section = header_name

        # Text runs from end of this header to start of next (or end of page)
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        content = page_text[text_start:text_end].strip()

        if content:
            sections.append(
                ParsedSection(
                    document_name=document_name,
                    page_number=page_number,
                    section_name=current_section,
                    text_content=content,
                    detected_grades=_detect_grades(content),
                )
            )

    return sections, current_section


def _normalise_header(header: str) -> str:
    """Map a matched header back to its canonical form in KNOWN_SECTIONS."""
    header_lower = header.lower().strip()
    for canonical in KNOWN_SECTIONS:
        if canonical.lower() == header_lower:
            return canonical
    # Fallback: title-case the raw match
    return header.strip().title()


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str | Path) -> list[ParsedSection]:
    """
    Parse a single USPSTF PDF and return structured section records.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    list[ParsedSection]
        One record per detected section (or per page if no headers found).

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_name = pdf_path.stem
    logger.info("Parsing '%s' …", pdf_path.name)

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info("  → %d page(s)", total_pages)

    all_sections: list[ParsedSection] = []
    carry_section = "General"
    skipped_toc = 0
    skipped_empty = 0

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_number = page_idx + 1
        text = page.get_text("text")

        # ── Skip empty / image-only pages ─────────────────────────
        if not text or len(text.strip()) < _MIN_PAGE_TEXT_LENGTH:
            skipped_empty += 1
            logger.debug("  Page %d: empty / image-only — skipped.", page_number)
            continue

        # ── Skip TOC pages ────────────────────────────────────────
        if _is_toc_page(text):
            skipped_toc += 1
            logger.debug("  Page %d: detected as TOC — skipped.", page_number)
            continue

        # ── Extract sections ──────────────────────────────────────
        page_sections, carry_section = _split_into_sections(
            page_text=text,
            document_name=document_name,
            page_number=page_number,
            carry_section=carry_section,
        )
        all_sections.extend(page_sections)

    doc.close()

    logger.info(
        "  → Done: %d section(s) extracted  |  %d TOC page(s) skipped  |  %d empty page(s) skipped",
        len(all_sections),
        skipped_toc,
        skipped_empty,
    )
    return all_sections


def parse_all_pdfs(directory: str | Path) -> list[ParsedSection]:
    """
    Parse every PDF in *directory* and return a combined list of sections.

    PDFs that cannot be found or opened are skipped with a warning.

    Parameters
    ----------
    directory : str | Path
        Directory containing PDF files.

    Returns
    -------
    list[ParsedSection]
        Combined sections from all successfully parsed PDFs.
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error("Directory not found: %s", directory)
        return []

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", directory)
        return []

    logger.info("Found %d PDF(s) in '%s'", len(pdf_files), directory)

    all_sections: list[ParsedSection] = []
    for pdf_path in pdf_files:
        try:
            sections = parse_pdf(pdf_path)
            all_sections.extend(sections)
        except Exception:
            logger.warning("Failed to parse '%s' — skipping.", pdf_path.name, exc_info=True)

    logger.info("Total: %d section(s) from %d PDF(s).", len(all_sections), len(pdf_files))
    return all_sections
