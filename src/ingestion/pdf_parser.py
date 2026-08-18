"""
PDF Parser — Extracts structured, section-aware text from USPSTF PDF documents.

Implements dual-backend extraction:
1. Primary extractor using pdfplumber for standard multi-page documents.
2. PyMuPDF (fitz) fallback extractor when a PDF yields fewer than 15 sections OR fewer than 2,000 characters
   (targeting formatted multi-column summaries such as the 2-page clinician summary).
3. Explicit logging of which backend (pdfplumber vs PyMuPDF) was selected per document.

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
import pdfplumber

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Known USPSTF section headers (order matters — matched top-down)
# ──────────────────────────────────────────────────────────────────────
KNOWN_SECTIONS: list[str] = [
    # Clinician summary subheadings & questions
    "What does the USPSTF recommend",
    "To whom does this recommendation apply",
    "What's new",
    "What’s new",
    "How to implement this recommendation",
    "What additional information should clinicians know about this recommendation",
    "What additional information should clinicians know",
    "Why is this recommendation and topic important",
    "What are other relevant USPSTF recommendations",
    "What are additional tools and resources",
    "Where to read the full recommendation statement",
    "Where to get more information",
    # Standard USPSTF Evidence Report / Recommendation Sections
    "Patient Population Under Consideration",
    "Update of Previous Recommendations",
    "Treatment/Interventions",
    "Treatment and Interventions",
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

# Pre-compile header pattern supporting punctuation and question marks
_HEADER_PATTERN: re.Pattern[str] = re.compile(
    r"^[ \t]*("
    + "|".join(re.escape(h) for h in KNOWN_SECTIONS)
    + r")[ \t]*[\?\:\.\-]?[ \t]*$",
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

# Thresholds for fallback triggering
_MIN_SECTIONS_THRESHOLD = 15
_MIN_CHARS_THRESHOLD = 2000


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

def _normalize_extracted_unicode(text: str) -> str:
    """Normalize Unicode extracted from PDFs — replace ligatures, fix smart quotes, strip control chars."""
    import unicodedata
    # Normalize to NFC form
    text = unicodedata.normalize("NFC", text)
    # Replace common ligatures
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Remove control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def _is_toc_page(text: str) -> bool:
    """Return *True* if the page looks like a Table of Contents."""
    toc_hits = _TOC_LINE_PATTERN.findall(text)
    return len(toc_hits) >= 3


def _detect_grades(text: str) -> list[str]:
    """Extract unique USPSTF letter grades from *text*."""
    matches = _GRADE_PATTERN.findall(text)
    seen: set[str] = set()
    grades: list[str] = []
    for m in matches:
        g = m.upper()
        if g not in seen:
            seen.add(g)
            grades.append(g)
    return grades


def _normalise_header(header: str) -> str:
    """Map a matched header back to its canonical form in KNOWN_SECTIONS."""
    header_lower = header.lower().strip().rstrip("?:.- ")
    for canonical in KNOWN_SECTIONS:
        if canonical.lower() == header_lower:
            return canonical
    return header.strip().title()


def _split_into_sections(
    page_text: str,
    document_name: str,
    page_number: int,
    carry_section: str,
) -> tuple[list[ParsedSection], str]:
    """Split a single page's text into sections based on known headers."""
    sections: list[ParsedSection] = []
    current_section = carry_section

    matches = list(_HEADER_PATTERN.finditer(page_text))

    if not matches:
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

    for i, match in enumerate(matches):
        header_name = match.group(1).strip()
        header_name = _normalise_header(header_name)
        current_section = header_name

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


# ──────────────────────────────────────────────────────────────────────
# Backend Extractors
# ──────────────────────────────────────────────────────────────────────

def _parse_with_pdfplumber(pdf_path: Path) -> list[ParsedSection]:
    """Extract sections using pdfplumber."""
    document_name = pdf_path.stem
    sections: list[ParsedSection] = []
    carry_section = "General"

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_number = page_idx + 1
            text = _normalize_extracted_unicode(page.extract_text() or "")

            if len(text.strip()) < _MIN_PAGE_TEXT_LENGTH or _is_toc_page(text):
                continue

            page_sections, carry_section = _split_into_sections(
                page_text=text,
                document_name=document_name,
                page_number=page_number,
                carry_section=carry_section,
            )
            sections.extend(page_sections)

    return sections


def _parse_with_pymupdf(pdf_path: Path) -> list[ParsedSection]:
    """Extract sections using PyMuPDF (fitz) with fine-grained block parsing."""
    document_name = pdf_path.stem
    sections: list[ParsedSection] = []
    carry_section = "General"

    doc = fitz.open(str(pdf_path))
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1
        raw_text = page.get_text("text")
        text = _normalize_extracted_unicode(raw_text)

        if not text or len(text.strip()) < _MIN_PAGE_TEXT_LENGTH or _is_toc_page(text):
            continue

        # Normalize multiline wrapped question headers
        text = re.sub(
            r"What additional information should clinicians know about\s*\n\s*this recommendation\??",
            "What additional information should clinicians know about this recommendation?",
            text,
            flags=re.IGNORECASE,
        )

        lines = text.split("\n")
        current_sec = carry_section
        current_content: list[str] = []
        block_idx = 1

        for line in lines:
            l_str = line.strip()
            if not l_str:
                continue

            # Detect question / main header
            is_hdr = False
            for h in KNOWN_SECTIONS:
                if l_str.lower().startswith(h.lower()) or l_str.lower() == h.lower():
                    is_hdr = True
                    if current_content:
                        txt = "\n".join(current_content).strip()
                        if len(txt) >= 35:
                            sec_title = f"{current_sec} (Part {block_idx})" if block_idx > 1 else current_sec
                            sections.append(
                                ParsedSection(
                                    document_name=document_name,
                                    page_number=page_number,
                                    section_name=sec_title,
                                    text_content=txt,
                                    detected_grades=_detect_grades(txt),
                                )
                            )
                        current_content = []
                    current_sec = _normalise_header(l_str)
                    block_idx = 1
                    current_content.append(l_str)
                    break

            if is_hdr:
                continue

            # Detect discrete bullet points for granular clinician summary coverage
            if (l_str.startswith("•") or l_str.startswith("-") or l_str.startswith("*")) and current_content:
                txt = "\n".join(current_content).strip()
                if len(txt) >= 50:
                    sec_title = f"{current_sec} (Part {block_idx})" if block_idx > 1 else current_sec
                    sections.append(
                        ParsedSection(
                            document_name=document_name,
                            page_number=page_number,
                            section_name=sec_title,
                            text_content=txt,
                            detected_grades=_detect_grades(txt),
                        )
                    )
                    block_idx += 1
                    current_content = []

            current_content.append(l_str)

        if current_content:
            txt = "\n".join(current_content).strip()
            if len(txt) >= 35:
                sec_title = f"{current_sec} (Part {block_idx})" if block_idx > 1 else current_sec
                sections.append(
                    ParsedSection(
                        document_name=document_name,
                        page_number=page_number,
                        section_name=sec_title,
                        text_content=txt,
                        detected_grades=_detect_grades(txt),
                    )
                )

        carry_section = current_sec

    doc.close()
    return sections


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str | Path) -> list[ParsedSection]:
    """
    Parse a single USPSTF PDF and return structured section records.

    Tries pdfplumber first; if fewer than 15 sections OR fewer than 2000 chars
    are extracted, automatically falls back to PyMuPDF with block parsing.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Primary attempt: pdfplumber
    sections = _parse_with_pdfplumber(pdf_path)
    total_chars = sum(len(s.text_content) for s in sections)

    if len(sections) < _MIN_SECTIONS_THRESHOLD or total_chars < _MIN_CHARS_THRESHOLD:
        logger.info(
            "PDF '%s': pdfplumber yielded %d sections (%d chars) < threshold. Falling back to PyMuPDF.",
            pdf_path.name,
            len(sections),
            total_chars,
        )
        sections = _parse_with_pymupdf(pdf_path)
        backend_used = "PyMuPDF"
    else:
        backend_used = "pdfplumber"

    logger.info(
        "Parsed '%s' using backend [%s] → %d sections (%d total characters)",
        pdf_path.name,
        backend_used,
        len(sections),
        sum(len(s.text_content) for s in sections),
    )
    return sections


def parse_all_pdfs(directory: str | Path) -> list[ParsedSection]:
    """Parse every PDF in *directory* and return a combined list of sections."""
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
