"""
Table Extractor — Extracts structured tables from USPSTF PDF documents.

Uses pdfplumber to locate and parse tables, with a focus on screening
tool tables (PHQ-2, PHQ-9, Edinburgh, etc.) commonly found in USPSTF
depression-related reports.

Usage:
    from src.ingestion.table_extractor import extract_tables, extract_tables_from_dir

    tables = extract_tables("raw_documents/some_report.pdf")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Screening tool keywords — used to tag relevant tables
# ──────────────────────────────────────────────────────────────────────
SCREENING_TOOL_KEYWORDS: list[str] = [
    "PHQ-2",
    "PHQ-9",
    "PHQ-A",
    "Edinburgh",
    "EPDS",
    "Beck Depression Inventory",
    "BDI",
    "CES-D",
    "GDS",
    "Geriatric Depression Scale",
    "Hamilton",
    "HAM-D",
    "HDRS",
    "K6",
    "Kessler",
    "MADRS",
    "SRQ",
    "Zung",
    "Columbia",
    "C-SSRS",
    "ASQ",
    "sensitivity",
    "specificity",
    "screening",
]

_SCREENING_PATTERN: re.Pattern[str] = re.compile(
    "|".join(re.escape(kw) for kw in SCREENING_TOOL_KEYWORDS),
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedTable:
    """A single table extracted from a PDF page."""

    document_name: str
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    is_screening_table: bool = False
    matched_screening_tools: list[str] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        """Number of data rows (excluding header)."""
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        """Number of columns."""
        return len(self.headers) if self.headers else 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        d = asdict(self)
        d["num_rows"] = self.num_rows
        d["num_cols"] = self.num_cols
        return d


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _clean_cell(cell: Any) -> str:
    """Normalise a single table cell to a trimmed string."""
    if cell is None:
        return ""
    return str(cell).strip().replace("\n", " ")


def _detect_screening_tools(table_text: str) -> list[str]:
    """Return the list of screening-tool keywords found in *table_text*."""
    matches = _SCREENING_PATTERN.findall(table_text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        key = m.lower()
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def _flatten_table_text(headers: list[str], rows: list[list[str]]) -> str:
    """Join all table content into a single string for keyword search."""
    parts = list(headers)
    for row in rows:
        parts.extend(row)
    return " ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def extract_tables(pdf_path: str | Path) -> list[ExtractedTable]:
    """
    Extract all tables from a PDF, tagging those related to screening tools.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    list[ExtractedTable]
        One record per detected table.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    document_name = pdf_path.stem
    logger.info("Extracting tables from '%s' …", pdf_path.name)

    extracted: list[ExtractedTable] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        logger.info("  → %d page(s)", total_pages)

        for page_idx, page in enumerate(pdf.pages):
            page_number = page_idx + 1
            tables = page.extract_tables()

            if not tables:
                continue

            for tbl_idx, raw_table in enumerate(tables):
                if not raw_table or len(raw_table) < 1:
                    continue

                # First row → headers; rest → data rows
                raw_headers = raw_table[0]
                raw_rows = raw_table[1:] if len(raw_table) > 1 else []

                headers = [_clean_cell(c) for c in raw_headers]
                rows = [[_clean_cell(c) for c in row] for row in raw_rows]

                # Remove fully-empty rows
                rows = [r for r in rows if any(cell for cell in r)]

                # Screening tool detection
                full_text = _flatten_table_text(headers, rows)
                matched_tools = _detect_screening_tools(full_text)
                is_screening = len(matched_tools) > 0

                table_record = ExtractedTable(
                    document_name=document_name,
                    page_number=page_number,
                    table_index=tbl_idx,
                    headers=headers,
                    rows=rows,
                    is_screening_table=is_screening,
                    matched_screening_tools=matched_tools,
                )
                extracted.append(table_record)

                logger.debug(
                    "  Page %d, Table %d: %d×%d%s",
                    page_number,
                    tbl_idx,
                    table_record.num_rows,
                    table_record.num_cols,
                    "  ★ screening" if is_screening else "",
                )

    screening_count = sum(1 for t in extracted if t.is_screening_table)
    logger.info(
        "  → Done: %d table(s) extracted  |  %d screening table(s)",
        len(extracted),
        screening_count,
    )
    return extracted


def extract_tables_from_dir(directory: str | Path) -> list[ExtractedTable]:
    """
    Extract tables from every PDF in *directory*.

    PDFs that cannot be opened are skipped with a warning.

    Parameters
    ----------
    directory : str | Path
        Directory containing PDF files.

    Returns
    -------
    list[ExtractedTable]
        Combined tables from all successfully processed PDFs.
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

    all_tables: list[ExtractedTable] = []
    for pdf_path in pdf_files:
        try:
            tables = extract_tables(pdf_path)
            all_tables.extend(tables)
        except Exception:
            logger.warning("Failed to extract tables from '%s' — skipping.", pdf_path.name, exc_info=True)

    logger.info("Total: %d table(s) from %d PDF(s).", len(all_tables), len(pdf_files))
    return all_tables
