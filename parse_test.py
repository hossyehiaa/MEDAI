"""
Parse Test — Runs the PDF parser on all documents in raw_documents/ and
prints a diagnostic summary.

Usage:
    python parse_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path so we can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.pdf_parser import parse_all_pdfs, ParsedSection
from src.ingestion.table_extractor import extract_tables_from_dir, ExtractedTable

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
RAW_DOCS_DIR = Path("raw_documents")
OUTPUT_PATH = Path("data/parsed_output.json")
PREVIEW_CHARS = 200

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-38s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger("parse_test")


def _separator(char: str = "─", width: int = 80) -> str:
    return char * width


def run_parser_test() -> None:
    """Run the full parser test suite."""

    print(f"\n{'=' * 80}")
    print("  medAI — PDF Parser Test")
    print(f"{'=' * 80}\n")

    # ── 1. Parse all PDFs ────────────────────────────────────────────
    print(f"📂  Source directory: {RAW_DOCS_DIR.resolve()}\n")
    sections: list[ParsedSection] = parse_all_pdfs(RAW_DOCS_DIR)

    if not sections:
        print("⚠️  No sections were extracted. Check that PDFs exist in raw_documents/.")
        return

    # ── 2. Aggregate statistics ──────────────────────────────────────
    docs: dict[str, list[ParsedSection]] = defaultdict(list)
    for sec in sections:
        docs[sec.document_name].append(sec)

    total_pages = len({(s.document_name, s.page_number) for s in sections})
    total_sections = len(sections)
    all_grades: set[str] = set()
    for s in sections:
        all_grades.update(s.detected_grades)

    # ── 3. Print summary ─────────────────────────────────────────────
    print(f"\n{_separator('═')}")
    print("  SUMMARY")
    print(_separator("═"))
    print(f"  Documents parsed  : {len(docs)}")
    print(f"  Total pages       : {total_pages}")
    print(f"  Total sections    : {total_sections}")
    print(f"  USPSTF grades     : {sorted(all_grades) if all_grades else 'none detected'}")
    print(_separator("─"))

    # Per-document breakdown
    for doc_name, doc_sections in sorted(docs.items()):
        doc_pages = len({s.page_number for s in doc_sections})
        unique_section_names = sorted({s.section_name for s in doc_sections})
        doc_grades = set()
        for s in doc_sections:
            doc_grades.update(s.detected_grades)

        print(f"\n  📄 {doc_name}")
        print(f"     Pages processed : {doc_pages}")
        print(f"     Sections found  : {len(doc_sections)}")
        print(f"     Section names   : {', '.join(unique_section_names)}")
        if doc_grades:
            print(f"     Detected grades : {sorted(doc_grades)}")

    # ── 4. Preview each section ──────────────────────────────────────
    print(f"\n{_separator('═')}")
    print("  SECTION PREVIEWS")
    print(_separator("═"))

    for sec in sections:
        preview = sec.text_content[:PREVIEW_CHARS]
        if len(sec.text_content) > PREVIEW_CHARS:
            preview += " …"

        grade_tag = f"  [grades: {', '.join(sec.detected_grades)}]" if sec.detected_grades else ""
        print(f"\n  [{sec.document_name}] p.{sec.page_number} — §{sec.section_name}{grade_tag}")
        print(f"  {_separator('·', 76)}")
        # Indent the preview for readability
        for line in preview.split("\n"):
            print(f"    {line}")

    # ── 5. Extract tables ────────────────────────────────────────────
    print(f"\n{_separator('═')}")
    print("  TABLE EXTRACTION")
    print(_separator("═"))

    tables: list[ExtractedTable] = extract_tables_from_dir(RAW_DOCS_DIR)

    if tables:
        screening_tables = [t for t in tables if t.is_screening_table]
        print(f"\n  Tables found       : {len(tables)}")
        print(f"  Screening tables   : {len(screening_tables)}")

        for tbl in tables:
            tag = "  ★ SCREENING" if tbl.is_screening_table else ""
            print(f"\n  📊 [{tbl.document_name}] p.{tbl.page_number}, table {tbl.table_index}{tag}")
            print(f"     Size    : {tbl.num_rows} rows × {tbl.num_cols} cols")
            print(f"     Headers : {tbl.headers}")
            if tbl.matched_screening_tools:
                print(f"     Tools   : {tbl.matched_screening_tools}")
            # Show first 2 data rows as preview
            for row in tbl.rows[:2]:
                print(f"     Row     : {row}")
            if tbl.num_rows > 2:
                print(f"     … and {tbl.num_rows - 2} more row(s)")
    else:
        print("\n  No tables found.")

    # ── 6. Save full output ──────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "summary": {
            "documents_parsed": len(docs),
            "total_pages": total_pages,
            "total_sections": total_sections,
            "detected_grades": sorted(all_grades),
            "total_tables": len(tables),
            "screening_tables": len([t for t in tables if t.is_screening_table]),
        },
        "sections": [s.to_dict() for s in sections],
        "tables": [t.to_dict() for t in tables],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)

    print(f"\n{_separator('═')}")
    print(f"  ✅ Full output saved to: {OUTPUT_PATH.resolve()}")
    print(_separator("═"))
    print()


if __name__ == "__main__":
    run_parser_test()
