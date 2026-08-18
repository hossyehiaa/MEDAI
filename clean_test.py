"""
Clean Test — Runs the text cleaner on parsed_output.json and displays
diagnostic before/after statistics and visual text comparisons.

Usage:
    python clean_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.text_cleaner import clean_parsed_data

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
INPUT_PATH = Path("parsed_output.json")
if not INPUT_PATH.exists():
    INPUT_PATH = Path("data/parsed_output.json")

OUTPUT_PATH = Path("data/cleaned_output.json")
PREVIEW_CHARS = 400

# ──────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger("clean_test")


def _separator(char: str = "─", width: int = 80) -> str:
    return char * width


def run_clean_test() -> None:
    """Execute the text cleaning pipeline and display diagnostics."""
    print(f"\n{'=' * 80}")
    print("  medAI — Text Cleaning & Normalization Test")
    print(f"{'=' * 80}\n")

    if not INPUT_PATH.exists():
        print(f"❌ Error: Input file not found at '{INPUT_PATH}'. Run parse_test.py first.")
        return

    print(f"📂  Input file  : {INPUT_PATH.resolve()}")
    print(f"💾  Output file : {OUTPUT_PATH.resolve()}\n")

    with open(INPUT_PATH, "r", encoding="utf-8") as fh:
        raw_data = json.load(fh)

    raw_sections: list[dict] = raw_data.get("sections", [])
    if not raw_sections:
        print("⚠️  No sections found in input data.")
        return

    # Run cleaning pipeline
    cleaned_data = clean_parsed_data(raw_data)
    cleaned_sections: list[dict] = cleaned_data.get("sections", [])

    # Group by document
    raw_by_doc: dict[str, list[dict]] = defaultdict(list)
    cleaned_by_doc: dict[str, list[dict]] = defaultdict(list)

    for sec in raw_sections:
        raw_by_doc[sec["document_name"]].append(sec)
    for sec in cleaned_sections:
        cleaned_by_doc[sec["document_name"]].append(sec)

    # ── 1. Per-Document Statistics ───────────────────────────────────
    print(_separator("═"))
    print("  CLEANING STATISTICS PER DOCUMENT")
    print(_separator("═"))

    total_before = 0
    total_after = 0

    for doc_name in sorted(raw_by_doc.keys()):
        raw_list = raw_by_doc[doc_name]
        clean_list = cleaned_by_doc[doc_name]

        chars_before = sum(len(s.get("text_content", "")) for s in raw_list)
        chars_after = sum(len(s.get("text_content", "")) for s in clean_list)
        removed_chars = chars_before - chars_after
        pct_removed = (removed_chars / chars_before * 100) if chars_before > 0 else 0.0

        total_before += chars_before
        total_after += chars_after

        print(f"\n  📄 {doc_name}")
        print(f"     Sections       : {len(raw_list)}")
        print(f"     Before chars   : {chars_before:,}")
        print(f"     After chars    : {chars_after:,}")
        print(f"     Removed chars  : {removed_chars:,} ({pct_removed:.2f}%)")

    total_removed = total_before - total_after
    overall_pct = (total_removed / total_before * 100) if total_before > 0 else 0.0

    print(f"\n{_separator('─')}")
    print(f"  TOTAL ACROSS ALL DOCUMENTS:")
    print(f"     Before chars   : {total_before:,}")
    print(f"     After chars    : {total_after:,}")
    print(f"     Total removed  : {total_removed:,} ({overall_pct:.2f}%)")
    print(_separator("─"))

    # ── 2. Visual Check: First Section Before & After ────────────────
    print(f"\n{_separator('═')}")
    print("  VISUAL CHECK: FIRST SECTION BEFORE vs. AFTER")
    print(_separator("═"))

    for doc_name in sorted(raw_by_doc.keys()):
        first_raw = raw_by_doc[doc_name][0]
        first_clean = cleaned_by_doc[doc_name][0]

        print(f"\n{'━' * 80}")
        print(f"  DOCUMENT: {doc_name}")
        print(f"  Page: {first_raw.get('page_number')} │ Section: §{first_raw.get('section_name')}")
        print(f"{'━' * 80}")

        raw_preview = first_raw.get("text_content", "")[:PREVIEW_CHARS]
        if len(first_raw.get("text_content", "")) > PREVIEW_CHARS:
            raw_preview += " …"

        clean_preview = first_clean.get("text_content", "")[:PREVIEW_CHARS]
        if len(first_clean.get("text_content", "")) > PREVIEW_CHARS:
            clean_preview += " …"

        print("\n  [BEFORE CLEANING]:")
        print(f"  {_separator('·', 74)}")
        for line in raw_preview.split("\n"):
            print(f"    {line}")

        print("\n  [AFTER CLEANING]:")
        print(f"  {_separator('·', 74)}")
        for line in clean_preview.split("\n"):
            print(f"    {line}")

    # ── 3. Save Cleaned Output ───────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(cleaned_data, fh, indent=2, ensure_ascii=False)

    print(f"\n{_separator('═')}")
    print(f"  ✅ Cleaned data successfully saved to: {OUTPUT_PATH.resolve()}")
    print(_separator("═"))
    print()


if __name__ == "__main__":
    run_clean_test()
