"""
QA Validation Script for parsed_output.json
Validates:
1. STRUCTURE
2. COVERAGE
3. MEDICAL ACCURACY
4. QUALITY
5. PAGE SANITY
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

RAW_DOCS_DIR = Path("raw_documents")
JSON_PATH = Path("parsed_output.json")
if not JSON_PATH.exists():
    JSON_PATH = Path("data/parsed_output.json")


def run_validation(json_path: Path) -> bool:
    print("=" * 80)
    print(f"  QA VALIDATION REPORT: {json_path}")
    print("=" * 80)

    if not json_path.exists():
        print(f"❌ FAIL: JSON file not found at {json_path}")
        return False

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections: list[dict[str, Any]] = data.get("sections", [])
    tables: list[dict[str, Any]] = data.get("tables", [])

    all_passed = True

    # -------------------------------------------------------------
    # Check 1: STRUCTURE
    # -------------------------------------------------------------
    structure_failures = []
    for idx, sec in enumerate(sections):
        doc_name = sec.get("document_name")
        sec_name = sec.get("section_name")
        text = sec.get("text_content")
        page_num = sec.get("page_number")

        errors = []
        if not isinstance(doc_name, str) or not doc_name.strip():
            errors.append("empty/invalid document_name")
        if not isinstance(sec_name, str) or not sec_name.strip():
            errors.append("empty/invalid section_name")
        if not isinstance(text, str) or not text.strip():
            errors.append("empty/invalid text_content")
        if not isinstance(page_num, int) or page_num < 1:
            errors.append(f"invalid page_number: {page_num}")

        if errors:
            structure_failures.append((idx, doc_name, page_num, errors))

    status_1 = "PASS" if not structure_failures else "FAIL"
    if structure_failures:
        all_passed = False
    print(f"\n[1] STRUCTURE: {status_1}")
    print(f"    Total sections analyzed: {len(sections)}")
    if structure_failures:
        print(f"    Failed entries count: {len(structure_failures)}")
        for idx, doc_name, page_num, errs in structure_failures[:10]:
            print(f"      - Section #{idx} (doc: {doc_name}, p.{page_num}): {', '.join(errs)}")
        if len(structure_failures) > 10:
            print(f"      ... and {len(structure_failures) - 10} more.")
    else:
        print("    All sections have non-empty document_name, section_name, text_content, and valid page_number >= 1.")

    # -------------------------------------------------------------
    # Check 2: COVERAGE
    # -------------------------------------------------------------
    coverage_failures = []
    pdf_files = list(RAW_DOCS_DIR.glob("*.pdf"))
    pdf_stems = {p.stem: p for p in pdf_files}
    found_stems = set(sec.get("document_name") for sec in sections)

    missing_pdfs = set(pdf_stems.keys()) - found_stems
    total_sections_count = len(sections)

    if missing_pdfs:
        coverage_failures.append(f"Missing PDFs in parsed output: {missing_pdfs}")
    if total_sections_count < 15:
        coverage_failures.append(f"Total sections ({total_sections_count}) < 15")

    per_pdf_counts = {}
    for stem in pdf_stems:
        count = sum(1 for sec in sections if sec.get("document_name") == stem)
        per_pdf_counts[stem] = count
        if count == 0:
            coverage_failures.append(f"PDF '{stem}' has 0 sections parsed.")

    status_2 = "PASS" if not coverage_failures else "FAIL"
    if coverage_failures:
        all_passed = False
    print(f"\n[2] COVERAGE: {status_2}")
    print(f"    PDFs in raw_documents: {len(pdf_files)}")
    print(f"    Total sections: {total_sections_count}")
    for stem, count in per_pdf_counts.items():
        print(f"      - {stem}: {count} sections")
    if coverage_failures:
        for fail in coverage_failures:
            print(f"      ❌ {fail}")

    # -------------------------------------------------------------
    # Check 3: MEDICAL ACCURACY
    # -------------------------------------------------------------
    med_failures = []
    
    # Check 3a: "Recommendation" section contains "recommends screening for depression" AND "pregnant and postpartum"
    # Also check if detected_grades contains "B" for that section
    recommendation_matches = []
    for sec in sections:
        sec_name = sec.get("section_name", "")
        text = sec.get("text_content", "")
        text_clean = " ".join(text.lower().split())
        
        has_rec_keyword = "recommend" in sec_name.lower() or "recommendation" in sec_name.lower()
        has_screening_dep = "recommends screening for depression" in text_clean or "recommend screening for depression" in text_clean or ("screening" in text_clean and "depression" in text_clean and "recommend" in text_clean)
        has_pregnant = "pregnant" in text_clean and "postpartum" in text_clean
        
        if ("recommend" in sec_name.lower() or "general" in sec_name.lower() or "what does the uspstf recommend" in sec_name.lower()) and has_pregnant and ("depression" in text_clean):
            recommendation_matches.append(sec)

    # Let's check strict recommendation criteria
    found_rec_match = False
    found_grade_b = False
    for sec in sections:
        text_clean = " ".join(sec.get("text_content", "").lower().split())
        grades = sec.get("detected_grades", [])
        
        contains_screening = "recommends screening for depression" in text_clean or "screening for depression" in text_clean
        contains_pregnant = "pregnant" in text_clean and "postpartum" in text_clean
        
        if contains_screening and contains_pregnant:
            found_rec_match = True
            if "B" in grades:
                found_grade_b = True

    # Also check if any section with "recommend" has Grade B or if global detected_grades has B
    global_grades = data.get("summary", {}).get("detected_grades", [])
    if not found_rec_match:
        med_failures.append("No section found containing both 'screening for depression' / 'recommends screening for depression' AND 'pregnant and postpartum'.")
    if not found_grade_b and "B" not in global_grades:
        med_failures.append("Grade 'B' not found in recommendation sections or summary.")

    # Check 3b: At least one entry or table mentions PHQ-2 or PHQ-9
    phq_found_in_sections = any(
        ("phq-2" in sec.get("text_content", "").lower() or "phq-9" in sec.get("text_content", "").lower())
        for sec in sections
    )
    phq_found_in_tables = any(
        any("phq" in tool.lower() for tool in tbl.get("matched_screening_tools", []))
        or "phq-2" in str(tbl).lower() or "phq-9" in str(tbl).lower()
        for tbl in tables
    )

    if not (phq_found_in_sections or phq_found_in_tables):
        med_failures.append("Neither PHQ-2 nor PHQ-9 was mentioned in sections or tables.")

    status_3 = "PASS" if not med_failures else "FAIL"
    if med_failures:
        all_passed = False
    print(f"\n[3] MEDICAL ACCURACY: {status_3}")
    print(f"    - Recommendation text check: {'FOUND' if found_rec_match else 'NOT FOUND'}")
    print(f"    - Grade 'B' detected: {'YES' if ('B' in global_grades or found_grade_b) else 'NO'}")
    print(f"    - PHQ-2 / PHQ-9 in sections: {'YES' if phq_found_in_sections else 'NO'}")
    print(f"    - PHQ-2 / PHQ-9 in tables: {'YES' if phq_found_in_tables else 'NO'}")
    if med_failures:
        for fail in med_failures:
            print(f"      ❌ {fail}")

    # -------------------------------------------------------------
    # Check 4: QUALITY
    # -------------------------------------------------------------
    quality_failures = []
    short_sections = []
    garbage_sections = []

    garbage_patterns = [
        re.compile(r"â€"),
        re.compile(r"ï¿½"),
        re.compile(r"[^\w\s\.,;:?!'\(\)\[\]\/\-\–\—\%\$\&\+\=\<\>\@\#\*\"°±²³µ]{3,}"),
    ]

    for idx, sec in enumerate(sections):
        text = sec.get("text_content", "")
        doc_name = sec.get("document_name")
        page_num = sec.get("page_number")
        sec_name = sec.get("section_name")

        if len(text.strip()) < 50:
            short_sections.append((idx, doc_name, page_num, sec_name, len(text.strip()), text.strip()))

        for pat in garbage_patterns:
            if pat.search(text):
                garbage_sections.append((idx, doc_name, page_num, sec_name, pat.pattern))
                break

    if short_sections:
        quality_failures.append(f"{len(short_sections)} section(s) shorter than 50 chars.")
    if garbage_sections:
        quality_failures.append(f"{len(garbage_sections)} section(s) with garbage/mojibake symbols.")

    status_4 = "PASS" if not quality_failures else "FAIL"
    if quality_failures:
        all_passed = False
    print(f"\n[4] QUALITY: {status_4}")
    print(f"    - Short sections (<50 chars): {len(short_sections)}")
    print(f"    - Garbage/encoding issues: {len(garbage_sections)}")
    if short_sections:
        print("    Sample short sections:")
        for idx, doc_name, page_num, sec_name, length, txt in short_sections[:5]:
            print(f"      - #{idx} [{doc_name} p.{page_num} §{sec_name}] ({length} chars): {repr(txt)}")
    if garbage_sections:
        print("    Sample garbage sections:")
        for idx, doc_name, page_num, sec_name, pat in garbage_sections[:5]:
            print(f"      - #{idx} [{doc_name} p.{page_num} §{sec_name}] matched pattern: {pat}")

    # -------------------------------------------------------------
    # Check 5: PAGE SANITY
    # -------------------------------------------------------------
    page_sanity_failures = []
    pdf_page_counts = {}

    if fitz:
        for stem, path in pdf_stems.items():
            try:
                doc = fitz.open(str(path))
                pdf_page_counts[stem] = len(doc)
                doc.close()
            except Exception as e:
                pdf_page_counts[stem] = None
                print(f"    ⚠️ Warning: Could not open {path}: {e}")

    for idx, sec in enumerate(sections):
        doc_name = sec.get("document_name")
        page_num = sec.get("page_number", 0)

        max_pages = pdf_page_counts.get(doc_name)
        if max_pages is not None:
            if page_num > max_pages:
                page_sanity_failures.append((idx, doc_name, page_num, max_pages))

    status_5 = "PASS" if not page_sanity_failures else "FAIL"
    if page_sanity_failures:
        all_passed = False
    print(f"\n[5] PAGE SANITY: {status_5}")
    for stem, count in pdf_page_counts.items():
        print(f"    - {stem}: {count} total pages")
    if page_sanity_failures:
        print(f"    Failed page entries count: {len(page_sanity_failures)}")
        for idx, doc_name, page_num, max_pages in page_sanity_failures[:5]:
            print(f"      - #{idx} [{doc_name}] page_number={page_num} > max_pages={max_pages}")
    else:
        print("    All page_numbers are <= actual PDF page count.")

    # -------------------------------------------------------------
    # FINAL OVERALL VERDICT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    final_verdict = "OVERALL: PASS" if all_passed else "OVERALL: FAIL / REVIEW NEEDED"
    print(f"  {final_verdict}")
    print("=" * 80 + "\n")
    return all_passed


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(JSON_PATH)
    run_validation(Path(target))
