"""
Pipeline — End-to-end RAG pipeline orchestrator.

Wires all layers together:
  Step 0: Self-healing data check (ensure_data_ready)
  Step 1: Safety gates (CRISIS / DOSING short-circuit)
  Step 2: RetrievalManager.retrieve (hybrid RRF + cross-encoder + boosts)
  Step 3: Confidence gate (< 0.76 → REFUSAL LOW_CONFIDENCE)
  Step 4: Prompt building (6-section schema + anti-hallucination rules)
  Step 5: LLM generation via OpenRouter (or mock fallback)
  Step 6: Citation verification (verbatim quote check; FAILED → regenerate ONCE)
  Step 7: Append professional disclaimer

Self-healing: On first run in a fresh clone, if data/chunks.json or
data/vector_db are missing, the pipeline automatically builds them by
running the full ingestion flow (prefer parsed_output.json to skip PDF
re-parsing; fall back to PDF parsing one-by-one with gc.collect()).

Usage:
    from src.pipeline import run_pipeline, ensure_data_ready
    ensure_data_ready()  # optional explicit call
    result = run_pipeline("Should pregnant women be screened for depression?")
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.settings import (
    CONFIDENCE_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    GENERATION_LOG_PATH,
    DATA_DIR,
    PROJECT_ROOT,
    CHUNKS_OUTPUT_PATH,
    VECTOR_DB_PATH,
    CLEANED_OUTPUT_PATH,
    RAW_DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    COLLECTION_NAME,
)
from src.safety.guardrails import (
    check_input,
    verify_citations,
    check_response_schema,
    check_faithfulness,
    CRISIS_MESSAGE,
    CRISIS_RESOURCE_LINE,
    DOSING_REFUSAL_MESSAGE,
    PROFESSIONAL_DISCLAIMER,
)
from src.retrieval.retrieval_manager import RetrievalManager
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Singleton instances (lazy init)
_retrieval_manager: RetrievalManager | None = None
_llm_client: LLMClient | None = None
_data_ready: bool = False


# ──────────────────────────────────────────────────────────────────────
# Self-Healing Data Readiness Check
# ──────────────────────────────────────────────────────────────────────
def _vector_db_populated() -> bool:
    """Check if ChromaDB vector_db directory exists and has content."""
    vdb = Path(VECTOR_DB_PATH)
    if not vdb.exists():
        return False
    for f in vdb.iterdir():
        if f.is_file() and f.stat().st_size > 0:
            return True
    return False


def ensure_data_ready() -> None:
    """
    Ensure data/chunks.json and data/vector_db exist before running the pipeline.

    Self-healing flow (for fresh clones):
      1. If data/chunks.json + vector_db both exist → return immediately.
      2. If data/cleaned_output.json exists → skip PDF parse, run chunk+embed.
      3. If parsed_output.json exists (repo root or data/) → clean it first.
      4. Otherwise → parse PDFs one-by-one with gc.collect() between files,
         then clean, chunk, embed, and upsert into ChromaDB.

    Prints a clear rich-style message so the user knows what's happening.
    """
    global _data_ready
    if _data_ready:
        return

    chunks_exist = CHUNKS_OUTPUT_PATH.exists() and CHUNKS_OUTPUT_PATH.stat().st_size > 100
    vdb_populated = _vector_db_populated()

    if chunks_exist and vdb_populated:
        _data_ready = True
        return  # Data already ready — nothing to do

    # ── Need to build data ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Building search index (first run)...")
    print("  This may take 2-5 minutes on first launch.")
    print("=" * 72 + "\n")

    # Ensure .env exists (copy from .env.example if missing)
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        example_path = PROJECT_ROOT / ".env.example"
        if example_path.exists():
            import shutil
            shutil.copy2(example_path, env_path)
            print("  Created .env from .env.example")
            print("  Please add your OPENROUTER_API_KEY to .env for LLM features.\n")

    # Check what data sources are available
    cleaned_exists = CLEANED_OUTPUT_PATH.exists() and CLEANED_OUTPUT_PATH.stat().st_size > 100
    parsed_at_root = (PROJECT_ROOT / "parsed_output.json").exists()
    parsed_in_data = (DATA_DIR / "parsed_output.json").exists()

    try:
        if cleaned_exists:
            # Skip PDF parsing entirely — just re-chunk + embed
            print("  Found data/cleaned_output.json — re-chunking & embedding...")
            _run_chunk_and_embed()
        elif parsed_at_root or parsed_in_data:
            # Run text cleaner on existing parsed output, then chunk + embed
            parsed_src = PROJECT_ROOT / "parsed_output.json" if parsed_at_root else DATA_DIR / "parsed_output.json"
            print(f"  Found {parsed_src.name} — cleaning, chunking & embedding...")
            _run_clean_parse(parsed_src)
            _run_chunk_and_embed()
        else:
            # Full pipeline: parse PDFs one-by-one → clean → chunk → embed
            print("  Parsing PDFs one-by-one (memory-safe)...")
            _run_full_ingestion()
    except Exception as e:
        logger.error("Self-healing ingestion failed: %s", e)
        print(f"\n  Auto-ingestion failed: {e}")
        print("  Please run manually:  python ingest.py\n")
        raise

    _data_ready = True
    print("\n" + "=" * 72)
    print("  Search index built successfully!")
    print("=" * 72 + "\n")


def _run_clean_parse(parsed_src: Path) -> None:
    """Run text cleaner on parsed_output.json to produce cleaned_output.json."""
    from src.ingestion.text_cleaner import clean_parsed_file
    clean_parsed_file(str(parsed_src), str(CLEANED_OUTPUT_PATH))
    gc.collect()


def _run_chunk_and_embed() -> None:
    """Chunk cleaned_output.json → embed → upsert into ChromaDB → save chunks.json."""
    from src.ingestion.chunker import load_cleaned_data, chunk_cleaned_data, save_chunks
    from src.ingestion.embedder import Embedder
    from src.retrieval.vector_store import VectorStore

    data = load_cleaned_data()
    chunks = chunk_cleaned_data(data, include_tables=True)

    embedder = Embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.embed_texts(texts, show_progress=True)

    store = VectorStore()
    store.reset()
    chunk_dicts = [c.to_dict() for c in chunks]
    store.add_chunks(chunk_dicts, embeddings)

    save_chunks(chunks)
    gc.collect()


def _run_full_ingestion() -> None:
    """
    Full ingestion: parse PDFs one-by-one → clean → chunk → embed → upsert.
    Uses gc.collect() between PDFs to avoid OOM on low-memory systems.
    """
    from src.ingestion.pdf_parser import parse_pdf
    from src.ingestion.text_cleaner import clean_parsed_file

    pdf_dir = RAW_DOCUMENTS_DIR
    if not pdf_dir.exists() or not list(pdf_dir.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. "
            "Please add USPSTF PDF documents or provide parsed_output.json."
        )

    # Parse each PDF individually to control memory
    all_sections: list[dict] = []
    all_tables: list[dict] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"    Parsing {pdf_path.name}...")
        result = parse_pdf(str(pdf_path))
        all_sections.extend(result.get("sections", []))
        all_tables.extend(result.get("tables", []))
        gc.collect()

    # Save combined parsed output
    parsed_data = {"sections": all_sections, "tables": all_tables}
    parsed_path = DATA_DIR / "parsed_output.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=2, ensure_ascii=False)

    # Clean → chunk → embed
    clean_parsed_file(str(parsed_path), str(CLEANED_OUTPUT_PATH))
    gc.collect()
    _run_chunk_and_embed()


def _get_retrieval_manager() -> RetrievalManager:
    global _retrieval_manager
    if _retrieval_manager is None:
        _retrieval_manager = RetrievalManager()
    return _retrieval_manager


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def clean_llm_response(text: str) -> str:
    """Strip think blocks, prompt echoes, and normalize markdown headers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for marker in [
        "### Response Instructions",
        "### Strict Grounding",
        "### Strict Citation Rules",
        "### Strict Grounding & Citation Rules",
        "CRITICAL FIXES REQUIRED",
    ]:
        if marker in text:
            text = text.split(marker)[0].strip()
    if "## Recommendation" in text:
        text = "## Recommendation" + text.split("## Recommendation", 1)[1]
    elif "## recommendation" in text:
        text = "## Recommendation" + text.split("## recommendation", 1)[1]
    return text.strip()


def run_pipeline(
    query: str,
    retrieval_manager: RetrievalManager | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """
    Execute the full RAG pipeline end-to-end.

    Parameters
    ----------
    query : str
        The clinical question.
    retrieval_manager : RetrievalManager, optional
        Override the default retrieval manager.
    llm_client : LLMClient, optional
        Override the default LLM client.

    Returns
    -------
    dict
        Full structured result with keys:
        'query', 'status', 'response', 'citations', 'retrieval', 'generation',
        'safety', 'disclaimer', 'steps', 'total_time_ms'.
    """
    # ── Step 0: Self-Healing Data Check ────────────────────────────
    ensure_data_ready()

    t0 = time.perf_counter()
    manager = retrieval_manager or _get_retrieval_manager()
    client = llm_client or _get_llm_client()
    steps: list[dict[str, Any]] = []

    # ── Step 1: Safety Gates (CRISIS / DOSING) ────────────────────────
    safety_result = check_input(query)
    steps.append({
        "step": 0,
        "name": "safety_gate",
        "status": safety_result.status,
        "passed": safety_result.passed,
    })

    if not safety_result.passed:
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        # CRISIS refusal vs DOSING refusal
        refusal_response = safety_result.message
        disclaimer_to_append = PROFESSIONAL_DISCLAIMER
        if disclaimer_to_append:
            refusal_response = refusal_response + "\n\n---\n\n" + disclaimer_to_append

        return {
            "query": query,
            "status": safety_result.status,
            "response": refusal_response,
            "citations": None,
            "retrieval": None,
            "generation": None,
            "safety": {
                "status": safety_result.status,
                "reason": safety_result.reason,
                "flags": safety_result.flags,
            },
            "disclaimer": disclaimer_to_append,
            "steps": steps,
            "total_time_ms": total_ms,
        }

    # ── Step 1: Retrieval ─────────────────────────────────────────────
    retrieval_result = manager.retrieve(query)
    final_chunks = retrieval_result.get("final_chunks", [])
    top1_confidence = retrieval_result.get("top1_confidence", 0.0)
    is_in_scope = retrieval_result.get("is_in_scope", False)

    steps.append({
        "step": 1,
        "name": "retrieval",
        "chunks_retrieved": len(final_chunks),
        "top1_confidence": top1_confidence,
        "is_in_scope": is_in_scope,
        "retrieval_time_ms": retrieval_result.get("retrieval_time_ms", 0),
    })

    # ── Step 2: Confidence Gate ───────────────────────────────────────
    if top1_confidence < CONFIDENCE_THRESHOLD:
        steps.append({
            "step": 2,
            "name": "confidence_gate",
            "status": "REFUSAL_LOW_CONFIDENCE",
            "top1_confidence": top1_confidence,
            "threshold": CONFIDENCE_THRESHOLD,
        })
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        low_conf_response = (
            f"The retrieval confidence ({top1_confidence:.1%}) is below the "
            f"calibrated threshold ({CONFIDENCE_THRESHOLD:.0%}). This query may be "
            f"outside the scope of USPSTF depression screening guidelines. "
            f"Please consult a clinical specialist."
            f"\n\n---\n\n{PROFESSIONAL_DISCLAIMER}"
        )
        return {
            "query": query,
            "status": "REFUSAL_LOW_CONFIDENCE",
            "response": low_conf_response,
            "citations": None,
            "retrieval": retrieval_result,
            "generation": None,
            "safety": {"status": "OK"},
            "disclaimer": PROFESSIONAL_DISCLAIMER,
            "steps": steps,
            "total_time_ms": total_ms,
        }

    steps.append({
        "step": 2,
        "name": "confidence_gate",
        "status": "PASSED",
        "top1_confidence": top1_confidence,
        "threshold": CONFIDENCE_THRESHOLD,
    })

    # ── Step 3: Prompt Building ───────────────────────────────────────
    context_chunks = final_chunks[:MAX_CONTEXT_CHUNKS]
    diversity_warning = retrieval_result.get("diversity_warning", False)
    system_prompt, user_prompt = build_prompt(
        query=query,
        context_chunks=context_chunks,
        diversity_warning=diversity_warning,
    )

    steps.append({
        "step": 3,
        "name": "prompt_building",
        "context_chunks_used": len(context_chunks),
        "diversity_warning": diversity_warning,
    })

    # ── Step 4: LLM Generation ────────────────────────────────────────
    generation_result = client.generate(
        prompt=user_prompt,
        context_chunks=context_chunks,
        system_prompt=system_prompt,
    )
    llm_response = clean_llm_response(generation_result.get("response", ""))

    steps.append({
        "step": 4,
        "name": "generation",
        "provider": generation_result.get("provider"),
        "model": generation_result.get("model"),
        "status": generation_result.get("status"),
        "latency_ms": generation_result.get("latency_ms"),
    })

    # ── Step 4: Verify Citations & Faithfulness ───────────────────────
    citation_result = verify_citations(llm_response, context_chunks)
    
    # Snap the response if quotes were fuzzy matched and fixed
    if "snapped_response" in citation_result and citation_result["snapped_response"] != llm_response:
        llm_response = citation_result["snapped_response"]
        
    faithfulness_result = check_faithfulness(query, llm_response, context_chunks)
    schema_result = check_response_schema(llm_response)

    # Fix #4: Retry if citation failed, faithfulness failed, or schema missing sections
    needs_retry = (
        generation_result.get("provider") != "mock"
        and (
            citation_result["status"] != "OK"
            or not faithfulness_result["passed"]
            or not schema_result["all_present"]
        )
    )

    if needs_retry:
        retry_notes: list[str] = []
        if "CITATION_RECYCLING" in faithfulness_result["flags"]:
            retry_notes.append("Each claim MUST have a UNIQUE, TOPICALLY-ALIGNED citation. Do NOT reuse or recycle the same quote for multiple sections.")
        if not schema_result["all_present"]:
            retry_notes.append("You MUST structure response with ALL 6 markdown sections: ## Recommendation, ## Population, ## Screening Tool, ## Harms & Considerations, ## Evidence, ## Source.")
        if citation_result["status"] != "OK":
            retry_notes.append("Ensure EVERY Quote: \"...\" uses an EXACT substantive verbatim phrase (>=25 alphanumeric chars, no table syntax, no PMIDs, no document titles) from the context.")
        if "CAVEAT_SUPPRESSED" in faithfulness_result["flags"]:
            retry_notes.append("You MUST state explicit caveats ('no evidence on frequency', 'uncertainty', 'only 1 study') and NOT claim adequate evidence.")
        if "ATTRIBUTION_MISSING" in faithfulness_result["flags"]:
            retry_notes.append("You MUST state: 'Note: This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance.'")
        if "SCOPE_UNACKNOWLEDGED" in faithfulness_result["flags"]:
            retry_notes.append("You MUST explicitly state: 'This guideline does not address adolescents or children; the following applies to adults only.'")
        if "UNGROUNDED_CLAIM" in faithfulness_result["flags"]:
            retry_notes.append("Every clinical claim MUST have an inline [Doc: ...] citation within 100 characters.")
        if "MISMATCHED_CITATION" in faithfulness_result["flags"]:
            retry_notes.append("Ensure citations in ## Population cite chunks regarding populations/ages, NOT screening intervals/frequency.")

        stricter_prompt = (
            user_prompt
            + "\n\nCRITICAL FIXES REQUIRED IN REGENERATION:\n"
            + "\n".join(f"- {n}" for n in retry_notes)
        )

        logger.info("Triggering pipeline regeneration retry with %d notes", len(retry_notes))
        generation_result_2 = client.generate(
            prompt=stricter_prompt,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
        )
        llm_response_2 = clean_llm_response(generation_result_2.get("response", ""))
        citation_result_2 = verify_citations(llm_response_2, context_chunks)
        faithfulness_result_2 = check_faithfulness(query, llm_response_2, context_chunks)
        schema_result_2 = check_response_schema(llm_response_2)

        if (
            schema_result_2["section_count"] >= schema_result["section_count"]
            and (
                citation_result_2["status"] == "OK"
                or citation_result_2["verified_quotes"] >= citation_result["verified_quotes"]
            )
        ):
            llm_response = llm_response_2
            citation_result = citation_result_2
            faithfulness_result = faithfulness_result_2
            schema_result = schema_result_2
            generation_result = generation_result_2
            steps.append({
                "step": "5_retry",
                "name": "regeneration_retry",
                "status": "IMPROVED",
                "citation_status": citation_result["status"],
                "faithfulness_flags": faithfulness_result["flags"],
                "schema_sections": schema_result["section_count"],
            })
        else:
            steps.append({
                "step": "5_retry",
                "name": "regeneration_retry",
                "status": "STILL_FAILED",
                "unverified": citation_result.get("unverified_quotes", []),
            })

    steps.append({
        "step": 5,
        "name": "citation_and_faithfulness_verification",
        "citation_status": citation_result["status"],
        "verified_quotes": citation_result["verified_quotes"],
        "total_quotes": citation_result["total_quotes"],
        "faithfulness_passed": faithfulness_result["passed"],
        "faithfulness_flags": faithfulness_result["flags"],
        "schema_sections": schema_result["section_count"],
    })

    # ── Step 6: Append Crisis Resource & Professional Disclaimer ──────
    # Fix 7: Deduplicate disclaimer and 988 touchpoint if already present in response
    normalized_text = llm_response.strip()
    appendices: list[str] = []
    if "988" not in normalized_text and "⚠️" not in normalized_text:
        appendices.append(CRISIS_RESOURCE_LINE)
    if "This information is based on USPSTF guidance" not in normalized_text and "not a substitute for professional" not in normalized_text:
        appendices.append(PROFESSIONAL_DISCLAIMER)

    if appendices:
        full_response = normalized_text + "\n\n---\n\n" + "\n\n".join(appendices)
    else:
        full_response = normalized_text

    steps.append({
        "step": 6,
        "name": "disclaimer_appended",
        "has_988_line": True,
        "schema_sections_present": schema_result["section_count"],
        "schema_all_present": schema_result["all_present"],
    })

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Fix #3: Determine distinct status codes
    if generation_result.get("provider") == "mock" or generation_result.get("status") == "error":
        pipeline_status = "REFUSAL_LLM_UNREACHABLE"
    elif schema_result["section_count"] == 0:
        pipeline_status = "REFUSAL_SCHEMA_FAILED"
    elif citation_result["status"] == "OK" and faithfulness_result["passed"] and (schema_result["all_present"] or schema_result["section_count"] >= 5):
        pipeline_status = "SUCCESS"
    elif schema_result["section_count"] >= 5:
        pipeline_status = "SUCCESS_WITH_WARNINGS"
    else:
        pipeline_status = "REFUSAL_QUALITY_FAILED"

    return {
        "query": query,
        "status": pipeline_status,
        "response": full_response,
        "llm_response_raw": llm_response,
        "citations": citation_result,
        "faithfulness": faithfulness_result,
        "schema": schema_result,
        "retrieval": retrieval_result,
        "generation": {
            "provider": generation_result.get("provider"),
            "model": generation_result.get("model"),
            "status": generation_result.get("status"),
            "latency_ms": generation_result.get("latency_ms"),
        },
        "safety": {"status": "OK", "has_988_line": True},
        "disclaimer": PROFESSIONAL_DISCLAIMER,
        "steps": steps,
        "total_time_ms": total_ms,
    }
