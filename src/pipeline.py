"""
Pipeline — End-to-end RAG pipeline orchestrator.

Wires all layers together:
  Step 0: Safety gates (CRISIS / DOSING short-circuit)
  Step 1: RetrievalManager.retrieve (hybrid RRF + cross-encoder + boosts)
  Step 2: Confidence gate (< 0.76 → REFUSAL LOW_CONFIDENCE)
  Step 3: Prompt building (6-section schema + anti-hallucination rules)
  Step 4: LLM generation via Groq (or mock fallback)
  Step 5: Citation verification (verbatim quote check; FAILED → regenerate ONCE)
  Step 6: Append professional disclaimer

Usage:
    from src.pipeline import run_pipeline
    result = run_pipeline("Should pregnant women be screened for depression?")
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.settings import (
    CONFIDENCE_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    GENERATION_LOG_PATH,
)
from src.safety.guardrails import (
    check_input,
    verify_citations,
    check_response_schema,
    CRISIS_MESSAGE,
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
    t0 = time.perf_counter()
    manager = retrieval_manager or _get_retrieval_manager()
    client = llm_client or _get_llm_client()
    steps: list[dict[str, Any]] = []

    # ── Step 0: Safety Gates (CRISIS / DOSING) ────────────────────────
    safety_result = check_input(query)
    steps.append({
        "step": 0,
        "name": "safety_gate",
        "status": safety_result.status,
        "passed": safety_result.passed,
    })

    if not safety_result.passed:
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "query": query,
            "status": safety_result.status,
            "response": safety_result.message,
            "citations": None,
            "retrieval": None,
            "generation": None,
            "safety": {
                "status": safety_result.status,
                "reason": safety_result.reason,
                "flags": safety_result.flags,
            },
            "disclaimer": None,
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
        return {
            "query": query,
            "status": "REFUSAL_LOW_CONFIDENCE",
            "response": (
                f"The retrieval confidence ({top1_confidence:.1%}) is below the "
                f"calibrated threshold ({CONFIDENCE_THRESHOLD:.0%}). This query may be "
                f"outside the scope of USPSTF depression screening guidelines. "
                f"Please consult a clinical specialist."
            ),
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
    llm_response = generation_result.get("response", "")

    steps.append({
        "step": 4,
        "name": "generation",
        "provider": generation_result.get("provider"),
        "model": generation_result.get("model"),
        "status": generation_result.get("status"),
        "latency_ms": generation_result.get("latency_ms"),
    })

    # ── Step 5: Citation Verification ─────────────────────────────────
    citation_result = verify_citations(llm_response, context_chunks)

    if citation_result["status"] == "CITATION_VERIFICATION_FAILED":
        # Regenerate ONCE with stricter prompt
        stricter_prompt = (
            user_prompt + "\n\nCRITICAL: Your previous response contained unverified quotes. "
            "Ensure EVERY Quote: \"...\" uses an EXACT verbatim phrase from the context passages above. "
            "Do NOT paraphrase. Copy word-for-word from the context."
        )
        generation_result_2 = client.generate(
            prompt=stricter_prompt,
            context_chunks=context_chunks,
            system_prompt=system_prompt,
        )
        llm_response_2 = generation_result_2.get("response", "")
        citation_result_2 = verify_citations(llm_response_2, context_chunks)

        if citation_result_2["status"] == "OK" or citation_result_2["verified_quotes"] > citation_result["verified_quotes"]:
            # Use the improved response
            llm_response = llm_response_2
            citation_result = citation_result_2
            generation_result = generation_result_2
            steps.append({
                "step": "5_retry",
                "name": "citation_verification_retry",
                "status": citation_result["status"],
                "verified": citation_result["verified_quotes"],
                "total": citation_result["total_quotes"],
            })
        else:
            steps.append({
                "step": "5_retry",
                "name": "citation_verification_retry",
                "status": "STILL_FAILED",
                "unverified": citation_result["unverified_quotes"],
            })

    steps.append({
        "step": 5,
        "name": "citation_verification",
        "status": citation_result["status"],
        "verified_quotes": citation_result["verified_quotes"],
        "total_quotes": citation_result["total_quotes"],
        "unverified_quotes": citation_result.get("unverified_quotes", []),
    })

    # ── Step 6: Append Professional Disclaimer ────────────────────────
    full_response = llm_response.strip() + "\n\n---\n\n" + PROFESSIONAL_DISCLAIMER

    # Schema check
    schema_result = check_response_schema(llm_response)

    steps.append({
        "step": 6,
        "name": "disclaimer_appended",
        "schema_sections_present": schema_result["section_count"],
        "schema_all_present": schema_result["all_present"],
    })

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "query": query,
        "status": "SUCCESS",
        "response": full_response,
        "llm_response_raw": llm_response,
        "citations": citation_result,
        "schema": schema_result,
        "retrieval": retrieval_result,
        "generation": {
            "provider": generation_result.get("provider"),
            "model": generation_result.get("model"),
            "status": generation_result.get("status"),
            "latency_ms": generation_result.get("latency_ms"),
        },
        "safety": {"status": "OK"},
        "disclaimer": PROFESSIONAL_DISCLAIMER,
        "steps": steps,
        "total_time_ms": total_ms,
    }
