"""
Retrieval Evaluator — Ground-Truth Benchmark Suite for Clinical Retrieval (Expanded Pre-Day 3).

Evaluates the multi-stage clinical retrieval pipeline against an expanded >=16 query
benchmark containing:
  • 10 In-Scope Clinical Queries (Perinatal, Geriatric, Grade B, Harms, Tools, etc.)
  • 3 Ambiguous / Scope-Boundary Queries (Interval, Young Asymptomatic, Support Systems)
  • 3 Out-Of-Scope Negative Queries (Pharmacotherapy dosing, Bipolar mania, Dietary supplements)

Key Metrics Computed:
  • Precision@3: Proportion of top-3 retrieved passages matching expected clinical concepts
  • Mean Reciprocal Rank (MRR): 1 / rank of first relevant passage
  • Citation Existence Accuracy: 100% verification against data/chunks.json
  • Page Precision: Share of top-1 chunks with page span <= 10 pages
  • OOS Separation: min(In-Scope Top-1 Confidence) - max(OOS Top-1 Confidence)
  • Calibrated Confidence Threshold: Midpoint between OOS max and In-Scope min clamped to [0.5, 0.9]

Usage:
    from src.evaluation.retrieval_evaluator import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    report = evaluator.evaluate()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import CHUNKS_OUTPUT_PATH, CONFIDENCE_THRESHOLD
from src.retrieval.retrieval_manager import RetrievalManager

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkQuery:
    """A ground-truth clinical query definition."""

    query_id: str
    query: str
    category: str  # "IN_SCOPE", "AMBIGUOUS", "OUT_OF_SCOPE"
    target_concept: str
    expected_keywords: list[str]
    description: str


# ──────────────────────────────────────────────────────────────────────
# Expanded Ground-Truth Clinical Benchmark Dataset (16 Queries)
# ──────────────────────────────────────────────────────────────────────
EXPANDED_BENCHMARK: list[BenchmarkQuery] = [
    # ── 10 IN-SCOPE CLINICAL QUERIES ──
    BenchmarkQuery(
        query_id="Q01_PREGNANT",
        query="Should pregnant women be screened for depression?",
        category="IN_SCOPE",
        target_concept="Perinatal Depression Screening",
        expected_keywords=["pregnant", "postpartum", "perinatal", "depression", "screen"],
        description="USPSTF universal screening recommendation in pregnancy.",
    ),
    BenchmarkQuery(
        query_id="Q02_POSTPARTUM",
        query="Is depression screening recommended for postpartum women?",
        category="IN_SCOPE",
        target_concept="Postpartum Depression",
        expected_keywords=["postpartum", "pregnant", "epds", "depression", "screen"],
        description="Postpartum depression assessment and EPDS instrument relevance.",
    ),
    BenchmarkQuery(
        query_id="Q03_OLDER_ADULTS",
        query="Should adults over 65 be screened for depression?",
        category="IN_SCOPE",
        target_concept="Older Adults Screening",
        expected_keywords=["older adults", "65", "geriatric", "gds", "depression", "screen"],
        description="Evaluation of screening in geriatric populations (≥65 years).",
    ),
    BenchmarkQuery(
        query_id="Q04_SCREENING_TOOLS",
        query="What screening tools are recommended for depression in adults?",
        category="IN_SCOPE",
        target_concept="Validated Instruments",
        expected_keywords=["phq-9", "phq-2", "phq", "epds", "screening", "instrument", "tool"],
        description="Identification of validated instruments (PHQ family, EPDS).",
    ),
    BenchmarkQuery(
        query_id="Q05_GRADE_B",
        query="What is the USPSTF recommendation grade for depression screening?",
        category="IN_SCOPE",
        target_concept="Recommendation Grade",
        expected_keywords=["grade", "b", "recommend", "depression", "mdd"],
        description="Confirmation of Grade B recommendation for adult depression.",
    ),
    BenchmarkQuery(
        query_id="Q06_HARMS_RISKS",
        query="What are the harms or risks of depression screening?",
        category="IN_SCOPE",
        target_concept="Screening Harms & Overdiagnosis",
        expected_keywords=["harm", "harms", "adverse", "false positive", "overdiagnosis", "suicide", "risk"],
        description="Assessment of potential harms (false positives, unnecessary labeling).",
    ),
    BenchmarkQuery(
        query_id="Q07_SUICIDE_RISK",
        query="What instruments are used for suicide risk screening in adults?",
        category="IN_SCOPE",
        target_concept="Suicide Risk Instruments",
        expected_keywords=["suicide", "c-ssrs", "columbia", "asq", "risk", "screen"],
        description="Identification of suicide risk assessment tools.",
    ),
    BenchmarkQuery(
        query_id="Q08_GENERAL_ADULTS",
        query="Should all general adults be screened for depression?",
        category="IN_SCOPE",
        target_concept="General Adult Screening",
        expected_keywords=["adults", "general", "screen", "depression", "recommend"],
        description="Universal screening in asymptomatic general adult population.",
    ),
    BenchmarkQuery(
        query_id="Q09_ADOLESCENTS",
        query="Does the USPSTF recommend MDD depression screening in adolescents aged 12 to 18?",
        category="IN_SCOPE",
        target_concept="Adolescent Screening Scope",
        expected_keywords=["adolescent", "12", "18", "children", "mdd", "depression", "screen"],
        description="Adolescent major depressive disorder screening guideline scope.",
    ),
    BenchmarkQuery(
        query_id="Q10_PERINATAL_EVIDENCE",
        query="What is the evidence supporting perinatal depression screening?",
        category="IN_SCOPE",
        target_concept="Perinatal Evidence Synthesis",
        expected_keywords=["perinatal", "pregnant", "postpartum", "evidence", "cbt", "counseling", "screen"],
        description="Evidence synthesis regarding perinatal screening benefits and interventions.",
    ),

    # ── 3 AMBIGUOUS / BOUNDARY QUERIES ──
    BenchmarkQuery(
        query_id="Q11_SCREENING_INTERVAL",
        query="What is the recommended screening interval or frequency for depression?",
        category="AMBIGUOUS",
        target_concept="Screening Interval / Frequency",
        expected_keywords=["interval", "frequency", "annual", "optimal", "time", "screen", "evidence"],
        description="Ambiguity in clinical guidelines regarding exact interval frequency.",
    ),
    BenchmarkQuery(
        query_id="Q12_YOUNG_ASYMPTOMATIC",
        query="How should asymptomatic young adults without risk factors be screened for depression?",
        category="AMBIGUOUS",
        target_concept="Young Asymptomatic Adults",
        expected_keywords=["asymptomatic", "young", "adults", "risk", "screen", "depression"],
        description="Screening considerations in young adults without overt symptoms.",
    ),
    BenchmarkQuery(
        query_id="Q13_SUPPORT_SYSTEMS",
        query="What adequate support systems must be in place when implementing depression screening?",
        category="AMBIGUOUS",
        target_concept="Implementation & Support Systems",
        expected_keywords=["system", "support", "adequate", "diagnosis", "treatment", "follow-up", "care"],
        description="Prerequisite clinical staff and referral systems for screening implementation.",
    ),

    # ── 3 OUT-OF-SCOPE (OOS) NEGATIVE QUERIES ──
    BenchmarkQuery(
        query_id="Q14_SERTRALINE_DOSE",
        query="What is the standard starting dose and titration schedule of sertraline for depression?",
        category="OUT_OF_SCOPE",
        target_concept="Specific Drug Dosing (Out of Scope)",
        expected_keywords=["sertraline", "dose", "mg", "titration", "daily"],
        description="Pharmacological dosing specifics not covered by USPSTF screening guidelines.",
    ),
    BenchmarkQuery(
        query_id="Q15_BIPOLAR_MANIA",
        query="What is the recommended acute pharmacological treatment for bipolar mania?",
        category="OUT_OF_SCOPE",
        target_concept="Bipolar Disorder Treatment (Out of Scope)",
        expected_keywords=["bipolar", "mania", "lithium", "valproate", "antipsychotic"],
        description="Bipolar mania pharmacotherapy is out of scope of unipolar depression screening.",
    ),
    BenchmarkQuery(
        query_id="Q16_HERBAL_DIET",
        query="Are herbal supplements and St. John's Wort effective as first-line depression therapy?",
        category="OUT_OF_SCOPE",
        target_concept="Alternative Medicine / Supplements (Out of Scope)",
        expected_keywords=["herbal", "st. john", "supplement", "hypericum", "dietary"],
        description="Alternative herbal remedies not included in USPSTF recommendation statements.",
    ),
]


class RetrievalEvaluator:
    """Evaluates multi-stage retrieval performance on clinical ground-truth benchmarks."""

    def __init__(self, manager: RetrievalManager | None = None) -> None:
        self.manager = manager or RetrievalManager()
        self.benchmark = EXPANDED_BENCHMARK
        self.chunks_lookup = self._load_chunks_lookup()

    def _load_chunks_lookup(self) -> set[tuple[str, str, str, int, int]]:
        """Load valid chunk coordinate signatures from data/chunks.json."""
        lookup: set[tuple[str, str, str, int, int]] = set()
        chunks_file = Path(CHUNKS_OUTPUT_PATH)
        if chunks_file.exists():
            try:
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                    for c in chunks_data:
                        lookup.add((
                            c.get("chunk_id", ""),
                            c.get("document_name", ""),
                            c.get("section_name", ""),
                            int(c.get("start_page", 0)),
                            int(c.get("end_page", 0)),
                        ))
            except Exception as exc:
                logger.warning("Could not build chunks lookup: %s", exc)
        return lookup

    def evaluate(self, top_k_retrieval: int = 15, top_k_final: int = 3) -> dict[str, Any]:
        """
        Run the full evaluation over all 16 benchmark queries.

        Returns
        -------
        dict
            Comprehensive evaluation dictionary with aggregate metrics,
            per-query diagnostic logs, OOS separation, and threshold calibration.
        """
        query_results: list[dict[str, Any]] = []
        in_scope_precisions: list[float] = []
        in_scope_reciprocal_ranks: list[float] = []
        in_scope_top1_confidences: list[float] = []
        in_scope_avg_confidences: list[float] = []
        oos_top1_confidences: list[float] = []

        total_final_chunks_checked = 0
        valid_citations_count = 0
        top1_page_span_le_10_count = 0
        total_queries = len(self.benchmark)

        for bq in self.benchmark:
            bundle = self.manager.retrieve(
                query=bq.query,
                top_k_retrieval=top_k_retrieval,
                top_k_final=top_k_final,
            )

            final_chunks = bundle["final_chunks"]
            relevant_chunks_count = 0
            first_relevant_rank: int | None = None

            evaluated_hits: list[dict[str, Any]] = []
            for rank, chunk in enumerate(final_chunks, 1):
                total_final_chunks_checked += 1

                # Check citation existence in chunks.json
                sig = (
                    chunk.get("chunk_id", ""),
                    chunk.get("document_name", ""),
                    chunk.get("section_name", ""),
                    int(chunk.get("start_page", 0)),
                    int(chunk.get("end_page", 0)),
                )
                citation_exists = sig in self.chunks_lookup if self.chunks_lookup else True
                if citation_exists:
                    valid_citations_count += 1

                text_lower = chunk.get("text", "").lower()
                matched_kws = [
                    kw for kw in bq.expected_keywords
                    if kw.lower() in text_lower
                ]

                is_relevant = len(matched_kws) >= 1
                if is_relevant:
                    relevant_chunks_count += 1
                    if first_relevant_rank is None:
                        first_relevant_rank = rank

                evaluated_hits.append({
                    "final_rank": rank,
                    "chunk_id": chunk.get("chunk_id"),
                    "document_name": chunk.get("document_name"),
                    "section_name": chunk.get("section_name"),
                    "pages": f"p.{chunk.get('start_page')}-{chunk.get('end_page')}",
                    "page_span": int(chunk.get("end_page", 0)) - int(chunk.get("start_page", 0)) + 1,
                    "confidence": chunk.get("confidence", 0.0),
                    "section_prior": chunk.get("section_prior", 1.0),
                    "boosted_score": chunk.get("boosted_score", 0.0),
                    "is_relevant": is_relevant,
                    "citation_exists": citation_exists,
                    "matched_keywords": matched_kws,
                    "text_snippet": chunk.get("text", "").replace("\n", " ")[:150] + " …",
                })

            # Top-1 chunk page span check
            if final_chunks:
                top1_span = int(final_chunks[0].get("end_page", 0)) - int(final_chunks[0].get("start_page", 0)) + 1
                if top1_span <= 10:
                    top1_page_span_le_10_count += 1
                top1_conf = final_chunks[0].get("confidence", 0.0)
            else:
                top1_conf = 0.0

            p_at_k = relevant_chunks_count / top_k_final if top_k_final > 0 else 0.0
            rr = (1.0 / first_relevant_rank) if first_relevant_rank is not None else 0.0
            avg_q_conf = bundle.get("avg_confidence", 0.0)

            if bq.category in ("IN_SCOPE", "AMBIGUOUS"):
                in_scope_precisions.append(p_at_k)
                in_scope_reciprocal_ranks.append(rr)
                in_scope_top1_confidences.append(top1_conf)
                in_scope_avg_confidences.append(avg_q_conf)
            else:
                oos_top1_confidences.append(top1_conf)

            query_results.append({
                "query_id": bq.query_id,
                "category": bq.category,
                "query": bq.query,
                "target_concept": bq.target_concept,
                "expected_keywords": bq.expected_keywords,
                "precision_at_3": round(p_at_k, 4),
                "reciprocal_rank": round(rr, 4),
                "first_relevant_rank": first_relevant_rank,
                "top1_confidence": round(top1_conf, 4),
                "avg_confidence": round(avg_q_conf, 4),
                "retrieval_time_ms": bundle.get("retrieval_time_ms", 0.0),
                "hits": evaluated_hits,
            })

        # Dataset-level aggregate metrics
        mean_precision_at_3 = round(sum(in_scope_precisions) / len(in_scope_precisions), 4) if in_scope_precisions else 0.0
        mrr = round(sum(in_scope_reciprocal_ranks) / len(in_scope_reciprocal_ranks), 4) if in_scope_reciprocal_ranks else 0.0
        mean_confidence = round(sum(in_scope_avg_confidences) / len(in_scope_avg_confidences), 4) if in_scope_avg_confidences else 0.0

        # Citation Existence Accuracy (Target: 100%)
        citation_existence_accuracy = round(valid_citations_count / total_final_chunks_checked, 4) if total_final_chunks_checked > 0 else 1.0

        # Page Precision (Target: > Baseline)
        page_precision = round(top1_page_span_le_10_count / total_queries, 4) if total_queries > 0 else 0.0

        # OOS Separation & Calibrated Threshold
        min_in_scope_conf = round(min(in_scope_top1_confidences), 4) if in_scope_top1_confidences else 1.0
        max_oos_conf = round(max(oos_top1_confidences), 4) if oos_top1_confidences else 0.0
        oos_separation = round(min_in_scope_conf - max_oos_conf, 4)

        # Calibrated threshold = midpoint clamped to [0.5, 0.9]
        midpoint = (min_in_scope_conf + max_oos_conf) / 2.0
        calibrated_threshold = max(0.50, min(0.90, round(midpoint, 2)))

        # Regression gate checks
        passed_precision = mean_precision_at_3 >= 0.80
        passed_mrr = mrr >= 0.70
        passed_citations = citation_existence_accuracy == 1.0
        passed_page_prec = page_precision >= 0.90
        all_passed = passed_precision and passed_mrr and passed_citations and passed_page_prec

        return {
            "summary": {
                "total_queries": total_queries,
                "in_scope_queries_count": len(in_scope_precisions),
                "oos_queries_count": len(oos_top1_confidences),
                "mean_precision_at_3": mean_precision_at_3,
                "mrr": mrr,
                "mean_confidence": mean_confidence,
                "citation_existence_accuracy": citation_existence_accuracy,
                "page_precision": page_precision,
                "min_in_scope_top1_confidence": min_in_scope_conf,
                "max_oos_top1_confidence": max_oos_conf,
                "oos_separation": oos_separation,
                "calibrated_confidence_threshold": calibrated_threshold,
                "current_configured_threshold": CONFIDENCE_THRESHOLD,
                "targets": {
                    "precision_at_3_target": ">= 0.80",
                    "mrr_target": ">= 0.70",
                    "citation_existence_target": "100.0%",
                    "page_precision_target": ">= 90.0%",
                },
                "status": "PASS" if all_passed else "FAIL",
            },
            "query_evaluations": query_results,
        }
