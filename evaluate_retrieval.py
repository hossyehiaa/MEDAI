"""
evaluate_retrieval.py — Expanded Retrieval Benchmark Runner (Pre-Day 3).

Executes the ground-truth clinical retrieval benchmark (16 queries), renders rich diagnostic
tables to the terminal, computes Citation Existence, Page Precision, and OOS Separation,
and serializes the evaluation metrics to ``data/retrieval_evaluation.json``.

Usage:
    python evaluate_retrieval.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.retrieval_evaluator import RetrievalEvaluator

# Suppress debug logs
logging.basicConfig(level=logging.WARNING)
console = Console()

EVAL_OUTPUT_PATH = Path("data/retrieval_evaluation.json")


def run_evaluation_suite() -> bool:
    start_time = time.time()

    console.print()
    console.print(
        Panel(
            Align.center(
                Text.from_markup(
                    "[bold cyan]medAI — EXPANDED CLINICAL RETRIEVAL BENCHMARK SUITE (Pre-Day 3)[/bold cyan]\n"
                    "[bold white]Multi-Stage Retrieval Evaluation: Hybrid RRF + Cross-Encoder + Section Priors + Diversity[/bold white]\n"
                    "[dim]Success Thresholds: Precision@3 ≥ 80% | MRR ≥ 0.70 | Citation Existence = 100% | Page Precision ≥ 90%[/dim]"
                )
            ),
            box=box.DOUBLE_EDGE,
            border_style="cyan",
        )
    )

    evaluator = RetrievalEvaluator()
    console.print("[dim]Evaluating 16 ground-truth clinical queries across 3 categories …[/dim]\n")

    report = evaluator.evaluate(top_k_retrieval=15, top_k_final=3)
    summary = report["summary"]
    queries = report["query_evaluations"]

    # ── Per-Query Diagnostic Table ─────────────────────────────────────
    query_table = Table(
        box=box.ROUNDED,
        border_style="bright_blue",
        title="[bold white]Per-Query Benchmark Results (16 Queries)[/bold white]",
        show_header=True,
    )
    query_table.add_column("Query ID", style="bold yellow", width=22)
    query_table.add_column("Category", justify="center", width=14)
    query_table.add_column("Clinical Scope / Concept", style="cyan", width=28)
    query_table.add_column("P@3", justify="center", style="bold green", width=8)
    query_table.add_column("MRR", justify="center", style="bold magenta", width=8)
    query_table.add_column("Top-1 Conf", justify="right", style="bold white", width=12)
    query_table.add_column("Latency", justify="right", style="dim white", width=10)

    for q in queries:
        cat = q["category"]
        if cat == "IN_SCOPE":
            cat_styled = "[green]IN-SCOPE[/green]"
        elif cat == "AMBIGUOUS":
            cat_styled = "[yellow]AMBIGUOUS[/yellow]"
        else:
            cat_styled = "[red]OUT-OF-SCOPE[/red]"

        p3 = f"{q['precision_at_3']:.1%}" if cat != "OUT_OF_SCOPE" else "[dim]N/A[/dim]"
        mrr_val = f"{q['reciprocal_rank']:.2f}" if cat != "OUT_OF_SCOPE" else "[dim]N/A[/dim]"
        conf = f"{q['top1_confidence']:.1%}"
        lat = f"{q['retrieval_time_ms']:.1f}ms"
        p_style = "green" if q["precision_at_3"] >= 0.66 else "yellow"

        query_table.add_row(
            q["query_id"],
            cat_styled,
            q["target_concept"],
            f"[{p_style}]{p3}[/{p_style}]" if cat != "OUT_OF_SCOPE" else p3,
            mrr_val,
            conf,
            lat,
        )

    console.print(query_table)
    console.print()

    # ── Aggregate Benchmark Scorecard ──────────────────────────────────
    mean_p3 = summary["mean_precision_at_3"]
    mrr = summary["mrr"]
    mean_conf = summary["mean_confidence"]
    citation_acc = summary["citation_existence_accuracy"]
    page_prec = summary["page_precision"]
    min_in_scope = summary["min_in_scope_top1_confidence"]
    max_oos = summary["max_oos_top1_confidence"]
    oos_sep = summary["oos_separation"]
    calibrated_thresh = summary["calibrated_confidence_threshold"]
    status = summary["status"]

    summary_table = Table(
        box=box.HEAVY_EDGE,
        border_style="green" if status == "PASS" else "red",
        title="[bold white]Day 2 Pre-Day 3 Retrieval Layer Aggregate Scorecard[/bold white]",
        show_header=True,
    )
    summary_table.add_column("Core Metric", style="white", justify="left")
    summary_table.add_column("Observed Score", style="bold cyan", justify="right")
    summary_table.add_column("Target Standard", style="dim white", justify="left")
    summary_table.add_column("Status", justify="center")

    summary_table.add_row(
        "Mean Precision@3 (In-Scope)",
        f"{mean_p3:.1%}",
        "≥ 80.0% relevant in Top-3",
        "[green]PASS[/green]" if mean_p3 >= 0.80 else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "Mean Reciprocal Rank (MRR)",
        f"{mrr:.4f}",
        "≥ 0.7000 (Top-1 priority)",
        "[green]PASS[/green]" if mrr >= 0.70 else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "Citation Existence Accuracy",
        f"{citation_acc:.1%}",
        "100.0% coordinates exist in chunks.json",
        "[green]PASS[/green]" if citation_acc == 1.0 else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "Page Precision (Span ≤ 10p)",
        f"{page_prec:.1%}",
        "≥ 90.0% of top-1 chunks",
        "[green]PASS[/green]" if page_prec >= 0.90 else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "Mean Top-3 Confidence",
        f"{mean_conf:.1%}",
        "≥ 70.0% calibrated certainty",
        "[green]PASS[/green]" if mean_conf >= 0.70 else "[red]FAIL[/red]",
    )
    summary_table.add_row(
        "OOS Confidence Separation",
        f"{oos_sep:+.1%} (In: {min_in_scope:.1%} vs OOS: {max_oos:.1%})",
        "Clear margin > 0%",
        "[green]PASS[/green]" if oos_sep > 0 else "[yellow]WARN[/yellow]",
    )
    summary_table.add_row(
        "Calibrated Confidence Threshold",
        f"{calibrated_thresh:.2f}",
        "Midpoint clamped to [0.50, 0.90]",
        "[cyan]CALIBRATED[/cyan]",
    )

    console.print(summary_table)

    # ── Save JSON Report ───────────────────────────────────────────────
    report["elapsed_seconds"] = round(time.time() - start_time, 2)
    EVAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    console.print(
        Panel(
            Align.center(
                Text(
                    f"DAY 2 RETRIEVAL BENCHMARK VERDICT: {'✅ PASSED ALL THRESHOLDS' if status == 'PASS' else '❌ FAILED'}",
                    style="bold green on black" if status == "PASS" else "bold red on black",
                )
            ),
            box=box.DOUBLE_EDGE,
            border_style="green" if status == "PASS" else "red",
        )
    )
    console.print(f"[dim]Evaluation completed in {report['elapsed_seconds']}s. Full metrics saved to [bold white]{EVAL_OUTPUT_PATH.resolve()}[/bold white].[/dim]\n")

    return status == "PASS"


if __name__ == "__main__":
    success = run_evaluation_suite()
    sys.exit(0 if success else 1)
