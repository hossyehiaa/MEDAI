"""
search_cli.py — Interactive Clinical Search & Retrieval Transparency CLI.

Demonstrates transparent, multi-stage retrieval BEFORE any LLM generation:
  • Step 1: Hybrid Semantic (ChromaDB) + Lexical (BM25) RRF Candidate Fusion
  • Step 2: Cross-Encoder (ms-marco-MiniLM-L-6-v2) Deep Re-ranking & Section Priors
  • Step 3: Top-3 High-Precision Passages Table with Confidence Metrics & Diversity

Usage:
    # Interactive CLI mode
    python search_cli.py

    # Single-query execution
    python search_cli.py "Should pregnant women be screened for depression?"

    # Automated test preset
    python search_cli.py /test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.settings import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_FINAL,
    RRF_K,
    CONFIDENCE_THRESHOLD,
)
from src.retrieval.retrieval_manager import RetrievalManager

console = Console()

PRESET_TEST_QUERIES: list[str] = [
    "Should pregnant women be screened for depression?",
    "What screening tools are recommended for depression in adults?",
    "What is the USPSTF recommendation grade for depression screening?",
    "What is the dose of sertraline for depression?",
]


def display_retrieval_bundle(result: dict[str, Any], query_idx: int | None = None) -> None:
    """Render the transparent multi-stage retrieval process in Rich format."""
    query = result["query"]
    final_chunks = result["final_chunks"]
    hybrid_candidates = result.get("hybrid_candidates", [])
    avg_conf = result["avg_confidence"]
    top1_conf = result.get("top1_confidence", 0.0)
    in_scope = result.get("is_in_scope", True)
    latency = result["retrieval_time_ms"]
    breakdown = result.get("latency_breakdown_ms", {})

    status_tag = "[bold green]IN-SCOPE[/bold green]" if in_scope else "[bold red]OUT-OF-SCOPE[/bold red]"
    header_title = f"Query: \"{query}\"" if query_idx is None else f"Query #{query_idx}: \"{query}\""

    console.print(
        Panel(
            f"[bold cyan]{header_title}[/bold cyan]  |  Status: {status_tag}\n"
            f"[dim]Total Latency: {latency:.1f}ms (Hybrid: {breakdown.get('hybrid', 0):.1f}ms | "
            f"Rerank: {breakdown.get('rerank', 0):.1f}ms) | Top-1 Conf: {top1_conf:.1%} | Avg Top-3: {avg_conf:.1%} (Threshold: {CONFIDENCE_THRESHOLD:.2f})[/dim]",
            box=box.ROUNDED,
            border_style="cyan" if in_scope else "red",
        )
    )

    # ── STEP 1: Hybrid Retrieval & Fusion Summary ──────────────────────
    step1_table = Table(
        box=box.SIMPLE_HEAD,
        border_style="bright_blue",
        title="[bold yellow]Stage 1: Hybrid RRF Retrieval (Top 15 Candidates Pool)[/bold yellow]",
        show_header=True,
    )
    step1_table.add_column("RRF Rank", justify="center", style="bold yellow", width=10)
    step1_table.add_column("RRF Score", justify="right", style="cyan", width=12)
    step1_table.add_column("Dense (ChromaDB)", justify="center", style="white", width=18)
    step1_table.add_column("Sparse (BM25)", justify="center", style="white", width=18)
    step1_table.add_column("Document & Section", style="dim white")

    for rank, cand in enumerate(hybrid_candidates[:5], 1):
        sem_rank = f"Rank #{cand['semantic_rank']}" if cand.get("semantic_rank") else "—"
        bm25_rank = f"Rank #{cand['bm25_rank']}" if cand.get("bm25_rank") else "—"
        doc = cand.get("document_name", "?")[:22]
        sec = cand.get("section_name", "?")[:16]
        pages = f"p.{cand.get('start_page', '?')}-{cand.get('end_page', '?')}"
        doc_str = f"{doc}.. | §{sec} ({pages})"
        step1_table.add_row(
            f"#{rank}",
            f"{cand.get('rrf_score', 0.0):.5f}",
            sem_rank,
            bm25_rank,
            doc_str,
        )

    if len(hybrid_candidates) > 5:
        step1_table.add_row("…", "…", "…", "…", f"[dim]… and {len(hybrid_candidates) - 5} more candidate passages evaluated[/dim]")

    console.print(step1_table)
    console.print()

    # ── STEP 2: Cross-Encoder Top-3 Final Precision Chunks ───────────
    step2_table = Table(
        box=box.HEAVY_EDGE,
        border_style="bright_green",
        title="[bold green]Stage 2: Cross-Encoder Reranked Passages (Retrieved BEFORE Generation)[/bold green]",
        show_header=True,
    )
    step2_table.add_column("#", justify="center", style="bold green", width=4)
    step2_table.add_column("Confidence", justify="right", style="bold cyan", width=12)
    step2_table.add_column("Prior", justify="right", style="dim white", width=8)
    step2_table.add_column("Boosted", justify="right", style="bold yellow", width=10)
    step2_table.add_column("Document & Section", style="white", width=32)
    step2_table.add_column("Clinical Passage Excerpt", style="white")

    for c in final_chunks:
        rank = c.get("final_rank", 1)
        conf = f"{c.get('confidence', 0.0):.1%}"
        prior = f"{c.get('section_prior', 1.0):.2f}x"
        boosted = f"{c.get('boosted_score', 0.0):.3f}"
        doc = c.get("document_name", "?")
        sec = c.get("section_name", "?")
        pages = f"p.{c.get('start_page', '?')}-{c.get('end_page', '?')}"
        table_tag = " [bold magenta][Table][/bold magenta]" if c.get("is_table") else ""
        doc_display = f"{doc[:18]}..\n§{sec[:14]}{table_tag} ({pages})"

        excerpt = c.get("text", "").replace("\n", " ").strip()
        if len(excerpt) > 220:
            excerpt = excerpt[:220] + " …"

        # Highlight screening tool tags if present
        tools = c.get("screening_tools", [])
        if tools:
            excerpt += f"\n[dim magenta]★ Screening tools: {', '.join(tools)}[/dim magenta]"

        step2_table.add_row(f"#{rank}", conf, prior, boosted, doc_display, excerpt)

    console.print(step2_table)
    console.print()


def run_test_suite(manager: RetrievalManager) -> None:
    """Run preset clinical test queries."""
    console.print(
        Panel(
            "[bold white]Executing Preset Clinical Test Suite (4 Queries: In-Scope + Out-Of-Scope)[/bold white]",
            box=box.ROUNDED,
            border_style="magenta",
        )
    )
    for idx, query in enumerate(PRESET_TEST_QUERIES, 1):
        result = manager.retrieve(query)
        display_retrieval_bundle(result, query_idx=idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="medAI Transparent Clinical Retrieval CLI")
    parser.add_argument("query", nargs="?", type=str, help="Clinical query string or '/test'")
    args = parser.parse_args()

    console.print(
        Panel(
            Align.center(
                Text.from_markup(
                    "[bold cyan]medAI — CLINICAL RETRIEVAL TRANSPARENCY CONSOLE (Day 2)[/bold cyan]\n"
                    f"[dim]Embedder: {EMBEDDING_MODEL} | Reranker: {RERANKER_MODEL} | Collection: {COLLECTION_NAME}[/dim]\n"
                    "[dim white]Multi-stage retrieval: ChromaDB (dense) + BM25 (sparse) → RRF (k=60) → Cross-Encoder → Prior Boost → Top 3 Diversity[/dim white]"
                )
            ),
            box=box.DOUBLE_EDGE,
            border_style="cyan",
        )
    )

    console.print("[dim]Initializing retrieval pipeline components …[/dim]")
    manager = RetrievalManager()
    console.print("[bold green]✔ Retrieval engine ready.[/bold green]\n")

    # Command line argument handling
    if args.query:
        if args.query.strip().lower() in ("/test", "--test"):
            run_test_suite(manager)
            return
        result = manager.retrieve(args.query.strip())
        display_retrieval_bundle(result)
        return

    # Interactive Loop
    console.print("[bold white]Commands:[/bold white] Type any clinical question, [cyan]/test[/cyan] for benchmark queries, or [cyan]/quit[/cyan] to exit.\n")
    while True:
        try:
            user_input = console.input("[bold cyan]medAI-Search > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting search console.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[bold green]Goodbye![/bold green]")
            break

        if user_input.lower() == "/test":
            run_test_suite(manager)
            continue

        result = manager.retrieve(user_input)
        display_retrieval_bundle(result)


if __name__ == "__main__":
    main()
