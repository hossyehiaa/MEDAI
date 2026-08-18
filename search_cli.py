"""
search_cli.py — Interactive Clinical RAG Pipeline Console (Day 3).

Full 6-step pipeline display:
  [0] Safety Gate (CRISIS/DOSING short-circuit)
  [1] Retrieval top-N table (doc/section/page/conf + boost tags + diversity warning)
  [2] Confidence Gate (0.76 threshold)
  [3] LLM Generation panel [GROQ/llama-3.3-70b-versatile] or [MOCK] + latency + 6-section response
  [4] Citation verification status
  [5] Professional disclaimer

Usage:
    python search_cli.py                                           # Interactive mode
    python search_cli.py "Should pregnant women be screened?"      # Single query
    python search_cli.py /test                                     # Automated test suite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
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
    LLM_MODEL,
    LLM_PROVIDER,
)
from src.pipeline import run_pipeline
from src.safety.guardrails import check_input, PROFESSIONAL_DISCLAIMER

console = Console()

PRESET_TEST_QUERIES: list[str] = [
    "Should pregnant women be screened for depression?",
    "What screening tools are recommended for depression in adults?",
    "What is the USPSTF recommendation grade for depression screening?",
    "Should adults over 65 be screened for depression?",
    "I want to kill myself",
    "What is the dose of sertraline for depression?",
    "quiero morir",
]


def display_safety_gate(query: str, result: Any, query_idx: int | None = None) -> None:
    """Render a safety gate interception (CRISIS or REFUSAL_OOS)."""
    header = f"Query: \"{query}\"" if query_idx is None else f"Query #{query_idx}: \"{query}\""
    status = result.status

    if status == "CRISIS":
        border = "bold red"
        status_tag = "[bold white on red] CRISIS GATE TRIGGERED [/bold white on red]"
    elif status == "REFUSAL_OOS":
        border = "yellow"
        status_tag = "[bold yellow]OUT-OF-SCOPE REFUSAL[/bold yellow]"
    else:
        border = "red"
        status_tag = f"[bold red]BLOCKED: {result.reason}[/bold red]"

    console.print(
        Panel(
            f"[bold cyan]{header}[/bold cyan]  |  {status_tag}\n\n"
            f"[bold white]{result.message}[/bold white]\n\n"
            f"[dim]Safety flags: {', '.join(result.flags)}[/dim]",
            box=box.HEAVY_EDGE,
            border_style=border,
        )
    )
    console.print()


def display_pipeline_result(result: dict[str, Any], query_idx: int | None = None) -> None:
    """Render the full 6-step pipeline result."""
    query = result["query"]
    status = result["status"]
    response = result.get("response", "")
    retrieval = result.get("retrieval")
    generation = result.get("generation")
    citations = result.get("citations")
    schema = result.get("schema")
    total_ms = result.get("total_time_ms", 0)

    header_title = f"Query: \"{query}\"" if query_idx is None else f"Query #{query_idx}: \"{query}\""

    # ── [0] Status & Scope ────────────────────────────────────────────
    if status == "CRISIS":
        console.print(Panel(
            f"[bold cyan]{header_title}[/bold cyan]  |  [bold white on red] CRISIS GATE [/bold white on red]\n\n"
            f"[bold white]{response}[/bold white]",
            box=box.HEAVY_EDGE, border_style="bold red",
        ))
        console.print()
        return

    if status == "REFUSAL_OOS":
        console.print(Panel(
            f"[bold cyan]{header_title}[/bold cyan]  |  [bold yellow]OUT-OF-SCOPE REFUSAL[/bold yellow]\n\n"
            f"[white]{response}[/white]",
            box=box.HEAVY_EDGE, border_style="yellow",
        ))
        console.print()
        return

    if status == "REFUSAL_LOW_CONFIDENCE":
        top1_conf = retrieval.get("top1_confidence", 0) if retrieval else 0
        console.print(Panel(
            f"[bold cyan]{header_title}[/bold cyan]  |  [bold red]LOW CONFIDENCE REFUSAL[/bold red]\n\n"
            f"[white]{response}[/white]\n\n"
            f"[dim]Top-1 Confidence: {top1_conf:.1%} | Threshold: {CONFIDENCE_THRESHOLD:.0%}[/dim]",
            box=box.HEAVY_EDGE, border_style="red",
        ))
        console.print()
        return

    # ── [1] Retrieval Summary ─────────────────────────────────────────
    if retrieval:
        final_chunks = retrieval.get("final_chunks", [])
        top1_conf = retrieval.get("top1_confidence", 0.0)
        is_perinatal = retrieval.get("is_perinatal_query", False)
        is_older_adults = retrieval.get("is_older_adults_query", False)
        diversity_warning = retrieval.get("diversity_warning", False)
        unique_docs = retrieval.get("unique_documents_count", 1)
        retrieval_ms = retrieval.get("retrieval_time_ms", 0)

        perinatal_tag = " [bold magenta][PERINATAL BOOST][/bold magenta]" if is_perinatal else ""
        older_tag = " [bold blue][OLDER ADULTS BOOST][/bold blue]" if is_older_adults else ""
        div_tag = " | [bold yellow][Low Diversity][/bold yellow]" if diversity_warning else f" | [dim]Docs: {unique_docs}[/dim]"

        console.print(Panel(
            f"[bold cyan]{header_title}[/bold cyan]  |  [bold green]IN-SCOPE[/bold green]{perinatal_tag}{older_tag}{div_tag}\n"
            f"[dim]Retrieval: {retrieval_ms:.0f}ms | Top-1 Conf: {top1_conf:.1%} | Threshold: {CONFIDENCE_THRESHOLD:.0%}[/dim]",
            box=box.ROUNDED, border_style="cyan",
        ))

        # Retrieval table
        ret_table = Table(box=box.SIMPLE_HEAD, border_style="bright_blue",
                          title="[bold yellow]Step 1: Retrieved Passages[/bold yellow]")
        ret_table.add_column("#", justify="center", width=4)
        ret_table.add_column("Confidence", justify="right", style="cyan", width=12)
        ret_table.add_column("Prior", justify="right", width=8)
        ret_table.add_column("Boosted", justify="right", style="yellow", width=10)
        ret_table.add_column("Document & Section", style="white")

        for c in final_chunks[:5]:
            rank = c.get("final_rank", "?")
            conf = f"{c.get('confidence', 0):.1%}"
            prior = f"{c.get('section_prior', 1.0):.2f}x"
            boosted = f"{c.get('boosted_score', 0):.3f}"
            doc = c.get("document_name", "?")[:20]
            sec = c.get("section_name", "?")[:16]
            pages = f"p.{c.get('start_page', '?')}-{c.get('end_page', '?')}"
            tags = ""
            if c.get("perinatal_boosted"):
                tags += " [magenta][Perinatal][/magenta]"
            if c.get("older_adults_boosted"):
                tags += " [blue][OlderAdults][/blue]"
            ret_table.add_row(f"#{rank}", conf, prior, boosted, f"{doc} | {sec}{tags} ({pages})")

        console.print(ret_table)
        console.print()

    # ── [2] Confidence Gate ───────────────────────────────────────────
    console.print(f"  [bold green]Step 2: Confidence Gate[/bold green] — [green]PASSED[/green] ({top1_conf:.1%} >= {CONFIDENCE_THRESHOLD:.0%})")

    # ── [3] LLM Generation ────────────────────────────────────────────
    if generation:
        provider = generation.get("provider", "unknown")
        model = generation.get("model", "unknown")
        gen_status = generation.get("status", "unknown")
        gen_ms = generation.get("latency_ms", 0)
        provider_tag = f"[bold green]{provider.upper()}/{model}[/bold green]" if gen_status == "real" else f"[bold yellow][MOCK][/bold yellow]"

        console.print(f"  [bold green]Step 3: Generation[/bold green] — {provider_tag} ({gen_ms:.0f}ms)")
        console.print()

        # Render the LLM response as markdown
        console.print(Panel(
            Markdown(result.get("llm_response_raw", response)),
            title=f"[bold]LLM Response [{provider.upper()}/{model}][/bold]",
            border_style="green" if gen_status == "real" else "yellow",
            box=box.ROUNDED,
        ))
        console.print()

    # ── [4] Citation Verification ─────────────────────────────────────
    if citations:
        verified = citations.get("verified_quotes", 0)
        total = citations.get("total_quotes", 0)
        cit_status = citations.get("status", "OK")
        if cit_status == "OK":
            console.print(f"  [bold green]Step 4: Citation Verification[/bold green] — [green]OK[/green] ({verified}/{total} quotes verified)")
        else:
            console.print(f"  [bold red]Step 4: Citation Verification[/bold red] — [red]{cit_status}[/red] ({verified}/{total} verified)")
            for uq in citations.get("unverified_quotes", []):
                console.print(f"    [red]Unverified:[/red] \"{uq[:60]}...\"")

    # ── [5] Disclaimer ────────────────────────────────────────────────
    console.print(f"  [bold green]Step 5: Disclaimer[/bold green] — [green]Appended[/green]")
    console.print(Panel(
        f"[dim]{PROFESSIONAL_DISCLAIMER}[/dim]",
        border_style="dim", box=box.SIMPLE,
    ))

    # ── [6] Schema Check ──────────────────────────────────────────────
    if schema:
        sec_count = schema.get("section_count", 0)
        all_present = schema.get("all_present", False)
        schema_tag = f"[green]{sec_count}/6 sections[/green]" if all_present else f"[yellow]{sec_count}/6 sections[/yellow]"
        console.print(f"  [bold]Schema Check:[/bold] {schema_tag} | Total Time: {total_ms:.0f}ms")

    console.print()


def run_test_suite() -> None:
    """Run preset clinical test queries through the full pipeline."""
    console.print(
        Panel(
            "[bold white]Executing Day 3 Pipeline Test Suite (7 Queries)[/bold white]",
            box=box.ROUNDED, border_style="magenta",
        )
    )
    for idx, query in enumerate(PRESET_TEST_QUERIES, 1):
        result = run_pipeline(query)
        display_pipeline_result(result, query_idx=idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="medAI Clinical RAG Pipeline CLI (Day 3)")
    parser.add_argument("query", nargs="?", type=str, help="Clinical query string, '/test', '/quit'")
    args = parser.parse_args()

    console.print(
        Panel(
            Align.center(
                Text.from_markup(
                    "[bold cyan]medAI — CLINICAL RAG PIPELINE CONSOLE[/bold cyan]\n"
                    f"[dim]LLM: {LLM_PROVIDER}/{LLM_MODEL} | Embedder: {EMBEDDING_MODEL} | Reranker: {RERANKER_MODEL}[/dim]\n"
                    "[dim white]Safety → Hybrid RRF → Cross-Encoder → Confidence Gate → OpenRouter LLM → Citation Verify → Disclaimer[/dim white]"
                )
            ),
            box=box.DOUBLE_EDGE,
            border_style="cyan",
        )
    )

    console.print("[dim]Initializing pipeline …[/dim]")

    # Command line argument handling
    if args.query:
        if args.query.strip().lower() in ("/test", "--test"):
            run_test_suite()
            return

        result = run_pipeline(args.query.strip())
        display_pipeline_result(result)
        return

    # Interactive Loop
    console.print("[bold white]Commands:[/bold white] Type any clinical question, [cyan]/test[/cyan] for test suite, or [cyan]/quit[/cyan] to exit.\n")
    while True:
        try:
            user_input = console.input("[bold cyan]medAI > [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[bold green]Goodbye![/bold green]")
            break

        if user_input.lower() == "/test":
            run_test_suite()
            continue

        result = run_pipeline(user_input)
        display_pipeline_result(result)


if __name__ == "__main__":
    main()
