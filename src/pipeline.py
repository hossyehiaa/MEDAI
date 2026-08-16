"""
RAG Pipeline — End-to-end orchestrator for the medAI system.

Usage:
    # Ingest all PDFs
    python -m src.pipeline ingest

    # Query the system
    python -m src.pipeline query "What are the symptoms of depression?"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_text, save_chunks
from src.retrieval.vector_store import VectorStore
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import LLMClient
from src.safety.guardrails import check_input, check_output

# Load environment variables
load_dotenv("configs/.env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Paths (override via env vars or configs/.env)
# ------------------------------------------------------------------
RAW_DOCS_DIR = Path("raw_documents")
CHUNKS_DIR = Path("data/chunks")
VECTOR_DB_DIR = Path("data/vector_db")


def ingest(pdf_dir: Path = RAW_DOCS_DIR) -> None:
    """Parse all PDFs, chunk them, and upsert into the vector store."""
    store = VectorStore(persist_dir=VECTOR_DB_DIR)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDFs found in %s", pdf_dir)
        return

    logger.info("Found %d PDF(s) to ingest.", len(pdf_files))

    for pdf_path in pdf_files:
        text = parse_pdf(pdf_path)
        chunks = chunk_text(text, source=pdf_path.name)
        save_chunks(chunks, CHUNKS_DIR)
        store.add_chunks(chunks)

    logger.info("Ingestion complete. Vector store has %d documents.", store.count())


def query(question: str, top_k: int = 5) -> str:
    """Run the full RAG pipeline: validate → retrieve → generate → validate."""
    # 1. Input guardrails
    input_check = check_input(question)
    if not input_check.passed:
        return f"⚠️  {input_check.reason}"

    # 2. Retrieve
    store = VectorStore(persist_dir=VECTOR_DB_DIR)
    hits = store.search(question, top_k=top_k)

    if not hits:
        return "No relevant documents found. Please ingest PDFs first."

    # 3. Generate
    messages = build_prompt(question, hits)
    llm = LLMClient()
    response = llm.generate(messages)

    # 4. Output guardrails
    output_check = check_output(response)
    if not output_check.passed:
        # Append disclaimer automatically if missing
        response += (
            "\n\n---\n⚠️ **Disclaimer**: This information is for educational "
            "purposes only and is not a substitute for professional medical advice."
        )

    return response


def main() -> None:
    parser = argparse.ArgumentParser(description="medAI RAG Pipeline")
    sub = parser.add_subparsers(dest="command")

    # ingest
    ingest_cmd = sub.add_parser("ingest", help="Ingest PDFs into the vector store")
    ingest_cmd.add_argument("--dir", type=Path, default=RAW_DOCS_DIR)

    # query
    query_cmd = sub.add_parser("query", help="Ask a question")
    query_cmd.add_argument("question", type=str)
    query_cmd.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "ingest":
        ingest(pdf_dir=args.dir)
    elif args.command == "query":
        answer = query(args.question, top_k=args.top_k)
        print("\n" + answer)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
