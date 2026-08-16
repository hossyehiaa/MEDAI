"""
Prompt Builder — Constructs LLM prompts for the medical RAG pipeline.

Centralises all prompt templates and provides helper functions to
assemble the final prompt from a user query + retrieved context chunks.
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a medical information assistant powered by a Retrieval-Augmented Generation (RAG) system.
You ONLY answer questions about USPSTF depression and suicide risk screening guidelines.

CRITICAL RULES:
1. Base your answers ONLY on the provided context passages. Never fabricate facts.
2. If the context does not contain enough information to answer, say so explicitly.
3. Never fabricate medical facts, recommendations, or screening guidance.
4. CITATION FORMAT — Every claim MUST include a citation in this exact format:
   [Doc: {document name} | Sec: {section name} | Pg: {page number} | Quote: "<verbatim 10-20 word phrase from the chunk>"]
   The quoted phrase must appear verbatim in the retrieved chunk text.
5. SOURCE ATTRIBUTION — If retrieved context cites organizations OTHER than USPSTF
   (e.g., AAFP, ICSI, APA, ACCP, AMA), you MUST state:
   "Note: This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance."
   NEVER conflate external organization recommendations with USPSTF's own Grade B recommendation.
6. When discussing screening tools (PHQ-2, PHQ-9, EPDS, Edinburgh, etc.), always specify
   which population each tool is validated for and what the USPSTF evidence review found.
7. Do NOT provide medication dosing, treatment protocols, or diagnostic conclusions.
   This system covers screening recommendations only.
"""

# ------------------------------------------------------------------
# User prompt template
# ------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """
### Retrieved Context
{context}

### Question
{question}

### Instructions
Answer the question using ONLY the context above.
For every factual claim, include a citation in this exact format:
[Doc: <document name> | Sec: <section name> | Pg: <page> | Quote: "<verbatim 10-20 word phrase>"]

If the context mentions recommendations from organizations other than USPSTF, clearly distinguish them:
"Note: This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance."

If the context is insufficient to answer, state that clearly.
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block with full metadata."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", chunk.get("document_name", "unknown"))
        section = chunk.get("section_name", "unknown")
        start_page = chunk.get("start_page", "?")
        end_page = chunk.get("end_page", "?")
        text = chunk.get("text", "")
        confidence = chunk.get("confidence", chunk.get("boosted_score", 0.0))

        header = (
            f"[{i}] (Source: {source} | Section: {section} | "
            f"Pages: {start_page}-{end_page} | Confidence: {confidence:.1%})"
        )
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def build_prompt(
    question: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Build the chat-completion messages list.

    Returns a list of ``{"role": ..., "content": ...}`` dicts
    compatible with the OpenAI chat API.
    """
    context_block = format_context(chunks)
    user_content = USER_PROMPT_TEMPLATE.format(context=context_block, question=question)

    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content.strip()},
    ]
