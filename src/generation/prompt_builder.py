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

IMPORTANT RULES:
1. Base your answers ONLY on the provided context passages.
2. If the context does not contain enough information, say so explicitly.
3. Always cite the source document for each claim.
4. Never fabricate medical facts or recommendations.
5. Include a disclaimer that this is for informational purposes only and not a substitute for professional medical advice.
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
Answer the question using ONLY the context above. Cite sources in [Source: filename] format.
If the context is insufficient, state that clearly.
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        lines.append(f"[{i}] (Source: {source})\n{text}")
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
