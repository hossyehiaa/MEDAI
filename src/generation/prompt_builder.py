"""
Prompt Builder — Constructs grounded LLM prompts for the medical RAG pipeline.

Assembles the system prompt (anti-hallucination rules, citation format, attribution
rules) and the user prompt (context chunks + query) into an OpenAI-compatible
messages list.

Key Rules Enforced in Prompt:
  1. Answer ONLY from provided context — external knowledge PROHIBITED
  2. Insufficient context → status REFUSAL INSUFFICIENT_EVIDENCE
  3. AAFP/ICSI/APA/ACCP attribution → "Note: This is [Org]'s recommendation..."
  4. Citations: [Doc: X | Sec: Y | Pg: Z | Quote: "<verbatim 10-20 word phrase>"]
  5. 6-section schema: Recommendation / Population / Screening Tool / Harms / Evidence / Source
  6. If evidence insufficient for an aspect, state limitation — never fabricate
  7. NO confidence scores in LLM-visible context
  8. If diversity_warning, prepend cross-reference note

Usage:
    from src.generation.prompt_builder import build_prompt
    messages = build_prompt(query, context_chunks, diversity_warning=False)
"""

from __future__ import annotations

from typing import Any

# ------------------------------------------------------------------
# System prompt — grounding rules, citation format, attribution
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical decision-support assistant powered by a Retrieval-Augmented Generation (RAG) system.
You ONLY answer questions about USPSTF depression and suicide risk screening guidelines.

## ABSOLUTE RULES

1. **GROUNDING**: Answer ONLY from the provided context passages. Using external medical knowledge is STRICTLY PROHIBITED. If you don't find the answer in the context, say so.

2. **INSUFFICIENT EVIDENCE**: If the provided context does not contain enough information to fully answer the question, respond with:
   "Based on the available USPSTF guideline excerpts, there is insufficient evidence in the retrieved passages to fully address this question. Please consult the full guideline document or a clinical specialist."

3. **SOURCE ATTRIBUTION**: If retrieved context cites organizations OTHER than USPSTF (e.g., AAFP, ICSI, APA, ACCP, AMA, VA/DoD), you MUST state:
   "Note: This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance."
   NEVER conflate external organization recommendations with USPSTF's own Grade B recommendation.

4. **CITATION FORMAT** — Every factual claim MUST include a citation in this EXACT format:
   [Doc: {document name} | Sec: {section name} | Pg: {page number} | Quote: "<verbatim 10-20 word phrase from the context>"]
   The quoted phrase MUST appear verbatim in the provided context. Do NOT paraphrase or invent quotes.

5. **6-SECTION RESPONSE SCHEMA** — Structure your answer with exactly these 6 markdown sections:
   ## Recommendation
   ## Population
   ## Screening Tool
   ## Harms & Considerations
   ## Evidence
   ## Source

6. **LIMITATIONS**: If the context says "evidence is insufficient" or "evidence is lacking" for an aspect, state that limitation explicitly. Do NOT fabricate findings.

7. **PROHIBITED CONTENT**: Do NOT provide medication dosing, treatment protocols, or diagnostic conclusions. This system covers screening recommendations only.
"""

# ------------------------------------------------------------------
# User prompt template
# ------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """### Retrieved Context Passages
{context}

### Clinical Question
{question}

### Response Instructions
1. Answer using ONLY the context passages above. Do NOT use external knowledge.
2. For EVERY factual claim, include a citation:
   [Doc: <document name> | Sec: <section name> | Pg: <page> | Quote: "<verbatim 10-20 word phrase>"]
3. If context cites AAFP/ICSI/APA or other organizations, distinguish from USPSTF:
   "Note: This is [Organization]'s recommendation, distinct from USPSTF guidance."
4. Structure your response using exactly these 6 sections:
   ## Recommendation, ## Population, ## Screening Tool, ## Harms & Considerations, ## Evidence, ## Source
5. If context is insufficient for any section, state the limitation — never fabricate.
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block.

    Confidence scores are intentionally EXCLUDED from LLM-visible context
    to prevent the model from using them as authority signals.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("document_name", "Unknown Document")
        section = chunk.get("section_name", "General")
        start_page = chunk.get("start_page", "?")
        end_page = chunk.get("end_page", "?")
        text = chunk.get("text", "")

        header = (
            f"[Passage {i}] (Source: {source} | Section: {section} | "
            f"Pages: {start_page}-{end_page})"
        )
        lines.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(lines)


def build_prompt(
    query: str,
    context_chunks: list[dict[str, Any]],
    diversity_warning: bool = False,
) -> tuple[str, str]:
    """
    Build the system and user prompts for the LLM.

    Parameters
    ----------
    query : str
        The clinical question.
    context_chunks : list[dict]
        Retrieved context passages with metadata.
    diversity_warning : bool, optional
        If True, prepends a cross-reference note to the user prompt.

    Returns
    -------
    tuple[str, str]
        (system_prompt, user_prompt) pair.
    """
    context_block = format_context(context_chunks)
    user_content = USER_PROMPT_TEMPLATE.format(context=context_block, question=query)

    if diversity_warning:
        cross_ref_note = (
            "**Note**: The retrieved passages come from a limited set of source documents. "
            "Cross-reference with other USPSTF guideline sections for completeness.\n\n"
        )
        user_content = cross_ref_note + user_content

    return SYSTEM_PROMPT.strip(), user_content.strip()
