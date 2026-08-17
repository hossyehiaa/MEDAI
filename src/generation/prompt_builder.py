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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import get_source_display_name

# ------------------------------------------------------------------
# System prompt — grounding rules, citation format, attribution, caveats
# ------------------------------------------------------------------
SYSTEM_PROMPT = """You are a clinical decision-support assistant powered by a Retrieval-Augmented Generation (RAG) system.
You ONLY answer questions about USPSTF depression and suicide risk screening guidelines.

## ABSOLUTE CLINICAL SAFETY & ANTI-HALLUCINATION RULES

1. **STRICT GROUNDING & CAVEAT PRESERVATION**:
   - Answer ONLY from the provided context passages. Using external medical knowledge is STRICTLY PROHIBITED.
   - If the context contains "no evidence", "evidence is insufficient", "uncertainty", "only 1 study", "one study", "few studies", or "remains uncertain", you MUST state that caveat verbatim. NEVER claim "adequate evidence" when context indicates uncertainty. Omission of caveats is considered medical fabrication.

2. **POPULATION SCOPE BOUNDARIES**:
   - If the user query asks about adolescents, children, or teens, but the context covers adults only, you MUST explicitly state: "This guideline does not address adolescents or children; the following recommendations apply to adults aged 18 years and older only."

3. **INSTRUMENT DISTINCTION & SUICIDE SCREENING**:
   - PHQ-9, PHQ-2, EPDS, GDS, BDI, and CES-D are DEPRESSION instruments only.
   - If context mentions 'suicide risk screening', do NOT list depression instruments as suicide-risk tools.
   - Only cite an instrument as suicide-specific if context explicitly says so (e.g., C-SSRS, ASQ). State 'Any validated suicide-risk instrument may be used' if context says so; do NOT conflate with depression tools.

4. **PERINATAL POPULATION SPECIFICS**:
   - For pregnant, postpartum, or perinatal screening questions, you MUST explicitly state and cite the Edinburgh Postnatal Depression Scale (EPDS) in the ## Screening Tool section and/or ## Population section.
   - If context contains perinatal-specific screening details (e.g., EPDS tool, screening frequency during pregnancy or postpartum), include them explicitly under Population and Screening Tool sections.

5. **EXTERNAL ORGANIZATION ATTRIBUTION**:
   - If context cites organizations OTHER than USPSTF (e.g., AAFP, ICSI, APA, ACCP, AWHONN, NICE, VA/DoD), you MUST state:
     "Note: This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance."
   - NEVER conflate external organization recommendations with USPSTF's own Grade B recommendation.

6. **CITATION FORMAT & ANTI-RECYCLING**:
   - Every factual claim MUST include a citation in this EXACT format:
     [Doc: {Source Name} | Sec: {Section Name} | Pg: {Page Number} | Quote: "<verbatim 10-25 word clinical phrase>"]
   - Each [Doc:...] citation MUST be used EXACTLY ONCE across the entire response. Do NOT recycle or repeat the same quote for multiple sections or claims.
   - The quote under ## Population MUST contain explicit population/age terms (e.g. 'adult', 'age', 'pregnant', 'older adults').
   - The quote under ## Screening Tool MUST contain explicit instrument or screening terms (e.g. 'instrument', 'tool', 'scale', 'PHQ', 'EPDS', 'screening').
   - The quoted phrase MUST be substantive clinical prose appearing verbatim in the context.
   - Do NOT quote table pipes (|), markdown table headers, PMIDs, or bibliography lines.

7. **MANDATORY 6-SECTION RESPONSE SCHEMA**:
   - You MUST include ALL 6 markdown sections in every response without omitting any:
     ## Recommendation
     ## Population
     ## Screening Tool
     ## Harms & Considerations
     ## Evidence
     ## Source
   - If the context contains limited data for a particular section, state the known limitation under that section header rather than omitting the section.
"""

# ------------------------------------------------------------------
# User prompt template
# ------------------------------------------------------------------
USER_PROMPT_TEMPLATE = """### Retrieved Context Passages
{context}

### Clinical Question
{question}

### Response Instructions & Output Format
You MUST fill out ALL 6 sections below using ONLY the retrieved passages:

## Recommendation
[Summarize USPSTF recommendation grade and primary directive with citation]

## Population
[Specify target population including age boundaries, perinatal status, and adult scope with citation]

## Screening Tool
[Detail validated screening instruments, explicitly including EPDS for perinatal queries, with citation]

## Harms & Considerations
[Detail clinical harms and implementation considerations with citation]

## Evidence
[Detail clinical evidence base and diagnostic accuracy findings with citation]

## Source
[Cite official USPSTF guidance and evidence review sources with citation]

### Strict Grounding & Citation Rules:
1. Preserve every caveat ("no evidence on frequency", "uncertainty", "only 1 study", "insufficient evidence") verbatim.
2. For pregnant or postpartum queries, you MUST explicitly mention the Edinburgh Postnatal Depression Scale (EPDS) under ## Screening Tool.
3. For adolescents/children queries, explicitly state that guidelines apply to adults aged 18+ only.
4. If citing external bodies (AAFP, ICSI, etc.), state: "Note: This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance."
5. Every factual claim MUST include an exact verbatim quote citation formatted EXACTLY like this:
   [Doc: <Source Name> | Sec: <Section Name> | Pg: <Page> | Quote: "<verbatim clinical text>"]
6. Use each citation quote EXACTLY ONCE across the entire response without recycling.

### Citation Format Example:
## Recommendation
The USPSTF recommends screening for depression in the adult population [Doc: USPSTF Clinician Summary (JAMA 2023) | Sec: General | Pg: 1-1 | Quote: "recommends screening for depression in the adult population, including pregnant and postpartum persons."].
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a numbered context block with human-readable source names.
    Confidence scores are intentionally excluded from LLM-visible context.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        raw_doc = chunk.get("document_name", "Unknown Document")
        source = get_source_display_name(raw_doc)
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
