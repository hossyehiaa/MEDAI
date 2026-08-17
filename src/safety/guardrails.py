"""
Guardrails — Input validation and output safety checks.

Enforces medical-domain safety policies:
- **CRISIS gate**: Detects suicide/self-harm language and returns 988 Lifeline referral
- **DOSING gate**: Refuses medication dosing queries (out-of-scope for screening guidelines)
- Blocks prompt-injection attempts
- Filters disallowed topics (prescriptions, diagnoses)
- Ensures responses include required disclaimers
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Outcome of a guardrail check."""

    passed: bool
    status: str = "OK"          # OK | CRISIS | REFUSAL_OOS | BLOCKED | FAILED_OUTPUT
    reason: str = ""
    message: str = ""           # User-facing message (for CRISIS / REFUSAL_OOS)
    flags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# CRISIS referral keywords — checked FIRST, before retrieval
# Supports multilingual crisis detection (EN, ES, FR, ZH, VI, AR)
# ------------------------------------------------------------------
CRISIS_KEYWORDS: list[str] = [
    # English
    "suicide",
    "kill myself",
    "end my life",
    "self-harm",
    "want to die",
    "ending it all",
    "hurt myself",
    # Spanish
    "quiero morir",
    "hacerme daño",
    "matarme",
    "suicidarme",
    "acabar con mi vida",
    # French
    "je veux mourir",
    "me suicider",
    # Chinese
    "想死",
    "自杀",
    "不想活",
    # Vietnamese
    "muốn chết",
    "tự tử",
    # Arabic
    "أريد أن أموت",
    "انتحر",
    "قتل نفسي",
    "أذية نفسي",
]

CRISIS_MESSAGE: str = (
    "⚠️ If you are in crisis or having thoughts of suicide, "
    "call or text 988 (US) or your local emergency number, "
    "or go to your nearest emergency department immediately."
)

# ------------------------------------------------------------------
# DOSING / Medication refusal keywords — out-of-scope for screening
# ------------------------------------------------------------------
DOSING_KEYWORDS: list[str] = [
    "dose",
    "mg",
    "prescribe",
    "milligrams",
    "dosage",
    "how much to take",
    "pills per day",
    "sertraline",
    "fluoxetine",
    "escitalopram",
    "zoloft",
    "prozac",
    "lexapro",
]

DOSING_REFUSAL_MESSAGE: str = (
    "This system provides screening recommendations only. "
    "For medication dosing, please consult a licensed prescriber."
)

# ------------------------------------------------------------------
# Prompt-injection patterns
# ------------------------------------------------------------------
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"pretend\s+you\s+are",
    r"override\s+(your\s+)?system\s+prompt",
    r"disregard\s+(all\s+)?(prior|previous)",
]

DISALLOWED_TOPICS: list[str] = [
    r"prescri(be|ption)\s+me",
    r"give\s+me\s+a\s+diagnosis",
    r"what\s+medication\s+should\s+I\s+take",
]

REQUIRED_DISCLAIMER_FRAGMENT = "not a substitute for professional medical"

# ------------------------------------------------------------------
# Professional disclaimer — always appended to every response
# ------------------------------------------------------------------
PROFESSIONAL_DISCLAIMER: str = (
    "This information is based on USPSTF guidance current as of June 2023 "
    "and is for clinical decision support only. It is not a substitute for "
    "professional medical judgment. Always verify current guidelines and "
    "consult appropriate specialists for individual patient care."
)


def check_input(query: str) -> GuardrailResult:
    """
    Validate user input before it reaches the retrieval layer.

    Priority order:
      1. CRISIS detection → immediate 988 referral
      2. DOSING / medication refusal → out-of-scope refusal
      3. Prompt-injection / disallowed topic blocking

    Returns a :class:`GuardrailResult` with ``passed=False`` if the
    query triggers any safety rule.
    """
    query_lower = query.lower()

    # ── Gate 1: CRISIS detection (highest priority) ───────────────
    is_clinical_query = any(term in query_lower for term in ["screen", "tool", "instrument", "guideline", "uspstf", "recommend", "how to assess"])
    for keyword in CRISIS_KEYWORDS:
        if keyword in query_lower:
            if keyword == "suicide" and is_clinical_query and not any(k in query_lower for k in ["myself", "i want", "feel like", "going to", "kill", "commit", "die"]):
                continue
            logger.warning("CRISIS gate triggered for keyword '%s'", keyword)
            return GuardrailResult(
                passed=False,
                status="CRISIS",
                reason="Crisis/suicide language detected.",
                message=CRISIS_MESSAGE,
                flags=[f"crisis:{keyword}"],
            )

    # ── Gate 2: DOSING / medication refusal ───────────────────────
    for keyword in DOSING_KEYWORDS:
        if keyword in query_lower:
            logger.warning("DOSING refusal triggered for keyword '%s'", keyword)
            return GuardrailResult(
                passed=False,
                status="REFUSAL_OOS",
                reason="Medication dosing query is out of scope.",
                message=DOSING_REFUSAL_MESSAGE,
                flags=[f"dosing_oos:{keyword}"],
            )

    # ── Gate 3: Prompt-injection and disallowed topics ────────────
    flags: list[str] = []

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            flags.append(f"injection:{pattern}")

    for pattern in DISALLOWED_TOPICS:
        if re.search(pattern, query_lower):
            flags.append(f"disallowed_topic:{pattern}")

    if flags:
        logger.warning("Input guardrail triggered: %s", flags)
        return GuardrailResult(
            passed=False,
            status="BLOCKED",
            reason="Your query was blocked by safety filters. Please rephrase.",
            message="Your query was blocked by safety filters. Please rephrase.",
            flags=flags,
        )

    return GuardrailResult(passed=True, status="OK")


def check_output(response: str) -> GuardrailResult:
    """
    Validate LLM output before returning to the user.

    Checks:
      1. Medical disclaimer presence
      2. 6-section schema presence
    """
    flags: list[str] = []

    if REQUIRED_DISCLAIMER_FRAGMENT not in response.lower():
        flags.append("missing_disclaimer")

    schema_result = check_response_schema(response)
    if not schema_result["all_present"]:
        for missing in schema_result["missing_sections"]:
            flags.append(f"missing_section:{missing}")

    if flags:
        logger.warning("Output guardrail triggered: %s", flags)
        return GuardrailResult(
            passed=False,
            status="FAILED_OUTPUT",
            reason="Response failed output safety checks.",
            message="Response failed output safety checks.",
            flags=flags,
        )

    return GuardrailResult(passed=True, status="OK")


# ------------------------------------------------------------------
# 6-Section Schema Validation
# ------------------------------------------------------------------
REQUIRED_SECTIONS: list[str] = [
    "## Recommendation",
    "## Population",
    "## Screening Tool",
    "## Harms & Considerations",
    "## Evidence",
    "## Source",
]


def check_response_schema(response: str) -> dict:
    """
    Verify the LLM response contains all 6 required sections.

    Returns
    -------
    dict
        Keys: 'all_present' (bool), 'present_sections' (list),
        'missing_sections' (list), 'section_count' (int).
    """
    response_lower = response.lower()
    present: list[str] = []
    missing: list[str] = []

    for section in REQUIRED_SECTIONS:
        if section.lower() in response_lower:
            present.append(section)
        else:
            missing.append(section)

    return {
        "all_present": len(missing) == 0,
        "present_sections": present,
        "missing_sections": missing,
        "section_count": len(present),
    }


# ------------------------------------------------------------------
# Citation Verification — Verbatim quote checking
# ------------------------------------------------------------------
def verify_citations(
    llm_response: str,
    context_chunks: list[dict],
) -> dict:
    """
    Extract and verify every verbatim citation quote in the LLM response.

    Regex-extracts every ``Quote: "..."`` from the response and checks
    that each quoted phrase appears (normalized) in at least one of the
    retrieved context chunks.

    Parameters
    ----------
    llm_response : str
        The LLM-generated response text.
    context_chunks : list[dict]
        Retrieved context chunks with 'text' field.

    Returns
    -------
    dict
        Keys: 'status' ('OK'|'CITATION_VERIFICATION_FAILED'),
        'verified_quotes' (int), 'total_quotes' (int),
        'unverified_quotes' (list[str]).
    """
    # Extract all Quote: "..." patterns
    quote_pattern = re.compile(r'Quote:\s*"([^"]+)"', re.IGNORECASE)
    quotes = quote_pattern.findall(llm_response)

    if not quotes:
        return {
            "status": "OK",
            "verified_quotes": 0,
            "total_quotes": 0,
            "unverified_quotes": [],
            "detail": "No citations found in response.",
        }

    # Normalize text for comparison
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    # Build normalized corpus from context chunks
    corpus_texts = [_normalize(c.get("text", "")) for c in context_chunks]

    verified: list[str] = []
    unverified: list[str] = []

    for quote in quotes:
        norm_quote = _normalize(quote)
        found = any(norm_quote in corpus for corpus in corpus_texts)
        if found:
            verified.append(quote)
        else:
            unverified.append(quote)

    status = "OK" if len(unverified) == 0 else "CITATION_VERIFICATION_FAILED"

    return {
        "status": status,
        "verified_quotes": len(verified),
        "total_quotes": len(quotes),
        "unverified_quotes": unverified,
    }

