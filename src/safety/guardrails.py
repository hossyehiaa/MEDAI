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
# CRISIS referral keywords & Personal Distress Detection
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

# Personal distress markers indicating acute crisis / ideation
PERSONAL_DISTRESS_PATTERNS: list[str] = [
    r"\bi\b.*\b(feel|want|going to|can'?t|need|wish|should|must|gonna|hopeless|suicidal)\b",
    r"\b(my|myself|me)\b.*\b(life|death|hopeless|pain|kill|die|end|hurt|suffer)\b",
    r"\b(feel|feeling)\b.*\b(hopeless|worthless|suicidal|empty|depressed|alone|done)\b",
    r"\b(kill myself|matarme|suicidarme|want to die|quiero morir|je veux mourir|me suicider|想死|不想活|muốn chết|أريد أن أموت)\b",
    r"\b(end my life|ending it all|hurt myself|hacerme daño|acabar con mi vida)\b",
]

CRISIS_MESSAGE: str = (
    "⚠️ If you are in crisis or having thoughts of suicide, "
    "call or text 988 (US) or your local emergency number, "
    "or go to your nearest emergency department immediately."
)

CRISIS_RESOURCE_LINE: str = (
    "⚠️ If you or someone you know is struggling or in crisis, help is available. "
    "Call or text 988 (US) or contact your local emergency services for immediate, confidential 24/7 support."
)

# ------------------------------------------------------------------
# DOSING / Medication refusal keywords & Patterns
# ------------------------------------------------------------------
ANTIDEPRESSANT_DRUGS: list[str] = [
    "sertraline", "zoloft",
    "fluoxetine", "prozac",
    "escitalopram", "lexapro",
    "citalopram", "celexa",
    "paroxetine", "paxil",
    "venlafaxine", "effexor",
    "duloxetine", "cymbalta",
    "bupropion", "wellbutrin",
    "mirtazapine", "remeron",
    "trazodone", "desvenlafaxine", "pristiq",
    "vilazodone", "vortioxetine", "trintellix",
    "amitriptyline", "nortriptyline", "imipramine",
]

DOSING_KEYWORDS: list[str] = [
    "dose",
    "dosing",
    "dosage",
    "starting dose",
    "typical dose",
    "maximum dose",
    "mg",
    "milligram",
    "milligrams",
    "mg per day",
    "mg/day",
    "mg/kg",
    "tablets",
    "pills",
    "prescribe",
    "prescription",
    "how many",
    "how much",
    "how to take",
    "amount",
    "typical amount",
    "titration",
    "titrate",
]

DOSING_PATTERNS: list[str] = [
    r"\b(how much|how many|what amount|typical amount|what dose|starting dose|dosage of|prescribe|prescribe me)\b.*\b(" + "|".join(ANTIDEPRESSANT_DRUGS) + r")\b",
    r"\b(" + "|".join(ANTIDEPRESSANT_DRUGS) + r")\b.*\b(dose|dosage|amount|mg|milligram|tablets?|pills?|take|taking|prescribe)\b",
    r"\b\d+\s*(?:mg|milligram|tablets?|pills?)\b.*\b(" + "|".join(ANTIDEPRESSANT_DRUGS) + r")\b",
    r"\b(" + "|".join(ANTIDEPRESSANT_DRUGS) + r")\b.*\b\d+\s*(?:mg|milligram|tablets?|pills?)\b",
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


def _is_personal_crisis(query_lower: str) -> bool:
    """Check if query contains personal distress markers or acute suicidal ideation."""
    for pattern in PERSONAL_DISTRESS_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    return False


def check_input(query: str) -> GuardrailResult:
    """
    Validate user input before it reaches the retrieval layer.

    Dual-Mode Crisis Logic:
      - CRISIS_REFUSAL: Crisis keyword + personal distress marker -> 988 referral, NO model answer.
      - CRISIS_RESOURCE: Crisis keyword in informational query -> proceed to retrieval, append 988 resource.
      - DOSING refusal: Medication / dosage queries -> out-of-scope refusal.
      - Prompt injection / disallowed topic blocking.
    """
    query_lower = query.lower()

    # ── Gate 1: CRISIS detection (highest priority) ───────────────
    has_crisis_keyword = any(kw in query_lower for kw in CRISIS_KEYWORDS)
    if has_crisis_keyword:
        if _is_personal_crisis(query_lower):
            logger.warning("CRISIS_REFUSAL triggered for acute crisis query: %s", query)
            return GuardrailResult(
                passed=False,
                status="CRISIS",
                reason="Personal crisis / suicidal distress detected.",
                message=CRISIS_MESSAGE,
                flags=["crisis_refusal"],
            )
        else:
            # Purely informational query about suicide screening / tools
            logger.info("CRISIS_RESOURCE flagged for informational query: %s", query)
            # Continues to retrieval but tags crisis_resource

    # ── Gate 2: DOSING / medication refusal ───────────────────────
    is_dosing_query = False
    for pat in DOSING_PATTERNS:
        if re.search(pat, query_lower):
            is_dosing_query = True
            break

    if not is_dosing_query:
        has_drug = any(d in query_lower for d in ANTIDEPRESSANT_DRUGS)
        has_dose_term = any(k in query_lower for k in ["dose", "dosing", "dosage", "amount", "mg", "tablets", "pills", "prescribe", "how many", "how much"])
        if has_drug and has_dose_term:
            is_dosing_query = True
        elif any(k in query_lower for k in ["starting dose", "typical dose", "maximum dose", "mg per day", "mg/day", "mg/kg", "prescribe me 50mg"]):
            is_dosing_query = True
        elif query_lower.startswith("what is the dose") or query_lower.startswith("what is the typical amount"):
            is_dosing_query = True

    if is_dosing_query:
        logger.warning("DOSING refusal triggered for query: %s", query)
        return GuardrailResult(
            passed=False,
            status="REFUSAL_OOS",
            reason="Medication dosing query is out of scope.",
            message=DOSING_REFUSAL_MESSAGE,
            flags=["dosing_refusal"],
        )

    # ── Gate 3: Prompt-injection and disallowed topics ────────────
    flags: list[str] = []
    if has_crisis_keyword:
        flags.append("crisis_resource")

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            flags.append(f"injection:{pattern}")

    for pattern in DISALLOWED_TOPICS:
        if re.search(pattern, query_lower):
            flags.append(f"disallowed_topic:{pattern}")

    if any(f.startswith("injection:") or f.startswith("disallowed_topic:") for f in flags):
        logger.warning("Input guardrail triggered: %s", flags)
        return GuardrailResult(
            passed=False,
            status="BLOCKED",
            reason="Your query was blocked by safety filters. Please rephrase.",
            message="Your query was blocked by safety filters. Please rephrase.",
            flags=flags,
        )

    return GuardrailResult(passed=True, status="OK", flags=flags)


def check_output(response: str) -> GuardrailResult:
    """
    Validate LLM output before delivering to the user.

    Ensures the response contains the required medical disclaimer
    and follows the structured 6-section schema.
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
# Citation Verification — Hardened verbatim quote checking
# ------------------------------------------------------------------
def is_invalid_or_metadata_quote(quote: str) -> tuple[bool, str]:
    """
    Check if a quote is a metadata fragment, table line, bibliography, or too short.
    """
    stripped = quote.strip()

    # (a) Starts with | or ---
    if stripped.startswith("|") or stripped.startswith("---") or "| --- |" in stripped:
        return True, "Starts with table markdown syntax or pipes/dashes"

    # (b) Contains PMID: or et al.
    if "PMID:" in quote or "pmid:" in quote.lower() or "et al." in quote or "et al," in quote:
        return True, "Contains bibliographic citation fragment (PMID/et al.)"

    # (c) Matches Table TOC pattern
    if re.search(r"^\s*Table\s+\d+\.", quote, re.IGNORECASE):
        return True, "Matches Table TOC header pattern"

    # (d) Has <40 alphanumeric characters
    alpha_chars = re.sub(r"[^a-zA-Z0-9]", "", quote)
    if len(alpha_chars) < 40:
        return True, f"Too short ({len(alpha_chars)} alphanumeric chars < 40)"

    return False, ""


def verify_citations(
    llm_response: str,
    context_chunks: list[dict],
) -> dict:
    """
    Extract and verify every verbatim citation quote in the LLM response.

    Rejects metadata fragments, table headers, PMIDs, or short fragments (<40 chars).
    Ensures that every valid quote appears verbatim in at least one retrieved context chunk.
    """
    quote_pattern = re.compile(r'Quote:\s*"([^"]+)"', re.IGNORECASE)
    quotes = quote_pattern.findall(llm_response)

    if not quotes:
        return {
            "status": "CITATION_VERIFICATION_FAILED",
            "verified_quotes": 0,
            "total_quotes": 0,
            "unverified_quotes": ["No citations found in response."],
            "detail": "Response contains zero citation quotes.",
        }

    # Normalize text for comparison
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    corpus_texts = [_normalize(c.get("text", "")) for c in context_chunks]

    verified: list[str] = []
    unverified: list[str] = []

    for quote in quotes:
        # Step 1: Reject metadata-only or short quotes
        is_invalid, reason = is_invalid_or_metadata_quote(quote)
        if is_invalid:
            unverified.append(f"{quote} (REJECTED: {reason})")
            continue

        # Step 2: Check normalized verbatim match in context corpus
        norm_quote = _normalize(quote)
        found = any(norm_quote in corpus for corpus in corpus_texts)
        if found:
            verified.append(quote)
        else:
            unverified.append(quote)

    status = "OK" if len(unverified) == 0 and len(verified) > 0 else "CITATION_VERIFICATION_FAILED"

    return {
        "status": status,
        "verified_quotes": len(verified),
        "total_quotes": len(quotes),
        "unverified_quotes": unverified,
    }


# ------------------------------------------------------------------
# Post-Generation Faithfulness & Caveat Verification
# ------------------------------------------------------------------
def check_faithfulness(
    query: str,
    response: str,
    context_chunks: list[dict],
) -> dict[str, Any]:
    """
    Perform post-generation clinical faithfulness checks:
      - Caveat preservation ("no evidence on frequency", "only 1 study", "uncertainty")
      - External organization attribution (AAFP/ICSI distinct from USPSTF)
      - Scope gap acknowledgment (adolescents/children -> adults only)
      - Instrument distinction (PHQ-9/EPDS are depression tools, not primary suicide tools)
    """
    flags: list[str] = []
    resp_lower = response.lower()
    q_lower = query.lower()

    # 1. Caveat preservation
    caveat_patterns = ["no evidence", "insufficient", "uncertainty", "only 1 study", "one study", "remains uncertain", "few studies", "limited evidence"]
    context_has_caveat = any(
        any(pat in c.get("text", "").lower() for pat in caveat_patterns)
        for c in context_chunks
    )
    if context_has_caveat:
        caveat_words = ["insufficient", "uncertain", "limited", "no evidence", "caveat", "gap", "one study", "only 1 study", "lacking", "absence of evidence"]
        has_caveat_word = any(cw in resp_lower for cw in caveat_words)
        if not has_caveat_word:
            flags.append("CAVEAT_SUPPRESSED")

    # 2. External organization attribution
    orgs = ["AAFP", "ICSI", "APA", "ACCP", "ACOG", "AWHONN", "VA/DoD", "NICE"]
    context_has_org = any(
        any(org.lower() in c.get("text", "").lower() for org in orgs)
        for c in context_chunks
    )
    if context_has_org:
        distinction_phrases = ["distinct from uspstf", "aligns with but is distinct", "recommendation, distinct", "recommendations of other", "other organization", "aafp recommends"]
        has_distinction = any(dp in resp_lower for dp in distinction_phrases) or ("distinct" in resp_lower and "uspstf" in resp_lower)
        if not has_distinction:
            flags.append("ATTRIBUTION_MISSING")

    # 3. Scope gap acknowledgment
    is_pediatric_query = any(w in q_lower for w in ["adolescent", "adolescents", "teen", "teens", "child", "children", "pediatric", "youth"])
    if is_pediatric_query:
        scope_phrases = ["does not address", "adults only", "18 years and older", "not address children", "not address adolescent", "limited to adult"]
        has_scope_ack = any(sp in resp_lower for sp in scope_phrases)
        if not has_scope_ack:
            flags.append("SCOPE_UNACKNOWLEDGED")

    # 4. Instrument distinction
    if "suicide" in q_lower and "instrument" in q_lower:
        if "phq-9 is a suicide" in resp_lower or "epds is a suicide" in resp_lower:
            flags.append("INSTRUMENT_CONFLATED")

    passed = len(flags) == 0
    return {
        "passed": passed,
        "flags": flags,
        "detail": f"Faithfulness checks {'passed' if passed else 'failed: ' + ', '.join(flags)}",
    }


