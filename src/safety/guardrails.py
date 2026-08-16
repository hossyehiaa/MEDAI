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
# ------------------------------------------------------------------
CRISIS_KEYWORDS: list[str] = [
    "suicide",
    "kill myself",
    "end my life",
    "self-harm",
    "want to die",
    "ending it all",
    "hurt myself",
]

CRISIS_MESSAGE: str = (
    "⚠️ If you are in crisis or having thoughts of suicide, "
    "call or text 988 (Suicide & Crisis Lifeline) in the US, "
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
    for keyword in CRISIS_KEYWORDS:
        if keyword in query_lower:
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

    Ensures the response contains a medical disclaimer.
    """
    flags: list[str] = []

    if REQUIRED_DISCLAIMER_FRAGMENT not in response.lower():
        flags.append("missing_disclaimer")

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
