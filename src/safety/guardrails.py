"""
Guardrails — Input validation and output safety checks.

Enforces medical-domain safety policies:
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
    reason: str = ""
    flags: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Blocked patterns (extend as needed)
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

REQUIRED_DISCLAIMER_FRAGMENT = "not a substitute for professional medical advice"


def check_input(query: str) -> GuardrailResult:
    """
    Validate user input before it reaches the LLM.

    Returns a :class:`GuardrailResult` with ``passed=False`` if the
    query triggers any safety rule.
    """
    flags: list[str] = []
    query_lower = query.lower()

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
            reason="Your query was blocked by safety filters. Please rephrase.",
            flags=flags,
        )

    return GuardrailResult(passed=True)


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
            reason="Response failed output safety checks.",
            flags=flags,
        )

    return GuardrailResult(passed=True)
