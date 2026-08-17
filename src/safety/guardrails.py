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
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import SOURCE_DISPLAY_NAMES

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
    "sin esperanza",
    "no tengo salida",
    "me siento vacio",
    "me siento vacío",
    # French
    "je veux mourir",
    "me suicider",
    "désespéré",
    "desespere",
    "sans espoir",
    "je veux en finir",
    # Chinese
    "想死",
    "自杀",
    "不想活",
    "感到绝望",
    "没有希望",
    "想结束",
    # Vietnamese
    "muốn chết",
    "tự tử",
    "tuyệt vọng",
    "không còn hy vọng",
    "muốn kết thúc",
    # Arabic
    "أريد أن أموت",
    "انتحر",
    "قتل نفسي",
    "أذية نفسي",
    "أشعر باليأس",
    "لا يوجد أمل",
    "أريد إنهاء حياتي",
    "تعبت من الحياة",
]

# Personal distress markers indicating acute crisis / ideation (Defect 6: multilingual)
PERSONAL_DISTRESS_PATTERNS: list[str] = [
    # English
    r"\bi\b.*\b(feel|want|going to|can'?t|need|wish|should|must|gonna|hopeless|suicidal)\b",
    r"\b(my|myself|me)\b.*\b(life|death|hopeless|pain|kill|die|end|hurt|suffer)\b",
    r"\b(feel|feeling)\b.*\b(hopeless|worthless|suicidal|empty|depressed|alone|done)\b",
    r"\b(kill myself|want to die|end my life|ending it all|hurt myself)\b",
    # Spanish ('sin esperanza', 'quiero terminar', 'no tengo salida', 'me siento vacío', 'me siento sin esperanza')
    r"\b(sin esperanza|quiero terminar|no tengo salida|me siento vac[ií]o|me siento sin esperanza|quiero morir|matarme|suicidarme|hacerme da[ñn]o|acabar con mi vida)\b",
    # Chinese ('感到绝望', '没有希望', '想结束', '不想活')
    r"(感到绝望|没有希望|想结束|不想活|想死|自杀)",
    # Arabic ('أشعر باليأس', 'لا يوجد أمل', 'أريد إنهاء حياتي', 'تعبت من الحياة')
    r"(أشعر باليأس|لا يوجد أمل|أريد إنهاء حياتي|تعبت من الحياة|أريد أن أموت|انتحر|قتل نفسي|أذية نفسي)",
    # Vietnamese ('tuyệt vọng', 'không còn hy vọng', 'muốn kết thúc')
    r"(tuyệt vọng|không còn hy vọng|muốn kết thúc|muốn chết|tự tử)",
    # French ('désespéré', 'sans espoir', 'je veux en finir')
    r"\b(d[eé]sesp[eé]r[eé]|sans espoir|je veux en finir|je veux mourir|me suicider)\b",
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
    "how many mg",
    "how much",
    "prescribe",
    "prescription",
    "titrate",
    "titration",
    "schedule",
    "typical amount",
]

DOSING_PATTERNS: list[str] = [
    r"\b(dose|dosing|dosage|amount|mg|milligram|tablets?|pills?|schedule|titrat\w*)\b.*\b(sertraline|zoloft|fluoxetine|prozac|escitalopram|lexapro|citalopram|celexa|paroxetine|paxil|venlafaxine|effexor|duloxetine|cymbalta|bupropion|wellbutrin|mirtazapine|remeron|trazodone)\b",
    r"\b(sertraline|zoloft|fluoxetine|prozac|escitalopram|lexapro|citalopram|celexa|paroxetine|paxil|venlafaxine|effexor|duloxetine|cymbalta|bupropion|wellbutrin|mirtazapine|remeron|trazodone)\b.*\b(dose|dosing|dosage|amount|mg|milligram|tablets?|pills?|schedule|titrat\w*)\b",
    r"\bprescribe\s+(?:me\s+)?(?:\d+\s*mg\s+)?(?:sertraline|zoloft|fluoxetine|prozac|escitalopram|lexapro|citalopram|celexa|paroxetine|paxil|venlafaxine|effexor|duloxetine|cymbalta|bupropion|wellbutrin|mirtazapine|remeron|trazodone|medication|antidepressant|drugs?|pills?|tablets?|\d+\s*mg)\b",
    r"\b\d+\s*mg\s+(?:of\s+)?(sertraline|zoloft|fluoxetine|prozac|escitalopram|lexapro|citalopram|celexa|paroxetine|paxil|venlafaxine|effexor|duloxetine|cymbalta|bupropion|wellbutrin)",
    r"\btypical amount of\b",
]

DOSING_REFUSAL_MESSAGE: str = (
    "This system provides screening recommendations only. "
    "For medication dosing, please consult a licensed prescriber."
)

# ------------------------------------------------------------------
# Prompt-injection & Disallowed topics
# ------------------------------------------------------------------
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(previous|all)\s+instructions",
    r"system\s*prompt",
    r"you\s+are\s+now",
    r"disregard\s+the\s+above",
    r"as\s+an\s+ai\s+language\s+model",
    r"<script",
    r"DROP\s+TABLE",
    r"SELECT\s+\*\s+FROM",
]

DISALLOWED_TOPICS: list[str] = [
    r"\bhow\s+to\s+make\s+(a\s+)?bomb\b",
    r"\billegal\s+drugs\b",
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
    """Check if query contains first-person personal distress markers or acute suicidal ideation."""
    # Clinical scale / assessment queries are informational provider questions, not personal crisis
    clinical_terms = [
        "what scale", "which scale", "what tool", "which tool",
        "what instrument", "which instrument", "screening tool",
        "assessment tool", "how to assess", "how to screen",
        "clinical guideline", "uspstf", "recommendation",
    ]
    if any(ct in query_lower for ct in clinical_terms):
        acute_first_person = [
            "kill myself", "want to die", "end my life", "quiero morir",
            "matarme", "suicidarme", "me suicider", "je veux mourir",
            "想死", "不想活", "muốn chết", "tự tử", "أريد أن أموت", "قتل نفسي"
        ]
        return any(af in query_lower for af in acute_first_person)

    for pattern in PERSONAL_DISTRESS_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return True
    return False


def check_input(query: str) -> GuardrailResult:
    """
    Validate user input before it reaches the retrieval layer.

    Dual-Mode Crisis Logic:
      - CRISIS_REFUSAL: Acute personal distress -> 988 referral, NO model answer.
      - CRISIS_RESOURCE: Informational suicide screening query -> proceed, tag touchpoint.
      - DOSING refusal: Medication / dosage queries -> out-of-scope refusal.
      - Prompt injection / disallowed topic blocking.
    """
    query_lower = query.lower()

    # ── Gate 1: CRISIS detection (highest priority) ───────────────
    has_crisis_keyword = any(kw in query_lower for kw in CRISIS_KEYWORDS)
    is_distress = _is_personal_crisis(query_lower)
    has_clinical_context = any(
        term in query_lower
        for term in [
            "tool", "instrument", "scale", "screener", "screening",
            "questionnaire", "protocol", "recommendation", "guideline",
            "assess", "assessment", "uspstf"
        ]
    )

    if is_distress or has_crisis_keyword:
        if is_distress or not has_clinical_context:
            logger.warning("CRISIS_REFUSAL triggered for acute crisis query: %s", query)
            return GuardrailResult(
                passed=False,
                status="CRISIS",
                reason="Personal crisis / suicidal distress detected.",
                message=CRISIS_MESSAGE,
                flags=["crisis_refusal"],
            )
        else:
            # Purely informational / clinical query about suicide screening / tools
            logger.info("CRISIS_RESOURCE flagged for informational query: %s", query)
            logger.info("CRISIS_RESOURCE flagged for informational query: %s", query)

    # ── Gate 2: DOSING / medication refusal ───────────────────────
    is_dosing_query = False
    for pat in DOSING_PATTERNS:
        if re.search(pat, query_lower, re.IGNORECASE):
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
        if re.search(pattern, query_lower, re.IGNORECASE):
            flags.append(f"injection:{pattern}")

    for pattern in DISALLOWED_TOPICS:
        if re.search(pattern, query_lower, re.IGNORECASE):
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


def check_response_schema(response: str) -> dict[str, Any]:
    """
    Verify that the response contains all 6 required markdown sections.
    """
    present: list[str] = []
    missing: list[str] = []

    for section in REQUIRED_SECTIONS:
        header_name = section.replace("## ", "").lower()
        if re.search(rf"^##\s+{re.escape(header_name)}", response, re.IGNORECASE | re.MULTILINE):
            present.append(section)
        elif section.lower() in response.lower():
            present.append(section)
        elif header_name == "screening tool" and "## screening tools" in response.lower():
            present.append(section)
        elif header_name == "harms & considerations" and ("## harms" in response.lower() or "## harms and considerations" in response.lower()):
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
KNOWN_SECTION_HEADINGS: set[str] = {
    "recommendation",
    "recommendations",
    "population",
    "populations",
    "screening tool",
    "screening tools",
    "harms & considerations",
    "harms and considerations",
    "harms",
    "evidence",
    "source",
    "sources",
    "references",
    "bibliography",
    "metadata",
    "table",
    "tables",
    "clinical considerations",
    "practice considerations",
    "recommendations of others",
    "general",
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusions",
    "summary",
    "table of contents",
    "list of tables",
    "list of figures",
}


def is_invalid_or_metadata_quote(quote: str) -> tuple[bool, str]:
    """
    Check if a quote is a metadata fragment, table line, bibliography, document title,
    section heading, or too short (<25 alphanumeric chars).
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

    # (d) Section heading rejection (Defects 1b & 10)
    norm_q = re.sub(r"[^\w\s]", "", stripped).lower().strip()
    if norm_q in KNOWN_SECTION_HEADINGS or stripped.lower() in KNOWN_SECTION_HEADINGS:
        return True, f"Matches section heading '{stripped}'"

    # Document title rejection (Defects 1b & 10)
    doc_titles = [v.lower() for v in SOURCE_DISPLAY_NAMES.values()] + [k.lower() for k in SOURCE_DISPLAY_NAMES.keys()]
    if stripped.lower() in doc_titles or norm_q in [re.sub(r"[^\w\s]", "", dt).lower().strip() for dt in doc_titles]:
        return True, f"Matches document title '{stripped}'"

    # (e) Strip trailing numerals/citations before length check (Defect 1c)
    cleaned_quote = re.sub(r"[\s\d]+$", "", stripped)

    # (f) Has <25 alphanumeric characters (Defect 1a: changed to <25)
    alpha_chars = re.sub(r"[^a-zA-Z0-9]", "", cleaned_quote)
    if len(alpha_chars) < 25:
        return True, f"Too short ({len(alpha_chars)} alphanumeric chars < 25)"

    return False, ""


def _normalize_citation_text(text: str) -> str:
    """Normalize text for robust whitespace-independent citation matching (Defect 9)."""
    # Remove zero-width characters
    t = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)
    # Replace non-breaking spaces and multi-whitespace with single space
    t = re.sub(r"[\s\u00A0]+", " ", t)
    return t.strip().lower()


def verify_citations(
    llm_response: str,
    context_chunks: list[dict],
) -> dict:
    """
    Extract and verify every verbatim citation quote in the LLM response.

    - Bracket normalization: accepts `[Doc:`, `【Doc:`, `「Doc:` (Defect 5).
    - Rejects metadata fragments, table headers, PMIDs, or short fragments (<25 chars).
    - Ensures that every valid quote appears verbatim in at least one retrieved context chunk.
    """
    # Defect 5: Bracket normalization
    if "【Doc:" in llm_response or "「Doc:" in llm_response:
        logger.warning("verify_citations: non-standard bracket detected in response (normalized 【/「 to [)")
    norm_llm_response = re.sub(r"(?:\[|【|「)Doc:", "[Doc:", llm_response)

    quote_pattern = re.compile(r'Quote:\s*"([^"]+)"', re.IGNORECASE)
    quotes = quote_pattern.findall(norm_llm_response)

    if not quotes:
        return {
            "status": "CITATION_VERIFICATION_FAILED",
            "verified_quotes": 0,
            "total_quotes": 0,
            "unverified_quotes": ["No citations found in response."],
            "detail": "Response contains zero citation quotes.",
        }

    corpus_texts = [_normalize_citation_text(c.get("text", "")) for c in context_chunks]

    verified: list[str] = []
    unverified: list[str] = []

    for quote in quotes:
        # Step 1: Reject metadata-only or short quotes
        is_invalid, reason = is_invalid_or_metadata_quote(quote)
        if is_invalid:
            unverified.append(f"{quote} (REJECTED: {reason})")
            continue

        # Step 2: Check normalized verbatim match in context corpus (Defect 9)
        norm_quote = _normalize_citation_text(quote)
        norm_quote_clean = re.sub(r"\s*\d+$", "", norm_quote).strip()

        found = False
        for corpus in corpus_texts:
            if norm_quote in corpus or norm_quote_clean in corpus:
                found = True
                break
            # Token split / whitespace-stripped fallback (e.g. In2016,theUS...)
            alpha_quote = re.sub(r"[^a-z0-9]", "", norm_quote_clean)
            alpha_corpus = re.sub(r"[^a-z0-9]", "", corpus)
            if len(alpha_quote) >= 20 and alpha_quote in alpha_corpus:
                found = True
                break

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
      - Per-claim grounding check (Defect 4)
      - Claim-citation relevance check (Defect 8)
    """
    flags: list[str] = []
    resp_lower = response.lower()
    q_lower = query.lower()

    norm_response = re.sub(r"(?:\[|【|「)Doc:", "[Doc:", response)

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
    if "suicide" in q_lower:
        if "phq-9 is a suicide" in resp_lower or "epds is a suicide" in resp_lower:
            flags.append("INSTRUMENT_CONFLATED")

    # 5. Per-claim grounding check (Defect 4)
    # Check that every substantive sentence >60 chars outside Evidence/Source has [Doc: within 100 chars
    sections = re.split(r"(?=^##\s+)", norm_response, flags=re.MULTILINE)
    for sec_block in sections:
        sec_lower = sec_block.lower()
        if sec_lower.startswith("## evidence") or sec_lower.startswith("## source"):
            continue

        lines = [l.strip() for l in sec_block.split("\n") if l.strip() and not l.strip().startswith("#")]
        for line in lines:
            clean_line = re.sub(r"^[*\-\d\.\s]+", "", re.sub(r"\[Doc:[^\]]+\]", "", line)).strip()
            if len(clean_line) > 60 and not clean_line.startswith("⚠️") and "988" not in clean_line and "not a substitute for professional" not in clean_line.lower():
                line_idx = sec_block.find(line)
                window = sec_block[max(0, line_idx - 100):min(len(sec_block), line_idx + len(line) + 100)]
                if "[Doc:" not in window and "[doc:" not in window.lower():
                    flags.append("UNGROUNDED_CLAIM")
                    break

    # 6. Claim-citation relevance check (Defect 8)
    pop_section_match = re.search(r"## Population\s*\n(.*?)(?=^##|\Z)", norm_response, re.DOTALL | re.MULTILINE)
    if pop_section_match:
        pop_text = pop_section_match.group(1)
        pop_quotes = re.findall(r'Quote:\s*"([^"]+)"', pop_text, re.IGNORECASE)
        for pq in pop_quotes:
            for c in context_chunks:
                c_text = c.get("text", "").lower()
                c_sec = c.get("section_name", "").lower()
                if pq.lower() in c_text or re.sub(r"[^\w\s]", "", pq.lower()) in re.sub(r"[^\w\s]", "", c_text):
                    if "frequency" in c_sec or "screening interval" in c_sec:
                        flags.append("MISMATCHED_CITATION")
                        break

    passed = len(flags) == 0
    return {
        "passed": passed,
        "flags": flags,
        "detail": f"Faithfulness checks {'passed' if passed else 'failed: ' + ', '.join(flags)}",
    }
