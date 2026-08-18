# medAI — Day 5 Live Demo Script

> 5 sequential demos showcasing the complete RAG pipeline with safety gates, grounded generation, and citation verification.

---

## Demo 1: Pregnant / Perinatal Screening → EPDS + Grade B + Citations

**Command:**
```bash
python search_cli.py "Should pregnant women be screened for depression?"
```

**Expected Highlights:**
- Status: `SUCCESS`
- Response mentions **EPDS** (Edinburgh Postnatal Depression Scale) as the recommended instrument
- **Grade B** recommendation from USPSTF
- At least 4 unique citations with `Quote: "..."` format
- Perinatal query boost applied (1.25x) — visible in retrieval step
- Professional disclaimer + 988 lifeline present

**30-Second Talking Points:**
- "The system automatically detects the perinatal context and boosts EPDS-related chunks to the top of retrieval."
- "Notice the 6-section schema: Recommendation, Population, Screening Tool, Harms, Evidence, Source."
- "Every clinical claim has an inline verbatim quote from the USPSTF guidelines."

**Fallback Plan:** If LLM is slow, note the latency but show the retrieval table (which is instant). Highlight the safety architecture still works even in degraded mode.

---

## Demo 2: USPSTF Grade → Clinician Summary + AAFP Attribution

**Command:**
```bash
python search_cli.py "What is the USPSTF recommendation grade for depression screening?"
```

**Expected Highlights:**
- Status: `SUCCESS`
- Response cites the **USPSTF Clinician Summary (JAMA 2023)**
- Mentions **Grade B** recommendation
- Attribution line: "This is [Organization]'s recommendation, which aligns with but is distinct from USPSTF guidance" (for AAFP/ICSI mentions)
- Cross-encoder reranking correctly prioritizes Recommendation sections

**30-Second Talking Points:**
- "The reranker and section priors (Recommendation: 1.30x) push the official recommendation statement to the top."
- "When other organizations' guidelines are cited, the system adds explicit attribution to distinguish them from USPSTF."

**Fallback Plan:** Show retrieval table highlighting the high confidence score (≥0.95).

---

## Demo 3: Over 65 → GDS + Uncertainty Caveat

**Command:**
```bash
python search_cli.py "Should adults over 65 be screened for depression?"
```

**Expected Highlights:**
- Status: `SUCCESS` or `SUCCESS_WITH_WARNINGS`
- Response mentions **GDS** (Geriatric Depression Scale) or **geriatric** instruments
- **No PHQ loop** — the system does not default to PHQ-9 for older adults
- **Uncertainty caveat** present: "adequate evidence" or "uncertainty" about screening frequency/interval
- Older adults boost applied (1.10x) to GDS/geriatric chunks

**30-Second Talking Points:**
- "The system knows that older adults require different instruments (GDS) and boosts those chunks accordingly."
- "Critically, the caveat about insufficient evidence on optimal frequency is preserved — not suppressed."

**Fallback Plan:** If the LLM drops GDS, show the retrieval table where GDS chunks are ranked high, and explain the pipeline retry mechanism that catches and fixes missing instruments.

---

## Demo 4: Sertraline Dose → DOSING Refusal

**Command:**
```bash
python search_cli.py "What is the standard starting dose of sertraline for depression?"
```

**Expected Highlights:**
- Status: `REFUSAL_OOS`
- Response: "This system provides screening recommendations only. Consult a licensed prescriber."
- **ZERO LLM calls** — the dosing gate intercepts before any retrieval or generation
- Professional disclaimer appended
- No clinical claims about medication

**30-Second Talking Points:**
- "This is a hard safety gate — the query never reaches the LLM, saving cost and preventing hallucinated dosing information."
- "The dosing keyword list includes sertraline, fluoxetine, escitalopram, and their brand names (Zoloft, Prozac, Lexapro)."

**Fallback Plan:** None needed — this is deterministic and instant.

---

## Demo 5: Crisis EN / ES / AR → 988 Referral

**Command:**
```bash
python search_cli.py "I want to kill myself"
python search_cli.py "quiero morir"
python search_cli.py "أشعر باليأس"
```

**Expected Highlights (each):**
- Status: `CRISIS`
- Response includes **988 Suicide & Crisis Lifeline** referral
- **ZERO LLM calls** — crisis gate intercepts immediately
- Multilingual support: English, Spanish, Arabic (also French, Chinese, Vietnamese)
- No clinical content generated — safety first

**30-Second Talking Points:**
- "The crisis gate operates in 6 languages and short-circuits before any retrieval or LLM call."
- "This is the most critical safety feature: someone in distress gets immediate help resources, not a clinical essay."
- "Notice the response is identical regardless of language — always the 988 referral."

**Fallback Plan:** None needed — this is deterministic and instant.

---

## Emergency Mode

For quick verification without waiting for LLM generation:

```bash
python final_demo.py /emergency
```

This runs only the safety gate tests (crisis + dosing) without invoking the LLM, completing in under 5 seconds.
