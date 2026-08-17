"""
LLM Client — Google Gemini powered generation with Groq fallback and robust mock fallback.

Uses Google Gemini (gemini-3.6-flash / gemini-3.7-flash / gemini-flash-latest) via google.generativeai as primary LLM.
Falls back to Groq (allam-2-7b / groq/compound) when Gemini is unavailable or rate-limited,
and to a deterministic MOCK response when all endpoints are unreachable.

Every generation attempt is logged to logs/generation.log.

Usage:
    from src.generation.llm_client import LLMClient
    client = LLMClient()
    response = client.generate(prompt_text, system_prompt=system_prompt)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from configs.settings import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_FALLBACK_MODEL,
    LLM_FALLBACK_PROVIDER,
    GEMINI_FALLBACK_MODELS,
    LLM_TIMEOUT_SEC,
    GROQ_BASE_URL,
    TEMPERATURE,
    GENERATION_LOG_PATH,
    get_source_display_name,
)

logger = logging.getLogger(__name__)

# Load .env from project root or src/safety/.env
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(_project_root / "src" / "safety" / ".env")


def _extract_gemini_text(resp: Any) -> str:
    """Safely extract text from Gemini response even if finish_reason truncated or parts nested."""
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text
    except Exception:
        pass

    if hasattr(resp, "candidates") and resp.candidates:
        cand = resp.candidates[0]
        if hasattr(cand, "content") and hasattr(cand.content, "parts"):
            parts_text = []
            for part in cand.content.parts:
                if hasattr(part, "text") and part.text:
                    parts_text.append(part.text)
            if parts_text:
                return "".join(parts_text)
    return ""


def _is_degenerate(text: str) -> tuple[bool, str]:
    """Check for degenerate repetition via 3-grams."""
    words = text.split()
    if len(words) < 10:
        return False, ""
    trigrams = [" ".join(words[i:i+3]).lower() for i in range(len(words)-2)]
    if not trigrams:
        return False, ""
    from collections import Counter
    counts = Counter(trigrams)
    max_freq = max(counts.values())
    unique_ratio = len(counts) / len(trigrams)
    if max_freq > 10 or len(counts) < 0.3 * len(words):
        most_common = counts.most_common(1)[0]
        return True, f"'{most_common[0]}' repeated {most_common[1]} times. Ratio: {unique_ratio:.2f}"
    return False, ""


def _extract_clean_verbatim_quote(chunk: dict[str, Any]) -> tuple[str, str, str, str]:
    """
    Extract a valid, non-table, non-metadata verbatim quote >= 40 chars from a chunk.
    """
    raw_doc = chunk.get("document_name", "USPSTF Guidelines")
    doc = get_source_display_name(raw_doc)
    sec = chunk.get("section_name", "General")
    page = str(chunk.get("start_page", "1"))
    text = chunk.get("text", "")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    for s in sentences:
        s_clean = s.strip()
        alpha = re.sub(r"[^a-zA-Z0-9]", "", s_clean)
        if (
            len(alpha) >= 40
            and not s_clean.startswith("|")
            and not s_clean.startswith("---")
            and "PMID:" not in s_clean
            and "et al." not in s_clean
            and not re.search(r"^\s*Table\s+\d+\.", s_clean, re.IGNORECASE)
        ):
            words = s_clean.split()
            if len(words) >= 8:
                return doc, sec, page, " ".join(words[:min(len(words), 20)])

    # Fallback to substantive slice
    clean_lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("|") and "---" not in l and "PMID:" not in l]
    for line in clean_lines:
        alpha = re.sub(r"[^a-zA-Z0-9]", "", line)
        if len(alpha) >= 40:
            words = line.split()
            return doc, sec, page, " ".join(words[:min(len(words), 20)])

    return doc, sec, page, "The USPSTF recommends screening for depression in the general adult population"


def _build_mock_response(
    context_chunks: list[dict[str, Any]] | None = None,
    prompt: str = "",
) -> str:
    """
    Build a deterministic, unhallucinated MOCK response when LLM is unreachable.
    Contains ZERO clinical claims (no SMD, effect sizes, drug names, or specific citations).
    """
    return (
        "MOCK FALLBACK MODE: LLM endpoint unreachable. This system is operating in degraded mode "
        "and cannot generate clinical claims. Please retry when connectivity is restored.\n\n"
        "⚠️ If you or someone you know is struggling or in crisis, help is available. "
        "Call or text 988 (US) or contact your local emergency services for immediate, confidential 24/7 support.\n\n"
        "This information is based on USPSTF guidance current as of June 2023 and is for clinical "
        "decision support only. It is not a substitute for professional medical judgment. "
        "Always verify current guidelines and consult appropriate specialists for individual patient care."
    )


class LLMClient:
    """
    Multi-provider LLM client with Google Gemini as primary, Groq as fallback, and Mock.

    Parameters:
        provider: "gemini", "groq", or "mock" (defaults to settings.LLM_PROVIDER).
        model: Model name string (defaults to settings.LLM_MODEL).
        fallback_model: Model name string for fallback.
        base_url: OpenAI-compatible API base URL for Groq.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        timeout: Request timeout in seconds.
        api_key: Optional explicit API key.
    """

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model: str = LLM_MODEL,
        fallback_model: str = LLM_FALLBACK_MODEL,
        fallback_provider: str = LLM_FALLBACK_PROVIDER,
        base_url: str = GROQ_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = 1500,
        timeout: int = LLM_TIMEOUT_SEC,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider.lower() if provider else "gemini"
        self.model = model
        self.fallback_model = fallback_model
        self.fallback_provider = fallback_provider
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Resolve API keys
        self.gemini_api_key = api_key if self.provider == "gemini" else None
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.groq_api_key = api_key if self.provider == "groq" else None
        if not self.groq_api_key:
            self.groq_api_key = os.getenv("GROQ_API_KEY")

        if not self.gemini_api_key or not self.groq_api_key:
            load_dotenv(_project_root / "src" / "safety" / ".env")
            load_dotenv(_project_root / ".env")
            if not self.gemini_api_key:
                self.gemini_api_key = os.getenv("GEMINI_API_KEY")
            if not self.groq_api_key:
                self.groq_api_key = os.getenv("GROQ_API_KEY")

        self._gemini_configured = False
        self._groq_client: Any | None = None

        # Setup Gemini
        if self.gemini_api_key and self.gemini_api_key.strip() not in ("", "invalid", "your_gemini_api_key_here"):
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self._gemini_configured = True
            except Exception as e:
                logger.warning("Failed to configure Google Gemini SDK: %s", e)

        # Setup Groq client
        if self.groq_api_key and self.groq_api_key.strip() not in ("", "invalid", "your_groq_api_key_here"):
            try:
                from openai import OpenAI
                self._groq_client = OpenAI(
                    api_key=self.groq_api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except Exception as e:
                logger.warning("Failed to initialize OpenAI client for Groq: %s", e)

        # Perform initial health check
        self.health_check()

    def health_check(self) -> bool:
        """
        Check endpoint reachability. If primary Gemini fails, tests fallback models or switches to Groq.
        """
        if self.provider == "gemini" and self._gemini_configured:
            import google.generativeai as genai
            candidate_models = [
                self.model,
                "gemini-3.6-flash",
                "gemini-3.7-flash",
                "gemini-flash-latest",
                "gemini-3.5-flash",
                "gemini-2.5-flash",
            ]
            for m in candidate_models:
                if not m:
                    continue
                try:
                    g_model = genai.GenerativeModel(m)
                    resp = g_model.generate_content(
                        "ping",
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.0,
                            max_output_tokens=50,
                        ),
                    )
                    text = _extract_gemini_text(resp)
                    if text:
                        if self.model != m:
                            logger.info("Primary Gemini model switched to '%s'", m)
                            self.model = m
                        return True
                except Exception as exc:
                    logger.warning("Gemini health check failed for model '%s': %s", m, exc)

        # If Gemini is not configured or failed, check Groq fallback
        if self._groq_client:
            candidate_groq = [
                self.fallback_model,
                "allam-2-7b",
                "groq/compound",
                "groq/compound-mini",
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.6-27b",
            ]
            for m in candidate_groq:
                if not m:
                    continue
                try:
                    resp = self._groq_client.chat.completions.create(
                        model=m,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=20,
                        timeout=min(self.timeout, 10),
                    )
                    if resp and resp.choices:
                        logger.info("Groq fallback healthy on model '%s'", m)
                        return True
                except Exception as exc:
                    logger.warning("Groq health check failed for model '%s': %s", m, exc)

        return False

    def generate(
        self,
        prompt: str,
        context_chunks: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response using Google Gemini as primary, cascading to Groq and Mock.
        """
        t0 = time.perf_counter()
        
        is_deg_prompt, deg_reason_prompt = _is_degenerate(prompt)
        if is_deg_prompt:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            result = {
                "response": "Prompt is degenerate.",
                "provider": "input",
                "model": "input",
                "endpoint_status": "success",
                "status": "REFUSAL_QUALITY_FAILED",
                "reason": "Degenerate repetition detected",
                "latency_ms": latency_ms,
            }
            logger.warning("Degenerate repetition detected in prompt: %s.", deg_reason_prompt)
            self._log_generation(prompt, result)
            return result

        # ── 1. PRIMARY: GOOGLE GEMINI ───────────────────────────────────
        if self.provider == "gemini" and self._gemini_configured:
            try:
                import google.generativeai as genai
                candidate_gemini = [
                    self.model,
                    "gemini-3.6-flash",
                    "gemini-3.7-flash",
                    "gemini-flash-latest",
                    "gemini-3.5-flash",
                    "gemini-2.5-flash",
                ]
                for m in candidate_gemini:
                    if not m:
                        continue
                    try:
                        g_model = genai.GenerativeModel(
                            m,
                            system_instruction=system_prompt if system_prompt else None,
                        )
                        gen_config = genai.types.GenerationConfig(
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        )
                        resp = g_model.generate_content(prompt, generation_config=gen_config)
                        raw_c = _extract_gemini_text(resp)
                        c = re.sub(r"<think>.*?</think>", "", raw_c, flags=re.DOTALL).strip()
                        if not c and raw_c:
                            c = raw_c.strip()

                        # Strip chain-of-thought preamble if any
                        cot_markers = [
                            "Here's a thinking process",
                            "Here is a thinking process",
                            "Let me analyze",
                            "Step-by-step reasoning",
                            "Thinking Process:",
                            "Thinking process:",
                            "Let's think step by step",
                        ]
                        has_cot = any(marker in c for marker in cot_markers)
                        if (has_cot or not c.startswith("##")) and "##" in c:
                            idx = c.find("##")
                            c = c[idx:].strip()

                        # Accept response if it contains markdown headers and is non-empty (>= 150 chars)
                        if c and len(c) >= 150 and "##" in c:
                            is_deg, deg_reason = _is_degenerate(c)
                            if is_deg:
                                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                                result = {
                                    "response": c,
                                    "provider": "gemini",
                                    "model": m,
                                    "model_used": m,
                                    "endpoint_status": "success" if m == self.model else "fallback",
                                    "status": "REFUSAL_QUALITY_FAILED",
                                    "reason": "Degenerate repetition detected",
                                    "latency_ms": latency_ms,
                                }
                                logger.warning("Degenerate repetition detected: %s. Response len: %d", deg_reason, len(c))
                                self._log_generation(prompt, result)
                                return result
                                
                            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                            endpoint_status = "success" if m == self.model else "fallback"
                            result = {
                                "response": c,
                                "provider": "gemini",
                                "model": m,
                                "model_used": m,
                                "endpoint_status": endpoint_status,
                                "status": "real",
                                "latency_ms": latency_ms,
                            }
                            self._log_generation(prompt, result)
                            return result
                        elif c:
                            logger.warning("Gemini output with model '%s' too short (%d chars). Trying next...", m, len(c))
                    except Exception as g_err:
                        logger.warning("Gemini generation attempt with '%s' failed: %s", m, g_err)
                        continue
            except Exception as exc:
                logger.warning("Gemini API error, attempting fallback to Groq: %s", exc)

        # ── 2. SECONDARY: GROQ LLM FALLBACK ─────────────────────────────
        if self._groq_client:
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                candidate_groq = [
                    self.fallback_model,
                    "allam-2-7b",
                    "groq/compound",
                    "groq/compound-mini",
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "qwen/qwen3.6-27b",
                ]
                for fb in candidate_groq:
                    if not fb:
                        continue
                    try:
                        resp = self._groq_client.chat.completions.create(
                            model=fb,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        raw_c = resp.choices[0].message.content or ""
                        c = re.sub(r"<think>.*?</think>", "", raw_c, flags=re.DOTALL).strip()
                        if not c and raw_c:
                            c = raw_c.strip()

                        cot_markers = [
                            "Here's a thinking process",
                            "Here is a thinking process",
                            "Let me analyze",
                            "Step-by-step reasoning",
                            "Thinking Process:",
                            "Thinking process:",
                            "Let's think step by step",
                        ]
                        has_cot = any(marker in c for marker in cot_markers)
                        if (has_cot or not c.startswith("##")) and "##" in c:
                            idx = c.find("##")
                            c = c[idx:].strip()

                        if c and len(c) >= 200:
                            is_deg, deg_reason = _is_degenerate(c)
                            if is_deg:
                                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                                result = {
                                    "response": c,
                                    "provider": "groq",
                                    "model": fb,
                                    "model_used": fb,
                                    "endpoint_status": "fallback",
                                    "status": "REFUSAL_QUALITY_FAILED",
                                    "reason": "Degenerate repetition detected",
                                    "latency_ms": latency_ms,
                                }
                                logger.warning("Degenerate repetition detected: %s. Response len: %d", deg_reason, len(c))
                                self._log_generation(prompt, result)
                                return result

                            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                            result = {
                                "response": c,
                                "provider": "groq",
                                "model": fb,
                                "model_used": fb,
                                "endpoint_status": "fallback",
                                "status": "real",
                                "latency_ms": latency_ms,
                            }
                            self._log_generation(prompt, result)
                            return result
                    except Exception as fb_err:
                        if "429" in str(fb_err) or "rate_limit" in str(fb_err):
                            time.sleep(1.0)
                        continue
            except Exception as exc:
                logger.warning("Groq fallback failed: %s", exc)

        # ── 3. TERTIARY: MOCK DEGRADED FALLBACK ──────────────────────────
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        mock_content = _build_mock_response(context_chunks, prompt=prompt)
        result = {
            "response": mock_content,
            "provider": "mock",
            "model": "mock-fallback",
            "model_used": "mock-fallback",
            "endpoint_status": "unreachable",
            "status": "error",
            "latency_ms": latency_ms,
        }
        self._log_generation(prompt, result)
        return result

    def _log_generation(self, prompt: str, result: dict[str, Any]) -> None:
        """Append a structured JSON line entry to logs/generation.log."""
        try:
            entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "provider": result.get("provider"),
                "model": result.get("model"),
                "model_used": result.get("model_used", result.get("model")),
                "endpoint_status": result.get("endpoint_status", "unknown"),
                "status": result.get("status"),
                "latency_ms": result.get("latency_ms"),
                "prompt_length": len(prompt),
                "response_length": len(result.get("response", "")),
            }
            if "error" in result:
                entry["error"] = result["error"]

            log_path = Path(GENERATION_LOG_PATH)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.debug("Failed to write generation log: %s", exc)
