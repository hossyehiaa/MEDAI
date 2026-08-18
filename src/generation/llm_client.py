"""
LLM Client — OpenRouter paid cascade: DeepSeek V3 → Llama-3.3-70B → Qwen3-235B → Mock.

Uses OpenAI SDK pointed at OpenRouter with proper headers.
Every generation attempt is logged to logs/generation.log.

Usage:
    from src.generation.llm_client import LLMClient
    client = LLMClient()
    response = client.generate(prompt_text, system_prompt=system_prompt)
"""

from __future__ import annotations

import hashlib
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
    LLM_FALLBACK_MODELS,
    LLM_TIMEOUT_SEC,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_BASE_URL,
    TEMPERATURE,
    GENERATION_LOG_PATH,
    get_source_display_name,
)

logger = logging.getLogger(__name__)

# Load .env from project root (auto-create from .env.example if missing)
_project_root = Path(__file__).resolve().parent.parent.parent
_env_path = _project_root / ".env"
if not _env_path.exists():
    _example_path = _project_root / ".env.example"
    if _example_path.exists():
        import shutil
        shutil.copy2(_example_path, _env_path)
        logger.info("Created .env from .env.example — please add your OPENROUTER_API_KEY.")
load_dotenv(_env_path)


# ── OpenRouter Headers ───────────────────────────────────────────────
_OR_HEADERS = {
    "HTTP-Referer": "https://medai.local",
    "X-Title": "medAI Clinical RAG",
}


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
        "If you or someone you know is struggling or in crisis, help is available. "
        "Call or text 988 (US) or contact your local emergency services for immediate, confidential 24/7 support.\n\n"
        "This information is based on USPSTF guidance current as of June 2023 and is for clinical "
        "decision support only. It is not a substitute for professional medical judgment. "
        "Always verify current guidelines and consult appropriate specialists for individual patient care."
    )


def _clean_response(raw: str) -> str:
    """Strip think blocks, chain-of-thought preambles, and normalize."""
    # Strip <think>...</think>
    c = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not c and raw:
        c = raw.strip()

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

    return c


# Response cache (simple dict — sufficient for single-process)
_response_cache: dict[str, dict[str, Any]] = {}


class LLMClient:
    """
    OpenRouter paid LLM client with model cascade and mock fallback.

    Cascade: deepseek/deepseek-chat-v3-0324 → llama-3.3-70b-instruct → qwen3-235b-a22b → mock.
    """

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model: str = LLM_MODEL,
        fallback_models: list[str] = LLM_FALLBACK_MODELS,
        base_url: str = OPENROUTER_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_OUTPUT_TOKENS,
        timeout: int = LLM_TIMEOUT_SEC,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider.lower() if provider else "openrouter"
        self.model = model
        self.fallback_models = fallback_models
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Resolve API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            load_dotenv(_project_root / ".env")
            self.api_key = os.getenv("OPENROUTER_API_KEY")

        # Setup OpenRouter client via OpenAI SDK
        self._client: Any | None = None
        if self.api_key and self.api_key.strip() not in ("", "invalid", "your_openrouter_api_key_here"):
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                    default_headers=_OR_HEADERS,
                )
                logger.info("OpenRouter client initialized (model=%s)", self.model)
            except Exception as e:
                logger.warning("Failed to initialize OpenRouter client: %s", e)

        # Perform initial health check
        self._healthy = self.health_check()

    def health_check(self) -> bool:
        """Check OpenRouter endpoint reachability."""
        if not self._client:
            logger.warning("No OpenRouter client available for health check.")
            return False

        candidate_models = [self.model] + self.fallback_models
        for m in candidate_models:
            if not m:
                continue
            try:
                resp = self._client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=20,
                    timeout=min(self.timeout, 15),
                )
                if resp and resp.choices:
                    logger.info("OpenRouter health check passed on model '%s'", m)
                    return True
            except Exception as exc:
                logger.warning("OpenRouter health check failed for model '%s': %s", m, exc)

        return False

    def generate(
        self,
        prompt: str,
        context_chunks: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Generate a response using OpenRouter cascade: primary → fallbacks → mock."""
        t0 = time.perf_counter()

        # ── Degenerate prompt check ────────────────────────────────────────
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

        # ── Response cache check ─────────────────────────────────────────
        cache_key = hashlib.md5(prompt.encode()).hexdigest() if len(prompt) > 200 else prompt[:200]
        if cache_key in _response_cache:
            cached = _response_cache[cache_key]
            logger.info("Response cache hit (key=%s...)", cache_key[:16])
            return cached

        # ── OpenRouter cascade ───────────────────────────────────────────
        if self._client:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            candidate_models = [self.model] + self.fallback_models
            for idx, m in enumerate(candidate_models):
                if not m:
                    continue
                try:
                    resp = self._client.chat.completions.create(
                        model=m,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    raw_c = resp.choices[0].message.content or ""
                    c = _clean_response(raw_c)

                    # Accept response if non-empty and substantive (>= 150 chars)
                    if c and len(c) >= 150:
                        is_deg, deg_reason = _is_degenerate(c)
                        if is_deg:
                            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                            result = {
                                "response": c,
                                "provider": "openrouter",
                                "model": m,
                                "model_used": m,
                                "endpoint_status": "fallback" if idx > 0 else "success",
                                "status": "REFUSAL_QUALITY_FAILED",
                                "reason": "Degenerate repetition detected",
                                "latency_ms": latency_ms,
                            }
                            logger.warning("Degenerate repetition detected: %s. Response len: %d", deg_reason, len(c))
                            self._log_generation(prompt, result)
                            return result

                        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                        endpoint_status = "success" if idx == 0 else "fallback"
                        result = {
                            "response": c,
                            "provider": "openrouter",
                            "model": m,
                            "model_used": m,
                            "endpoint_status": endpoint_status,
                            "status": "real",
                            "latency_ms": latency_ms,
                        }
                        self._log_generation(prompt, result)
                        _response_cache[cache_key] = result
                        return result
                    elif c:
                        logger.warning("OpenRouter output with model '%s' too short (%d chars). Trying next...", m, len(c))
                except Exception as err:
                    err_str = str(err)
                    if "429" in err_str or "rate_limit" in err_str:
                        logger.warning("Rate limited on model '%s', waiting 2s then cascading...", m)
                        time.sleep(2.0)
                    else:
                        logger.warning("OpenRouter generation with '%s' failed: %s", m, err)
                    continue

        # ── MOCK DEGRADED FALLBACK ─────────────────────────────────────
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
