"""
LLM Client — Groq-powered generation with OpenAI SDK and robust mock fallback.

Uses OpenAI SDK pointed at Groq's API endpoint for llama-3.3-70b-versatile.
Falls back to a deterministic MOCK response (valid 6-section format + verbatim quote)
when the API key is missing, provider is "mock", or an API error occurs.

Every generation attempt is logged to logs/generation.log.

Usage:
    from src.generation.llm_client import LLMClient
    client = LLMClient()
    response = client.generate(prompt_text)
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
    Groq-powered LLM client with automatic mock fallback.

    Generation client supporting Groq-hosted LLMs with graceful mock fallback.

    Parameters:
        provider: "groq" or "mock" (defaults to settings.LLM_PROVIDER).
        model: Model name string (defaults to settings.LLM_MODEL).
        base_url: OpenAI-compatible API base URL (defaults to settings.GROQ_BASE_URL).
        temperature: Sampling temperature (defaults to settings.TEMPERATURE).
        max_tokens: Maximum tokens to generate (default 4096).
    """

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        model: str = LLM_MODEL,
        fallback_model: str = LLM_FALLBACK_MODEL,
        base_url: str = GROQ_BASE_URL,
        temperature: float = TEMPERATURE,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.fallback_model = fallback_model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Resolve API key
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            # Check secondary safety env
            load_dotenv("src/safety/.env")
            self.api_key = os.getenv("GROQ_API_KEY")

        self._client: Any | None = None
        if self.provider == "groq":
            if not self.api_key or self.api_key.strip() in ("", "your_groq_api_key_here", "invalid"):
                logger.warning(
                    "GROQ_API_KEY is missing or invalid. LLMClient will operate in MOCK mode."
                )
                self.provider = "mock"
            else:
                try:
                    from openai import OpenAI  # inline to avoid hard failure if missing
                    self._client = OpenAI(
                        api_key=self.api_key,
                        base_url=self.base_url,
                    )
                    logger.info("LLMClient initialized with Groq model '%s'", self.model)
                except Exception as exc:
                    logger.warning("Failed to initialize OpenAI client for Groq: %s. Using MOCK.", exc)
                    self.provider = "mock"

    def generate(
        self,
        prompt: str,
        context_chunks: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response for the given prompt using Groq or mock fallback.

        Returns dict with keys: 'response' (str), 'provider' (str), 'model' (str),
            'status' ('real'|'mock'|'error'), 'latency_ms' (float).
        """
        t0 = time.perf_counter()

        if self._client and self.provider != "mock":
            try:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                model_to_try = self.model
                response = None
                used_model = self.model
                try:
                    resp = self._client.chat.completions.create(
                        model=model_to_try,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    raw_c = resp.choices[0].message.content or ""
                    c = re.sub(r"<think>.*?</think>", "", raw_c, flags=re.DOTALL).strip()
                    if c or raw_c.strip():
                        response = resp
                        used_model = model_to_try
                except Exception:
                    response = None

                if response is None:
                    fallback_models = [self.fallback_model, "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b", "groq/compound-mini", "groq/compound"]
                    for fb in fallback_models:
                        if fb == model_to_try or not fb:
                            continue
                        try:
                            resp = self._client.chat.completions.create(
                                model=fb,
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                            )
                            raw_c = resp.choices[0].message.content or ""
                            c = re.sub(r"<think>.*?</think>", "", raw_c, flags=re.DOTALL).strip()
                            if not c and raw_c:
                                c = raw_c.strip()
                            if c:
                                response = resp
                                used_model = fb
                                break
                        except Exception:
                            continue
                if response is None:
                    raise RuntimeError("All configured LLM models failed or rate limited")

                raw_content = response.choices[0].message.content or ""
                # Strip internal reasoning/thinking blocks from reasoning models (e.g. Qwen/DeepSeek/GPT-OSS)
                content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                if not content and raw_content:
                    content = raw_content.strip()

                # Fix #2: Strip chain-of-thought / scratchpad
                cot_markers = [
                    "Here's a thinking process",
                    "Here is a thinking process",
                    "Let me analyze",
                    "Step-by-step reasoning",
                    "Thinking Process:",
                    "Thinking process:",
                    "Let's think step by step",
                ]
                has_cot = any(marker in content for marker in cot_markers)
                if has_cot or not content.startswith("##"):
                    if "##" in content:
                        idx = content.find("##")
                        content = content[idx:].strip()

                if not content and raw_content:
                    content = raw_content.strip()

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                endpoint_status = "success" if used_model == model_to_try else "fallback"

                result = {
                    "response": content,
                    "provider": "groq",
                    "model": used_model,
                    "model_used": used_model,
                    "endpoint_status": endpoint_status,
                    "status": "real",
                    "latency_ms": latency_ms,
                }
                self._log_generation(prompt, result)
                return result

            except Exception as exc:
                logger.warning("Groq API error, falling back to MOCK: %s", exc)
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                mock_content = _build_mock_response(context_chunks, prompt=prompt)
                result = {
                    "response": mock_content,
                    "provider": "mock",
                    "model": "mock-fallback",
                    "model_used": "mock-fallback",
                    "endpoint_status": "unreachable",
                    "status": "error",
                    "error": str(exc),
                    "latency_ms": latency_ms,
                }
                self._log_generation(prompt, result)
                return result
        else:
            # MOCK mode
            mock_content = _build_mock_response(context_chunks, prompt=prompt)
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            result = {
                "response": mock_content,
                "provider": "mock",
                "model": "mock-fallback",
                "model_used": "mock-fallback",
                "endpoint_status": "unreachable",
                "status": "mock",
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
