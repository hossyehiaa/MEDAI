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

    The client uses the OpenAI SDK pointed at Groq's API endpoint.
    If the API key is missing, the provider is set to 'mock', or any API
    error occurs, a deterministic MOCK response is returned instead.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        provider: str = LLM_PROVIDER,
        temperature: float = TEMPERATURE,
        max_tokens: int = 2048,
    ) -> None:
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.log_path = Path(GENERATION_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._active_model = self.model
        self._client = None
        if self.api_key and self.provider != "mock":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=GROQ_BASE_URL,
                )
                logger.info("Groq LLM client initialized: model=%s", self.model)
            except Exception as exc:
                logger.warning("Failed to initialize OpenAI client: %s", exc)
                self._client = None
        else:
            logger.info("LLM client in MOCK mode (provider=%s, key_present=%s)", self.provider, bool(self.api_key))

    def generate(
        self,
        prompt: str,
        context_chunks: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
    ) -> dict[str, Any]:
        """
        Generate a response from the LLM or mock fallback.

        Parameters
        ----------
        prompt : str
            The fully assembled user prompt.
        context_chunks : list[dict], optional
            Retrieved context chunks (used for mock fallback verbatim quotes).
        system_prompt : str, optional
            System prompt for the chat completion.

        Returns
        -------
        dict
            Keys: 'response' (str), 'provider' (str), 'model' (str),
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
                try:
                    response = self._client.chat.completions.create(
                        model=model_to_try,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    used_model = model_to_try
                except Exception as model_err:
                    fallback_models = ["openai/gpt-oss-120b", "groq/compound-mini", "groq/compound", "qwen/qwen3.6-27b"]
                    response = None
                    used_model = self.model
                    for fb in fallback_models:
                        if fb == model_to_try:
                            continue
                        try:
                            response = self._client.chat.completions.create(
                                model=fb,
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                            )
                            used_model = fb
                            break
                        except Exception:
                            continue
                    if response is None:
                        raise model_err

                raw_content = response.choices[0].message.content or ""
                # Strip internal reasoning/thinking blocks from reasoning models (e.g. Qwen/DeepSeek/GPT-OSS)
                content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                # Discard any conversational preamble / scratchpad before ## Recommendation
                if "## Recommendation" in content:
                    content = "## Recommendation" + content.split("## Recommendation", 1)[1]
                elif "## recommendation" in content:
                    content = "## Recommendation" + content.split("## recommendation", 1)[1]
                elif not content and raw_content:
                    content = raw_content.strip()

                latency_ms = round((time.perf_counter() - t0) * 1000, 2)

                result = {
                    "response": content,
                    "provider": "groq",
                    "model": used_model,
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
                "status": result.get("status"),
                "latency_ms": result.get("latency_ms"),
                "prompt_length": len(prompt),
                "response_length": len(result.get("response", "")),
            }
            if result.get("error"):
                entry["error"] = result["error"]
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("Failed to log generation: %s", exc)
