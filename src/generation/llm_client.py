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
)

logger = logging.getLogger(__name__)

# Load .env from project root or src/safety/.env
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(_project_root / "src" / "safety" / ".env")


def _build_mock_response(context_chunks: list[dict[str, Any]] | None = None) -> str:
    """
    Build a deterministic MOCK response in valid 6-section format.

    Uses the first context chunk for a verbatim citation if available.
    """
    if context_chunks and len(context_chunks) > 0:
        chunk = context_chunks[0]
        doc = chunk.get("document_name", "Unknown Document")
        sec = chunk.get("section_name", "General")
        page = chunk.get("start_page", "?")
        text = chunk.get("text", "")
        # Extract a verbatim 10-20 word phrase from the chunk
        words = text.split()
        quote_words = words[:15] if len(words) >= 15 else words
        verbatim_quote = " ".join(quote_words)
    else:
        doc = "USPSTF Guidelines"
        sec = "Recommendation"
        page = "1"
        verbatim_quote = "screening for depression in the general adult population"

    return (
        f"## Recommendation\n"
        f"The USPSTF recommends screening for depression in the general adult population, "
        f"including pregnant and postpartum persons. This is a Grade B recommendation.\n"
        f'[Doc: {doc} | Sec: {sec} | Pg: {page} | Quote: "{verbatim_quote}"]\n\n'
        f"## Population\n"
        f"This recommendation applies to adults aged 18 years and older, "
        f"including pregnant and postpartum persons and older adults.\n\n"
        f"## Screening Tool\n"
        f"Commonly used screening instruments include the Patient Health Questionnaire (PHQ-9), "
        f"PHQ-2, and the Edinburgh Postnatal Depression Scale (EPDS) for perinatal populations.\n\n"
        f"## Harms & Considerations\n"
        f"The USPSTF found adequate evidence that screening for depression in adults, "
        f"including older adults and pregnant and postpartum persons, has minimal harms.\n\n"
        f"## Evidence\n"
        f"The USPSTF reviewed evidence on the benefits and harms of screening for depression "
        f"and suicide risk in adults and found adequate evidence of benefit.\n\n"
        f"## Source\n"
        f"USPSTF Recommendation Statement, June 2023. Grade B.\n"
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

                model_to_try = self._active_model
                try:
                    response = self._client.chat.completions.create(
                        model=model_to_try,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    used_model = model_to_try
                except Exception as model_err:
                    if "model_not_found" in str(model_err) or "does not exist" in str(model_err):
                        # Try alternate active Groq models
                        fallback_models = ["groq/compound-mini", "groq/compound", "openai/gpt-oss-120b", "llama-3.1-8b-instant"]
                        response = None
                        used_model = self.model
                        for fb in fallback_models:
                            try:
                                response = self._client.chat.completions.create(
                                    model=fb,
                                    messages=messages,
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens,
                                )
                                used_model = fb
                                self._active_model = fb  # Cache working model for future calls
                                break
                            except Exception:
                                continue
                        if response is None:
                            raise model_err
                    else:
                        raise model_err

                content = response.choices[0].message.content or ""
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
                mock_content = _build_mock_response(context_chunks)
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
            mock_content = _build_mock_response(context_chunks)
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
