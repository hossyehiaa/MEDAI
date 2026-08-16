"""
LLM Client — Thin wrapper for calling the language model.

Supports OpenAI-compatible APIs. Swap the client for local models
(e.g. Ollama) by changing the ``base_url``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class LLMClient:
    """Manages LLM calls with retry and logging."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
        )

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """
        Send messages to the LLM and return the assistant's reply.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style chat messages (``role`` + ``content``).

        Returns
        -------
        str
            The assistant's response text.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        logger.info("Calling LLM model=%s, msgs=%d", self.model, len(messages))

        response = self.client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""

        logger.info("LLM response: %d chars", len(content))
        return content
