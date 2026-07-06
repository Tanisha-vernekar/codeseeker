"""Optional LLM integration for richer explanations and Q&A.

This layer is entirely optional. When an OpenAI-compatible API key is available
(via ``OPENAI_API_KEY``) and the ``openai`` package is installed, ``codeseeker``
can use an LLM to summarise a repository or answer a natural-language question
grounded in retrieved code (RAG). When it is not available, every feature
degrades gracefully to a deterministic, heuristic response so the tool always
works offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    max_tokens: int = 600
    temperature: float = 0.2


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion API."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.config.api_key_env)

    def available(self) -> bool:
        """Return ``True`` if both an API key and the SDK are present."""
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            base_url = os.environ.get(self.config.base_url_env)
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def complete(self, system: str, user: str) -> str:
        """Run a single chat completion and return the text response."""
        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def get_client(config: LLMConfig | None = None) -> LLMClient:
    return LLMClient(config)
