"""
Pluggable LLM provider layer.

Both providers use *structured outputs* so the model is constrained to emit JSON
matching our schema — no more string-scraping for the first '{' and last '}'.

- OllamaProvider: local models (Gemma, Qwen, ...) via Ollama's `format` parameter,
  which accepts a JSON schema.
- OpenAIProvider: OpenAI models via response_format json_schema.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import List

import requests

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Raised for any provider-side failure (connection, HTTP, invalid output)."""


class LLMProvider(ABC):
    @abstractmethod
    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        """Return a dict guaranteed to be valid JSON (schema-constrained)."""
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, temperature: float = 0.2,
                 timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            # Thinking models (e.g. Gemma) otherwise spend the token budget on hidden
            # reasoning and can return empty content; go straight to the JSON answer.
            "think": False,
            "format": schema,  # <-- structured output: constrains generation to schema
            "options": {
                "temperature": self.temperature,
                "num_ctx": 16384,
                # Ensure the model has room to finish the JSON (avoids truncated,
                # unterminated output on longer analyses).
                "num_predict": 6144,
            },
        }

        data = self._post(payload)
        content = (data.get("message", {}) or {}).get("content", "") or ""
        if not content.strip():
            reason = data.get("done_reason", "unknown")
            raise ProviderError(
                f"Ollama returned empty content (done_reason={reason}). "
                "Try a different model or reduce the query length."
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderError(f"Ollama returned non-JSON content: {e}") from e

    def _post(self, payload: dict) -> dict:
        """POST to /api/chat, retrying without `think` if the model rejects it."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.RequestException as e:
            raise ProviderError(
                f"Could not reach Ollama at {self.base_url}. Is it running? ({e})"
            ) from e

        # Some models don't support the `think` field; drop it and retry once.
        if resp.status_code == 400 and "think" in resp.text.lower() and "think" in payload:
            payload = {k: v for k, v in payload.items() if k != "think"}
            return self._post(payload)

        if resp.status_code != 200:
            raise ProviderError(f"Ollama returned {resp.status_code}: {resp.text}")
        return resp.json()


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.2):
        # Imported lazily so the local-only path doesn't require the openai package.
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "analysis",
                        "schema": schema,
                        "strict": False,
                    },
                },
            )
        except Exception as e:  # openai raises a variety of typed errors
            raise ProviderError(f"OpenAI request failed: {e}") from e

        content = resp.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ProviderError(f"OpenAI returned non-JSON content: {e}") from e


def list_ollama_models(base_url: str, timeout: int = 5) -> List[str]:
    """List locally installed chat-capable Ollama models (for the UI dropdown).

    Embedding-only models (e.g. nomic-embed-text) are excluded since they can't
    generate a completion.
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        chat = []
        for m in models:
            name = m.get("name")
            if not name:
                continue
            caps = m.get("capabilities")
            # If capabilities are reported, keep only completion models; otherwise
            # keep the model (older Ollama versions omit the field).
            if caps is None or "completion" in caps:
                chat.append(name)
        return sorted(chat)
    except requests.RequestException as e:
        logger.warning("Could not list Ollama models: %s", e)
        return []
