"""OpenAI-compatible LLM client routing to OpenRouter or NVIDIA NIM.

Used by:
  - qa_dataset.synth_generator (synthetic QA generation)
  - eval.llm_judge (LLM-as-judge scoring)
  - generation.qa_pipeline (text-only generator ablation)

Picks the provider based on env vars at construction time. Retries with
exponential backoff via tenacity.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from cxr_intel.utils.logging import get_logger

log = get_logger(__name__)


def _pick_provider() -> tuple[str, str, str]:
    """Return (provider_name, base_url, api_key)."""
    if os.getenv("OPENROUTER_API_KEY"):
        return (
            "openrouter",
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            os.environ["OPENROUTER_API_KEY"],
        )
    if os.getenv("NVIDIA_API_KEY"):
        return (
            "nvidia",
            os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            os.environ["NVIDIA_API_KEY"],
        )
    raise RuntimeError(
        "No LLM API credentials found. Set OPENROUTER_API_KEY or NVIDIA_API_KEY."
    )


@dataclass
class LLMRouter:
    model: str = field(default_factory=lambda: os.getenv("QA_SYNTH_MODEL",
                                                         "meta-llama/llama-3.3-70b-instruct"))
    temperature: float = 0.2
    max_tokens: int = 1024
    response_format_json: bool = False
    _client: Any = None
    _provider: str = ""

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        self._provider, base_url, api_key = _pick_provider()
        log.info("LLMRouter using provider=%s model=%s", self._provider, self.model)
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
    def chat(self, system: str, user: str) -> str:
        self._ensure_client()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if self.response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        self.response_format_json = True
        text = self.chat(system, user)
        # Some providers ignore response_format; fall back to robust parse.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise
