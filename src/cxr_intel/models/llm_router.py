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
    """Return (provider_name, base_url, api_key).

    Set LLM_PROVIDER=nvidia (or =openrouter) to force a choice; otherwise
    OpenRouter is preferred when both keys are present.
    """
    forced = (os.getenv("LLM_PROVIDER") or "").strip().lower()

    if forced == "nvidia" or (not forced and not os.getenv("OPENROUTER_API_KEY")):
        if os.getenv("NVIDIA_API_KEY"):
            return (
                "nvidia",
                os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
                os.environ["NVIDIA_API_KEY"],
            )
    if forced == "openrouter" or not forced:
        if os.getenv("OPENROUTER_API_KEY"):
            return (
                "openrouter",
                os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                os.environ["OPENROUTER_API_KEY"],
            )
    # Final fallback to NVIDIA if openrouter not present
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

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
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
        choice = resp.choices[0]
        content = getattr(choice.message, "content", None)
        if content is None:
            # Some providers null content on moderation/refusal/finish_reason=content_filter
            reason = getattr(choice, "finish_reason", "unknown")
            log.warning("Provider returned content=None (finish_reason=%s) — returning empty", reason)
            return ""
        return content.strip()

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        self.response_format_json = True
        text = self.chat(system, user)
        if not text:
            raise ValueError("LLM returned empty content (likely refusal/moderation)")
        # Strip markdown code fences if present (```json ... ```)
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise
