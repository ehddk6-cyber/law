from __future__ import annotations

import os

from providers.base import BaseProvider, LLMResult
from providers.http_utils import post_json


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model=model or os.getenv("LEGAL_QA_LLM_MODEL") or "claude-3-5-sonnet-latest")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LEGAL_QA_LLM_API_KEY")

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        if not self.api_key:
            return LLMResult(self.provider_name, self.model, None, "ANTHROPIC_API_KEY가 없습니다.")
        payload = {
            "model": self.model,
            "max_tokens": 1200,
            "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            raw = post_json(
                "https://api.anthropic.com/v1/messages",
                payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
            blocks = raw.get("content", [])
            text = "\n".join(block.get("text", "") for block in blocks if block.get("type") == "text").strip()
            return LLMResult(self.provider_name, self.model, text or None, raw=raw)
        except Exception as exc:
            return LLMResult(self.provider_name, self.model, None, str(exc))
