from __future__ import annotations

import os

from providers.base import BaseProvider, LLMResult
from providers.http_utils import post_json


class OpenAICompatibleProvider(BaseProvider):
    provider_name = "openai-compatible"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(model=model or os.getenv("LEGAL_QA_LLM_MODEL") or "gpt-4o-mini")
        if provider_name:
            self.provider_name = provider_name
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LEGAL_QA_LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("LEGAL_QA_LLM_API_KEY")
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        if not self.api_key:
            return LLMResult(self.provider_name, self.model, None, "OPENAI_API_KEY가 없습니다.")
        payload = {
            "model": self.model,
            "messages": [],
            "temperature": 0.2,
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": prompt})
        try:
            raw = post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                    "X-Title": os.getenv("OPENROUTER_X_TITLE", "legal-qa"),
                },
            )
            text = raw["choices"][0]["message"]["content"]
            return LLMResult(self.provider_name, self.model, text, raw=raw)
        except Exception as exc:
            return LLMResult(self.provider_name, self.model, None, str(exc))
