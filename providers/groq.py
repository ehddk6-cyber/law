from __future__ import annotations

import os

from providers.base import BaseProvider, LLMResult
from providers.http_utils import post_json


class GroqProvider(BaseProvider):
    provider_name = "groq"

    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        super().__init__(model=model or os.getenv("LEGAL_QA_LLM_MODEL") or "llama-3.3-70b-versatile")
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("LEGAL_QA_LLM_API_KEY")
        self.base_url = (base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        if not self.api_key:
            return LLMResult(self.provider_name, self.model, None, "GROQ_API_KEY가 없습니다.")
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
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            text = raw["choices"][0]["message"]["content"]
            return LLMResult(self.provider_name, self.model, text, raw=raw)
        except Exception as exc:
            return LLMResult(self.provider_name, self.model, None, str(exc))
