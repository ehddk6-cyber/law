from __future__ import annotations

import os

from providers.base import BaseProvider, LLMResult
from providers.http_utils import post_json


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        super().__init__(model=model or os.getenv("LEGAL_QA_LLM_MODEL") or "gemini-2.0-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("LEGAL_QA_LLM_API_KEY")

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        if not self.api_key:
            return LLMResult(self.provider_name, self.model, None, "GEMINI_API_KEY 또는 GOOGLE_API_KEY가 없습니다.")
        parts = []
        if system_prompt:
            parts.append({"text": system_prompt})
        parts.append({"text": prompt})
        try:
            raw = post_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}",
                {"contents": [{"parts": parts}]},
                headers={},
            )
            candidates = raw.get("candidates", [])
            text_parts = candidates[0]["content"]["parts"] if candidates else []
            text = "\n".join(part.get("text", "") for part in text_parts).strip()
            return LLMResult(self.provider_name, self.model, text or None, raw=raw)
        except Exception as exc:
            return LLMResult(self.provider_name, self.model, None, str(exc))
