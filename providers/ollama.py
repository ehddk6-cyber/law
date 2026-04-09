from __future__ import annotations

import os

from providers.base import BaseProvider, LLMResult
from providers.http_utils import post_json


class OllamaProvider(BaseProvider):
    provider_name = "ollama"

    def __init__(self, model: str | None = None, base_url: str | None = None):
        super().__init__(model=model or os.getenv("LEGAL_QA_LLM_MODEL") or "qwen2.5:7b-instruct")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        try:
            raw = post_json(f"{self.base_url}/api/generate", payload, headers={})
            return LLMResult(self.provider_name, self.model, raw.get("response"), raw=raw)
        except Exception as exc:
            return LLMResult(self.provider_name, self.model, None, str(exc))
