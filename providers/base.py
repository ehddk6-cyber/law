from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResult:
    provider: str
    model: str | None
    output_text: str | None
    error: str | None = None
    raw: dict | None = None


class BaseProvider:
    provider_name = "base"

    def __init__(self, model: str | None = None):
        self.model = model

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResult:
        raise NotImplementedError
