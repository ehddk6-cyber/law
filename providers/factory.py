from __future__ import annotations

from providers.anthropic import AnthropicProvider
from providers.base import BaseProvider
from providers.gemini import GeminiProvider
from providers.glm import GLMProvider
from providers.groq import GroqProvider
from providers.ollama import OllamaProvider
from providers.openai_compatible import OpenAICompatibleProvider


def create_provider(name: str, model: str | None = None, **options) -> BaseProvider:
    normalized = name.strip().lower()
    if normalized in {"openai", "openai-compatible", "openai_compatible", "lmstudio", "vllm", "openrouter"}:
        return OpenAICompatibleProvider(
            model=model,
            base_url=options.get("base_url"),
            api_key=options.get("api_key"),
            provider_name="openrouter" if normalized == "openrouter" else normalized,
        )
    if normalized in {"anthropic", "claude"}:
        return AnthropicProvider(model=model, api_key=options.get("api_key"))
    if normalized in {"gemini", "google"}:
        return GeminiProvider(model=model, api_key=options.get("api_key"))
    if normalized in {"groq"}:
        return GroqProvider(model=model, api_key=options.get("api_key"), base_url=options.get("base_url"))
    if normalized in {"glm", "bigmodel"}:
        return GLMProvider(model=model, api_key=options.get("api_key"), base_url=options.get("base_url"))
    if normalized in {"ollama-minimax", "ollama_minimax"}:
        return OllamaProvider(model=model or "minimax-m2.7:cloud", base_url=options.get("base_url"))
    if normalized in {"ollama-qwen", "ollama_qwen"}:
        return OllamaProvider(model=model or "qwen3.5:cloud", base_url=options.get("base_url"))
    if normalized in {"ollama", "local"}:
        return OllamaProvider(model=model, base_url=options.get("base_url"))
    raise ValueError(f"unsupported_provider={name}")
