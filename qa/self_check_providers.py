from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from providers import load_provider_config, resolve_provider_options
from qa.answering import GroundedAnswerer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provider-config self-check.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    config = load_provider_config(args.root)
    failures = 0

    config_ok = bool(config.get("providers"))
    print(f"{'OK' if config_ok else 'FAIL'} provider_config providers={sorted(config.get('providers', {}).keys())}")
    if not config_ok:
        failures += 1

    resolved_glm = resolve_provider_options(args.root, "glm", None)
    glm_ok = resolved_glm["provider"] == "glm" and resolved_glm["options"].get("base_url")
    print(f"{'OK' if glm_ok else 'FAIL'} glm_resolve base_url={resolved_glm['options'].get('base_url')}")
    if not glm_ok:
        failures += 1

    resolved_groq = resolve_provider_options(args.root, "groq", None)
    groq_ok = resolved_groq["provider"] == "groq" and resolved_groq["options"].get("base_url")
    print(f"{'OK' if groq_ok else 'FAIL'} groq_resolve base_url={resolved_groq['options'].get('base_url')}")
    if not groq_ok:
        failures += 1

    resolved_openrouter = resolve_provider_options(args.root, "openrouter", None)
    openrouter_ok = resolved_openrouter["provider"] == "openrouter" and resolved_openrouter["options"].get("base_url")
    print(f"{'OK' if openrouter_ok else 'FAIL'} openrouter_resolve base_url={resolved_openrouter['options'].get('base_url')}")
    if not openrouter_ok:
        failures += 1

    resolved_ollama_minimax = resolve_provider_options(args.root, "ollama-minimax", None)
    ollama_minimax_ok = resolved_ollama_minimax["provider"] == "ollama-minimax" and resolved_ollama_minimax["options"].get("base_url")
    print(f"{'OK' if ollama_minimax_ok else 'FAIL'} ollama_minimax_resolve base_url={resolved_ollama_minimax['options'].get('base_url')}")
    if not ollama_minimax_ok:
        failures += 1

    resolved_ollama_qwen = resolve_provider_options(args.root, "ollama-qwen", None)
    ollama_qwen_ok = resolved_ollama_qwen["provider"] == "ollama-qwen" and resolved_ollama_qwen["options"].get("base_url")
    print(f"{'OK' if ollama_qwen_ok else 'FAIL'} ollama_qwen_resolve base_url={resolved_ollama_qwen['options'].get('base_url')}")
    if not ollama_qwen_ok:
        failures += 1

    answerer = GroundedAnswerer(args.root)
    packet = answerer.answer("행정심판법 제18조", llm_provider="openrouter")
    provider_ok = packet.llm_provider in {"openrouter", "groq", "codex", "claude-code", "claude-stepfree", "glm", "ollama"} and (
        packet.llm_error is not None or packet.llm_answer is not None
    )
    print(
        f"{'OK' if provider_ok else 'FAIL'} openrouter_attach provider={packet.llm_provider} "
        f"error={packet.llm_error} fallback={packet.fallback_trace}"
    )
    if not provider_ok:
        failures += 1

    grounded_packet = answerer.answer("행정심판법 제18조")
    grounded_ok = grounded_packet.response_mode == "llm_preferred" and grounded_packet.final_answer_source in {"llm", "grounded"}
    print(
        f"{'OK' if grounded_ok else 'FAIL'} grounded_default "
        f"response_mode={grounded_packet.response_mode} final_answer_source={grounded_packet.final_answer_source}"
    )
    if not grounded_ok:
        failures += 1

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
