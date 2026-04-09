from __future__ import annotations

import json
import os
import re
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("settings/provider.json")
SHELL_EXPORT_PATH = Path.home() / ".bashrc"


def load_provider_config(root: Path) -> dict:
    config_path = root / DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_shell_exports() -> dict[str, str]:
    if not SHELL_EXPORT_PATH.exists():
        return {}
    text = SHELL_EXPORT_PATH.read_text(encoding="utf-8", errors="ignore")
    exports: dict[str, str] = {}
    for key in ("OPENROUTER_API_KEY", "GROQ_API_KEY", "GLM_API_KEY", "BIGMODEL_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        match = re.search(rf'export\s+{re.escape(key)}="([^"]*)"', text)
        if match:
            exports[key] = match.group(1).strip()
    return exports


def resolve_provider_options(root: Path, provider_name: str | None, model: str | None) -> dict:
    config = load_provider_config(root)
    shell_exports = _load_shell_exports()
    defaults = config.get("default", {})
    providers = config.get("providers", {})

    resolved_provider = provider_name or os.getenv("LEGAL_QA_LLM_PROVIDER") or defaults.get("provider")
    provider_options = providers.get(resolved_provider, {}) if resolved_provider else {}
    resolved_model = (
        model
        or os.getenv("LEGAL_QA_LLM_MODEL")
        or provider_options.get("model")
        or defaults.get("model")
    )
    env_key = provider_options.get("env_key")
    api_key = None
    if env_key:
        api_key = os.getenv(env_key) or shell_exports.get(env_key)
    if api_key:
        provider_options = {**provider_options, "api_key": api_key}

    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "options": provider_options,
        "response_mode": defaults.get("response_mode", "grounded"),
        "judge_mode": defaults.get("judge_mode", "off"),
        "judge_chain": list(defaults.get("judge_chain", [])),
        "include_agent_prompts": bool(defaults.get("include_agent_prompts", False)),
        "fallback_chain": list(defaults.get("fallback_chain", [])),
        "agent_timeout_seconds": int(defaults.get("agent_timeout_seconds", 180)),
        "judge_timeout_seconds": int(defaults.get("judge_timeout_seconds", defaults.get("agent_timeout_seconds", 180))),
        "config_path": str(root / DEFAULT_CONFIG_PATH),
    }
