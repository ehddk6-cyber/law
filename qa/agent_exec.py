from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import json
from pathlib import Path


def _load_shell_function(function_name: str, rc_path: Path | None = None) -> str | None:
    target = rc_path or (Path.home() / ".bashrc")
    if not target.exists():
        return None
    text = target.read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf"(?ms)^{re.escape(function_name)}\(\)\s*\{{.*?^\}}", text)
    if not match:
        return None
    return match.group(0)


def probe_agent(agent_name: str, project_root: Path, timeout: int = 15) -> dict:
    normalized = agent_name.strip().lower()
    if normalized in {"codex", "qwen", "gemini"}:
        binary_name = normalized
        binary = shutil.which(binary_name)
        if not binary:
            return {
                "agent": normalized,
                "available": False,
                "ready": False,
                "reason": "missing_binary",
            }
        try:
            completed = subprocess.run(
                [binary, "--version"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            version = (
                (completed.stdout or completed.stderr).strip().splitlines()[0]
                if (completed.stdout or completed.stderr)
                else None
            )
            return {
                "agent": normalized,
                "available": True,
                "ready": completed.returncode == 0,
                "binary": binary,
                "version": version,
                "reason": None
                if completed.returncode == 0
                else f"returncode={completed.returncode}",
            }
        except subprocess.TimeoutExpired:
            return {
                "agent": normalized,
                "available": True,
                "ready": False,
                "binary": binary,
                "reason": f"timeout>{timeout}s",
            }

    if normalized == "claude-stepfree":
        function_def = _load_shell_function("claude-stepfree")
        if not function_def:
            return {
                "agent": normalized,
                "available": False,
                "ready": False,
                "reason": "missing_function",
            }
        try:
            completed = subprocess.run(
                ["bash", "-lc", f"{function_def}\nclaude-stepfree --version"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            combined = "\n".join(part for part in (stdout, stderr) if part).strip()
            return {
                "agent": normalized,
                "available": True,
                "ready": completed.returncode == 0 and bool(stdout),
                "binary": "bash-function:claude-stepfree",
                "reason": None
                if completed.returncode == 0 and bool(stdout)
                else (combined or f"returncode={completed.returncode}"),
                "detail": combined or None,
            }
        except subprocess.TimeoutExpired:
            return {
                "agent": normalized,
                "available": True,
                "ready": False,
                "binary": "bash-function:claude-stepfree",
                "reason": f"timeout>{timeout}s",
            }

    if normalized in {"claude", "claude-code"}:
        binary = shutil.which("claude")
        if not binary:
            return {
                "agent": "claude-code",
                "available": False,
                "ready": False,
                "reason": "missing_binary",
            }
        cmd = [
            binary,
            "-p",
            "--output-format",
            "text",
            "--input-format",
            "text",
            "--add-dir",
            str(project_root),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                input="ping",
                timeout=timeout,
                check=False,
            )
            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            combined = "\n".join(part for part in (stdout, stderr) if part).strip()
            if "Not logged in" in combined or "Please run /login" in combined:
                return {
                    "agent": "claude-code",
                    "available": True,
                    "ready": False,
                    "binary": binary,
                    "reason": "login_required",
                    "detail": combined,
                }
            return {
                "agent": "claude-code",
                "available": True,
                "ready": completed.returncode == 0 and bool(stdout),
                "binary": binary,
                "reason": None
                if completed.returncode == 0 and bool(stdout)
                else f"returncode={completed.returncode}",
                "detail": combined or None,
            }
        except subprocess.TimeoutExpired:
            return {
                "agent": "claude-code",
                "available": True,
                "ready": False,
                "binary": binary,
                "reason": f"timeout>{timeout}s",
            }

    return {"agent": normalized, "available": False, "ready": False, "reason": "unsupported_agent"}


def run_agent(
    agent_name: str, prompt: str, project_root: Path, model: str | None = None, timeout: int = 180
) -> dict:
    normalized = agent_name.strip().lower()
    if normalized == "codex":
        binary = shutil.which("codex")
        if not binary:
            return {"agent": normalized, "ok": False, "error": "codex 바이너리를 찾지 못했습니다."}
        cmd = [
            binary,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-C",
            str(project_root),
        ]
        if model:
            cmd += ["--model", model]
        cmd.append(prompt)
    elif normalized == "qwen":
        binary = shutil.which("qwen")
        if not binary:
            return {"agent": normalized, "ok": False, "error": "qwen 바이너리를 찾지 못했습니다."}
        cmd = [
            binary,
            "-p",
            prompt,
            "-o",
            "json",
            "--input-format",
            "text",
            "--yolo",
            "--add-dir",
            str(project_root),
        ]
        if model:
            cmd += ["--model", model]
    elif normalized == "gemini":
        binary = shutil.which("gemini")
        if not binary:
            return {"agent": normalized, "ok": False, "error": "gemini 바이너리를 찾지 못했습니다."}
        cmd = [
            binary,
            "-p",
            prompt,
            "-o",
            "json",
            "--yolo",
            "--include-directories",
            str(project_root),
        ]
        if model:
            cmd += ["--model", model]
    elif normalized == "claude-stepfree":
        function_def = _load_shell_function("claude-stepfree")
        if not function_def:
            return {
                "agent": normalized,
                "ok": False,
                "error": "claude-stepfree 함수를 ~/.bashrc에서 찾지 못했습니다.",
            }
        command = (
            function_def
            + "\nclaude-stepfree -p --output-format json --input-format text --add-dir "
            + shlex.quote(str(project_root))
        )
        if model:
            command += " --model " + shlex.quote(model)
        cmd = ["bash", "-lc", command]
    elif normalized in {"claude", "claude-code"}:
        binary = shutil.which("claude")
        if not binary:
            return {"agent": normalized, "ok": False, "error": "claude 바이너리를 찾지 못했습니다."}
        cmd = [
            binary,
            "-p",
            "--output-format",
            "text",
            "--input-format",
            "text",
            "--add-dir",
            str(project_root),
        ]
        if model:
            cmd += ["--model", model]
    else:
        return {"agent": normalized, "ok": False, "error": f"unsupported_agent={agent_name}"}

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            input=prompt if normalized in {"claude", "claude-code", "claude-stepfree"} else None,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if normalized in {"qwen", "gemini"} and stdout:
            try:
                raw = json.loads(stdout)
                if isinstance(raw, list):
                    result_items = [
                        item
                        for item in raw
                        if isinstance(item, dict) and item.get("type") == "result"
                    ]
                    text_items = [
                        item
                        for item in raw
                        if isinstance(item, dict)
                        and item.get("type") == "assistant"
                        and isinstance(item.get("message"), dict)
                    ]
                    if result_items:
                        result_text = str(result_items[-1].get("result") or "").strip()
                        if result_text:
                            return {
                                "agent": normalized,
                                "ok": True,
                                "returncode": completed.returncode,
                                "output": result_text,
                                "error": None,
                            }
                    for item in reversed(text_items):
                        content = item.get("message", {}).get("content") or []
                        for part in reversed(content):
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_value = str(part.get("text") or "").strip()
                                if text_value:
                                    return {
                                        "agent": normalized,
                                        "ok": True,
                                        "returncode": completed.returncode,
                                        "output": text_value,
                                        "error": None,
                                    }
                elif isinstance(raw, dict):
                    result_text = str(raw.get("result") or raw.get("text") or "").strip()
                    if result_text:
                        return {
                            "agent": normalized,
                            "ok": True,
                            "returncode": completed.returncode,
                            "output": result_text,
                            "error": None,
                        }
            except json.JSONDecodeError:
                pass
        if normalized == "claude-stepfree" and stdout:
            try:
                raw = json.loads(stdout)
                result_text = (raw.get("result") or "").strip()
                if result_text:
                    return {
                        "agent": normalized,
                        "ok": True,
                        "returncode": completed.returncode,
                        "output": result_text,
                        "error": None,
                    }
                return {
                    "agent": normalized,
                    "ok": False,
                    "returncode": completed.returncode,
                    "output": None,
                    "error": "empty_result",
                }
            except json.JSONDecodeError:
                pass
        ok = completed.returncode == 0 and bool(stdout)
        if not ok and completed.returncode == 0 and not stdout and not stderr:
            error = "empty_output"
        else:
            error = None if ok else (stderr or stdout or f"returncode={completed.returncode}")
        return {
            "agent": normalized,
            "ok": ok,
            "returncode": completed.returncode,
            "output": stdout or None,
            "error": error,
        }
    except subprocess.TimeoutExpired:
        return {"agent": normalized, "ok": False, "error": f"timeout>{timeout}s"}
