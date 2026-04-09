from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request, urlopen


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fetch_json(url: str, payload: dict | None = None) -> dict:
    if payload is None:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local HTTP API self-check.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    api_path = args.root / "qa" / "http_api.py"
    server = subprocess.Popen(
        [sys.executable, str(api_path), "--root", str(args.root), "--host", args.host, "--port", str(args.port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        health_url = f"http://{args.host}:{args.port}/health"
        answer_url = f"http://{args.host}:{args.port}/answer"

        last_error: Exception | None = None
        for _ in range(30):
            try:
                payload = _fetch_json(health_url)
                if payload.get("ok") is True:
                    last_error = None
                    break
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)
        if last_error is not None:
            raise last_error

        capability_result = _fetch_json(f"http://{args.host}:{args.port}/capabilities")

        get_result = _fetch_json(f"{answer_url}?query={quote('행정심판법 제18조')}")
        post_result = _fetch_json(answer_url, {"query": "2004헌마275"})
        llm_result = _fetch_json(
            answer_url,
            {
                "query": "행정심판법 제18조",
                "llm_provider": "groq",
                "response_mode": "llm_preferred",
            },
        )

        checks = [
            ("CAPABILITIES", "llm_provider_fallback" in capability_result.get("features", [])),
            ("GET", get_result["result"]["doc_id"] == "law:001363"),
            ("POST", post_result["result"]["doc_id"] == "detc:10026"),
            (
                "LLM",
                llm_result["result"]["response_mode"] == "llm_preferred"
                and llm_result["result"]["llm_provider"] in {"groq", "codex", "claude-code", "glm", "ollama"}
                and llm_result["result"]["final_answer_source"] in {"llm", "grounded"},
            ),
        ]
        failures = 0
        for label, ok in checks:
            status = "OK" if ok else "FAIL"
            print(f"{status} {label}")
            if not ok:
                failures += 1
        raise SystemExit(1 if failures else 0)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
