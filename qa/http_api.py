from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from providers import resolve_provider_options
from qa.answering import GroundedAnswerer
from qa.agent_exec import probe_agent
from qa.response_schema import ANSWER_RESULT_SCHEMA, SCHEMA_VERSION


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _html_bytes(path: Path) -> bytes:
    return path.read_text(encoding="utf-8").encode("utf-8")


HEALTH_CACHE_TTL_SECONDS = 15
_HEALTH_CACHE: dict | None = None
_HEALTH_CACHE_AT: float = 0.0


def _build_health_payload(answerer: GroundedAnswerer) -> dict:
    runtime = resolve_provider_options(answerer.root, None, None)
    return {
        "ok": True,
        "llm_config_path": str(answerer.root / "settings" / "provider.json"),
        "default_runtime": "grounded+llm-fallback",
        "defaults": {
            "provider": runtime.get("provider"),
            "model": runtime.get("model"),
            "response_mode": runtime.get("response_mode"),
            "judge_mode": runtime.get("judge_mode"),
            "judge_chain": runtime.get("judge_chain"),
            "fallback_chain": runtime.get("fallback_chain"),
        },
        "agents": {
            "codex": probe_agent("codex", answerer.root),
            "claude_code": probe_agent("claude-code", answerer.root),
            "claude_stepfree": probe_agent("claude-stepfree", answerer.root),
            "qwen": probe_agent("qwen", answerer.root),
            "gemini": probe_agent("gemini", answerer.root),
        },
    }


def _get_health_payload(answerer: GroundedAnswerer) -> dict:
    global _HEALTH_CACHE, _HEALTH_CACHE_AT
    import time

    now = time.time()
    if _HEALTH_CACHE and (now - _HEALTH_CACHE_AT) < HEALTH_CACHE_TTL_SECONDS:
        return _HEALTH_CACHE
    _HEALTH_CACHE = _build_health_payload(answerer)
    _HEALTH_CACHE_AT = now
    return _HEALTH_CACHE


def _extract_answer_params(params: dict | None = None, body: dict | None = None) -> dict:
    def _get(key: str, default=None):
        if body is not None:
            return body.get(key, default)
        if params is not None:
            vals = params.get(key)
            return (vals[0] if vals else default) if isinstance(vals, list) else default
        return default

    return {
        "limit": _get("limit", 5),
        "llm_provider": _get("llm_provider"),
        "llm_model": _get("llm_model"),
        "include_agent_prompts": _get("include_agent_prompts", False),
        "response_mode": _get("response_mode"),
        "explain_style": _get("explain_style"),
        "run_agent_name": _get("run_agent"),
        "run_agent_model": _get("run_agent_model"),
        "judge_mode": _get("judge_mode"),
        "judge_agent": _get("judge_agent"),
        "judge_model": _get("judge_model"),
        "judge_critic_agent": _get("judge_critic_agent"),
        "judge_critic_model": _get("judge_critic_model"),
    }


def make_handler(answerer: GroundedAnswerer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "LegalQAHTTP/0.1"

        def _send_json(self, status: int, payload: dict) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status: int, html_path: Path) -> None:
            body = _html_bytes(html_path)
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _parse_limit(self, raw: str | None) -> int:
            if not raw:
                return 5
            try:
                value = int(raw)
            except ValueError:
                return 5
            return max(1, min(value, 20))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(HTTPStatus.OK, _get_health_payload(answerer))
                return

            if parsed.path == "/capabilities":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "features": [
                            "grounded_retrieval",
                            "exam_evaluator",
                            "llm_provider_fallback",
                            "agent_exec_codex",
                            "agent_exec_claude_code",
                            "agent_exec_claude_stepfree",
                            "agent_exec_qwen",
                            "agent_exec_gemini",
                        ],
                    },
                )
                return

            if parsed.path in {"/", "/app"}:
                html_path = answerer.root / "ui" / "index.html"
                self._send_html(HTTPStatus.OK, html_path)
                return

            if parsed.path == "/schema":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "schema_version": SCHEMA_VERSION,
                        "result_schema": ANSWER_RESULT_SCHEMA,
                    },
                )
                return

            if parsed.path != "/answer":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return

            params_qs = parse_qs(parsed.query)
            query = (params_qs.get("query") or [""])[0].strip()
            if not query:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "query_required"})
                return

            params = _extract_answer_params(params=params_qs)
            limit = max(1, min(int(params["limit"] or 5), 20))
            if isinstance(params["include_agent_prompts"], str):
                params["include_agent_prompts"] = params["include_agent_prompts"].lower() == "true"
            params["limit"] = limit
            packet = answerer.answer(
                query, **{k: v for k, v in params.items() if k != "limit"}, limit=limit
            )
            self._send_json(HTTPStatus.OK, {"ok": True, "result": packet.to_dict()})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/answer":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid_json"})
                return

            query = str(payload.get("query", "")).strip()
            if not query:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "query_required"})
                return

            params = _extract_answer_params(body=payload)
            limit = max(1, min(int(params["limit"] or 5), 20))
            params["limit"] = limit
            packet = answerer.answer(
                query, **{k: v for k, v in params.items() if k != "limit"}, limit=limit
            )
            self._send_json(HTTPStatus.OK, {"ok": True, "result": packet.to_dict()})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve grounded legal answers over local HTTP.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    answerer = GroundedAnswerer(args.root)
    handler = make_handler(answerer)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving=http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
