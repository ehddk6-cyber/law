from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from providers import resolve_provider_options
from qa.agent_exec import probe_agent
from qa.answering import GroundedAnswerer
from qa.eval_report import CASES, run_case
from qa.legal_analysis import (
    analyze_law_query,
    build_quasi_provision_chain,
    detect_erroneous_omission,
    format_analysis_json,
    format_analysis_text,
    generate_ox_quiz,
)
from qa.response_schema import ANSWER_RESULT_SCHEMA, SCHEMA_VERSION
from qa.retrievers import LawRetriever
from qa.unified import UnifiedSearcher


def _json_dump(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_kv(key: str, value: object) -> None:
    if isinstance(value, list):
        print(f"{key}=" + ", ".join(str(item) for item in value))
        return
    print(f"{key}={value}")


def _handle_answer(args: argparse.Namespace) -> int:
    answerer = GroundedAnswerer(args.root)
    packet = answerer.answer(
        args.query,
        limit=args.limit,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        include_agent_prompts=args.include_agent_prompts,
        response_mode=args.response_mode,
        run_agent_name=args.run_agent,
        run_agent_model=args.run_agent_model,
        judge_mode=args.judge_mode,
        judge_agent=args.judge_agent,
        judge_model=args.judge_model,
        judge_critic_agent=args.judge_critic_agent,
        judge_critic_model=args.judge_critic_model,
        explain_style=args.explain_style,
    )

    if args.format == "json":
        _json_dump({"ok": True, "command": "answer", "result": packet.to_dict()})
        return 0

    _print_kv("query", packet.query)
    _print_kv("scope", packet.scope)
    _print_kv("question_type", packet.question_type)
    _print_kv("route", packet.route)
    _print_kv("status", packet.status)
    _print_kv("grounded", packet.grounded)
    _print_kv("source_type", packet.source_type)
    _print_kv("doc_id", packet.doc_id)
    _print_kv("explain_style", packet.explain_style)
    _print_kv("final_answer_source", packet.final_answer_source)
    print("answer=" + packet.answer)
    if packet.evidence:
        print("evidence:")
        for item in packet.evidence:
            print("- " + item)
    if packet.citations:
        print("citations:")
        for citation in packet.citations:
            print("- " + citation)
    if packet.warnings:
        print("warnings:")
        for warning in packet.warnings:
            print("- " + warning)
    if packet.exam_assessment:
        print("exam_assessment:")
        for key, value in packet.exam_assessment.items():
            if isinstance(value, list):
                print(f"- {key}: " + ", ".join(str(item) for item in value))
            else:
                print(f"- {key}: {value}")
    if packet.teaching_explanation:
        print("teaching_explanation:")
        for key, value in packet.teaching_explanation.items():
            if isinstance(value, list):
                print(f"- {key}: " + json.dumps(value, ensure_ascii=False))
            elif isinstance(value, dict):
                print(f"- {key}: " + json.dumps(value, ensure_ascii=False))
            else:
                print(f"- {key}: {value}")
    if packet.llm_provider or packet.llm_error:
        print("llm:")
        _print_kv("provider", packet.llm_provider)
        _print_kv("model", packet.llm_model)
        _print_kv("response_mode", packet.response_mode)
        if packet.llm_answer:
            print("llm_answer=" + packet.llm_answer)
        if packet.llm_error:
            print("llm_error=" + packet.llm_error)
    if packet.fallback_trace:
        _print_kv("fallback_trace", packet.fallback_trace)
    if packet.judge_result or packet.judge_error:
        print("judge:")
        _print_kv("judge_mode", packet.judge_mode)
        if packet.judge_result:
            print("judge_result=" + json.dumps(packet.judge_result, ensure_ascii=False))
        if packet.judge_error:
            _print_kv("judge_error", packet.judge_error)
        if packet.judge_trace:
            _print_kv("judge_trace", packet.judge_trace)
    if packet.agent_runs:
        print("agent_runs:")
        for name, result in packet.agent_runs.items():
            print(f"- {name}: ok={result.get('ok')} error={result.get('error')}")
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    searcher = UnifiedSearcher(args.root)
    decision, results = searcher.search(args.query, limit=args.limit)
    payload = {
        "ok": True,
        "command": "search",
        "route": {
            "strategy": decision.strategy,
            "raw_query": decision.raw_query,
            "normalized_query": decision.normalized_query,
            "law_name": decision.law_name,
            "article_no": decision.article_no,
            "serial_no": decision.serial_no,
            "case_no": decision.case_no,
            "precedent_case_no": decision.precedent_case_no,
            "decc_case_no": decision.decc_case_no,
            "detc_case_no": decision.detc_case_no,
            "expc_issue_no": decision.expc_issue_no,
        },
        "results": [
            {
                "source_type": item.source_type,
                "search_path": item.search_path,
                "score": item.score,
                "payload": item.payload,
            }
            for item in results
        ],
    }

    if args.format == "json":
        _json_dump(payload)
        return 0

    _print_kv("query", args.query)
    _print_kv("strategy", decision.strategy)
    if decision.law_name:
        _print_kv("law_name", decision.law_name)
    if decision.article_no:
        _print_kv("article_no", decision.article_no)
    if decision.serial_no:
        _print_kv("serial_no", decision.serial_no)
    if decision.precedent_case_no:
        _print_kv("precedent_case_no", decision.precedent_case_no)
    if decision.decc_case_no:
        _print_kv("decc_case_no", decision.decc_case_no)
    if decision.detc_case_no:
        _print_kv("detc_case_no", decision.detc_case_no)
    if decision.expc_issue_no:
        _print_kv("expc_issue_no", decision.expc_issue_no)
    print("results:")
    if not results:
        print("- none")
        return 0
    for item in results:
        payload = item.payload
        title = (
            payload.get("title")
            or payload.get("case_name")
            or payload.get("name")
            or "-"
        )
        doc_id = (
            payload.get("doc_id")
            or payload.get("source_id")
            or payload.get("law_id")
            or "-"
        )
        print(
            f"- source_type={item.source_type} search_path={item.search_path} "
            f"score={item.score} doc_id={doc_id} title={title}"
        )
    return 0


def _handle_health(args: argparse.Namespace) -> int:
    answerer = GroundedAnswerer(args.root)
    runtime = resolve_provider_options(args.root, None, None)
    payload = {
        "ok": True,
        "command": "health",
        "root": str(args.root),
        "schema_version": SCHEMA_VERSION,
        "default_runtime": {
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
        "artifacts": {
            "qa": str(args.root / "qa"),
            "providers": str(args.root / "providers"),
            "settings": str(args.root / "settings" / "provider.json"),
        },
    }

    if args.format == "json":
        _json_dump(payload)
        return 0

    _print_kv("root", payload["root"])
    _print_kv("schema_version", payload["schema_version"])
    print("default_runtime:")
    for key, value in payload["default_runtime"].items():
        if isinstance(value, list):
            print(f"- {key}: " + " -> ".join(str(item) for item in value))
        else:
            print(f"- {key}: {value}")
    print("agents:")
    for name, info in payload["agents"].items():
        print(
            f"- {name}: available={info.get('available')} ready={info.get('ready')} "
            f"reason={info.get('reason')}"
        )
    return 0


def _handle_schema(args: argparse.Namespace) -> int:
    payload = {
        "ok": True,
        "command": "schema",
        "schema_version": SCHEMA_VERSION,
        "result_schema": ANSWER_RESULT_SCHEMA,
    }
    if args.format == "json":
        _json_dump(payload)
        return 0

    _print_kv("schema_version", SCHEMA_VERSION)
    print("result_schema_keys=" + ", ".join(ANSWER_RESULT_SCHEMA["properties"].keys()))
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    answerer = GroundedAnswerer(args.root)
    rows = [run_case(answerer, case, args.response_mode) for case in CASES]
    passed = sum(1 for row in rows if row["ok"])
    summary = {
        "ok": passed == len(rows),
        "command": "eval",
        "response_mode": args.response_mode,
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round((passed / len(rows)) * 100, 2) if rows else 0.0,
        "cases": rows,
    }

    if args.format == "json":
        _json_dump(summary)
    else:
        _print_kv("response_mode", summary["response_mode"])
        _print_kv("passed", f"{summary['passed']}/{summary['total']}")
        _print_kv("pass_rate", summary["pass_rate"])
        for row in rows:
            print(
                f"{'OK' if row['ok'] else 'FAIL'} "
                f"name={row['name']} status={row['status']} grounded={row['grounded']} "
                f"doc_id={row['doc_id']} final_answer_source={row['final_answer_source']}"
            )

    return 1 if args.strict and summary["failed"] else 0


def _handle_analyze(args: argparse.Namespace) -> int:
    artifacts = args.root / ".artifacts"
    law_db = artifacts / "law" / "law.sqlite3"
    law_retriever = LawRetriever(law_db, artifacts / "law" / "vector")

    result = analyze_law_query(args.query, law_retriever)

    if args.format == "json":
        _json_dump(
            {"ok": True, "command": "analyze", "result": format_analysis_json(result)}
        )
        return 0

    print(format_analysis_text(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human-readable and agent-friendly CLI for Legal QA."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    subparsers = parser.add_subparsers(dest="command", required=True)

    answer = subparsers.add_parser("answer", help="Generate a grounded answer.")
    answer.add_argument("query")
    answer.add_argument("--limit", type=int, default=5)
    answer.add_argument("--llm-provider")
    answer.add_argument("--llm-model")
    answer.add_argument("--include-agent-prompts", action="store_true")
    answer.add_argument(
        "--response-mode", choices=["grounded", "llm_preferred", "llm_only"]
    )
    answer.add_argument("--explain-style", choices=["admin_exam"])
    answer.add_argument(
        "--run-agent",
        choices=["codex", "claude-code", "claude-stepfree", "qwen", "gemini"],
    )
    answer.add_argument("--run-agent-model")
    answer.add_argument(
        "--judge-mode", choices=["off", "single", "debate"], default=None
    )
    answer.add_argument(
        "--judge-agent",
        choices=["codex", "claude-code", "claude-stepfree", "qwen", "gemini"],
    )
    answer.add_argument("--judge-model")
    answer.add_argument(
        "--judge-critic-agent",
        choices=["codex", "claude-code", "claude-stepfree", "qwen", "gemini"],
    )
    answer.add_argument("--judge-critic-model")
    answer.set_defaults(handler=_handle_answer)

    search = subparsers.add_parser("search", help="Run retrieval only.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.set_defaults(handler=_handle_search)

    health = subparsers.add_parser("health", help="Show runtime and agent health.")
    health.set_defaults(handler=_handle_health)

    schema = subparsers.add_parser("schema", help="Print the stable result schema.")
    schema.set_defaults(handler=_handle_schema)

    eval_parser = subparsers.add_parser("eval", help="Run compact regression cases.")
    eval_parser.add_argument(
        "--response-mode",
        choices=["grounded", "llm_preferred", "llm_only"],
        default="grounded",
    )
    eval_parser.add_argument("--strict", action="store_true")
    eval_parser.set_defaults(handler=_handle_eval)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze a law article with neighboring articles for comparison. "
        "Query must contain law name + article ref, e.g. '행정소송법 제28조'.",
    )
    analyze_parser.add_argument("query")
    analyze_parser.set_defaults(handler=_handle_analyze)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
