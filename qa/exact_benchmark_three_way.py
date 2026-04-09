from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.answering import GroundedAnswerer


BEOPMANG_BASE = "https://api.beopmang.org/api/v4"
CASE_NO_RE = re.compile(r"사건번호:\s*(.+)")
ISSUE_NO_RE = re.compile(r"해석례번호:\s*(.+)")


@dataclass(frozen=True)
class ExactCase:
    query: str
    category: str
    expected_doc_id: str
    expected_external_id: str
    expected_title: str
    law_id: str | None = None
    article: str | None = None


CASES = [
    ExactCase("행정심판법 제18조", "law", "law:001363", "001363", "대리인의 선임", law_id="001363", article="18"),
    ExactCase("민법 제750조", "law", "law:001706", "001706", "불법행위의 내용", law_id="001706", article="750"),
    ExactCase("국가배상법 제2조", "law", "law:001242", "001242", "배상책임", law_id="001242", article="2"),
    ExactCase("행정소송법 제28조", "law", "law:001218", "001218", "사정판결", law_id="001218", article="28"),
    ExactCase("84누180", "prec", "prec:100006", "100006", "양도소득세부과처분취소"),
    ExactCase("83누699", "prec", "prec:206840", "206840", "양도소득세부과처분취소"),
    ExactCase("2004헌마275", "detc", "detc:10026", "10026", "기소유예처분취소"),
    ExactCase("2012헌바95", "detc", "detc:45914", "45914", "국가보안법 제7조 제1항 등 위헌소원"),
    ExactCase("2000-04033", "decc", "decc:17109", "17109", "재확인신체검사등외판정처분취소청구"),
    ExactCase("05-0096", "expc", "expc:313107", "313107", "1959년 12월 31일 이전에 퇴직한 군인의 퇴직급여금 지급에 관한특별법 시행령 제4조제2항 및 3항"),
]


@dataclass
class SystemResult:
    system: str
    query: str
    category: str
    ok: bool
    grounded: bool | None
    matched_id: str | None
    matched_title: str | None
    status: str
    latency_ms: float
    note: str
    official_ref: str | None = None
    failure_reason: str | None = None


def _run_command(args: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return False, stderr or stdout or f"exit={completed.returncode}"
    return True, stdout


def _fetch_json(url: str, timeout: int = 30) -> tuple[bool, dict[str, Any] | str]:
    try:
        with urlopen(url, timeout=timeout) as response:
            return True, json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def run_law_open_data(case: ExactCase, root: Path) -> SystemResult:
    answerer = GroundedAnswerer(root)
    t0 = time.perf_counter()
    packet = answerer.answer(case.query, limit=3, response_mode="grounded")
    latency_ms = (time.perf_counter() - t0) * 1000
    ok = packet.doc_id == case.expected_doc_id and packet.grounded
    official_ref = next(
        (
            citation.replace("[공식검증] ", "")
            for citation in packet.citations
            if citation.startswith("[공식검증] ")
        ),
        None,
    )
    note = packet.citations[0] if packet.citations else ""
    return SystemResult(
        system="law_open_data",
        query=case.query,
        category=case.category,
        ok=ok,
        grounded=packet.grounded,
        matched_id=packet.doc_id,
        matched_title=packet.answer,
        status=packet.status,
        latency_ms=latency_ms,
        note=note,
        official_ref=official_ref,
        failure_reason=None if ok else packet.status,
    )


def _korean_law_search_args(case: ExactCase) -> list[str]:
    if case.category == "prec":
        return ["search_precedents", "--query", case.query, "--display", "10", "--apiKey", "da"]
    if case.category == "detc":
        return [
            "search_constitutional_decisions",
            "--query",
            case.query,
            "--display",
            "10",
            "--apiKey",
            "da",
        ]
    if case.category == "decc":
        return ["search_admin_appeals", "--query", case.query, "--display", "10", "--apiKey", "da"]
    if case.category == "expc":
        return ["search_interpretations", "--query", case.query, "--display", "10", "--apiKey", "da"]
    raise ValueError(f"unsupported category={case.category}")


def _extract_korean_law_match(output: str, case: ExactCase) -> tuple[str | None, str | None]:
    lines = output.splitlines()
    current_id = None
    current_title = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            current_id = stripped.split("]", 1)[0].lstrip("[")
            current_title = stripped.split("]", 1)[1].strip()
        case_match = CASE_NO_RE.search(stripped)
        issue_match = ISSUE_NO_RE.search(stripped)
        if case_match and case_match.group(1).strip() == case.query:
            return current_id, current_title
        if issue_match and issue_match.group(1).strip() == case.query:
            return current_id, current_title
    return None, None


def run_korean_law(case: ExactCase) -> SystemResult:
    t0 = time.perf_counter()
    if case.category == "law":
        ok, output = _run_command(
            [
                "korean-law",
                "get_law_text",
                "--lawId",
                str(case.law_id),
                "--jo",
                f"제{case.article}조",
                "--apiKey",
                "da",
            ]
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        success = ok and case.expected_title in output and case.query.split()[0] in output
        return SystemResult(
            system="korean-law-mcp",
            query=case.query,
            category=case.category,
            ok=success,
            grounded=success,
            matched_id=case.law_id if success else None,
            matched_title=case.expected_title if success else None,
            status="grounded" if success else "not_found",
            latency_ms=latency_ms,
            note="get_law_text" if ok else output,
            official_ref=f"law.go.kr / lawId={case.law_id} / 제{case.article}조",
            failure_reason=None if success else "law_text_mismatch",
        )

    ok, output = _run_command(["korean-law", *_korean_law_search_args(case)])
    latency_ms = (time.perf_counter() - t0) * 1000
    if not ok:
        return SystemResult(
            system="korean-law-mcp",
            query=case.query,
            category=case.category,
            ok=False,
            grounded=False,
            matched_id=None,
            matched_title=None,
            status="error",
            latency_ms=latency_ms,
            note=output,
            failure_reason="search_error",
        )
    matched_id, matched_title = _extract_korean_law_match(output, case)
    success = matched_id == case.expected_external_id
    return SystemResult(
        system="korean-law-mcp",
        query=case.query,
        category=case.category,
        ok=success,
        grounded=success,
        matched_id=matched_id,
        matched_title=matched_title,
        status="grounded" if success else "not_found",
        latency_ms=latency_ms,
        note="search exact case" if success else "exact identifier not found in search output",
        failure_reason=None if success else "search_exact_miss",
    )


def run_beopmang(case: ExactCase) -> SystemResult:
    t0 = time.perf_counter()
    if case.category == "law":
        ok, payload = _fetch_json(
            f"{BEOPMANG_BASE}/law?action=get&law_id={case.law_id}&article={case.article}"
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if not ok:
            return SystemResult(
                system="beopmang",
                query=case.query,
                category=case.category,
                ok=False,
                grounded=False,
                matched_id=None,
                matched_title=None,
                status="error",
                latency_ms=latency_ms,
                note=str(payload),
                failure_reason="fetch_error",
            )
        articles = payload.get("data", {}).get("articles", [])
        article = articles[0] if articles else None
        content = article.get("content") if article else ""
        success = bool(article) and article.get("law_id") == case.law_id and (
            case.expected_title in content
            or f"제{case.article}조" in content
        )
        return SystemResult(
            system="beopmang",
            query=case.query,
            category=case.category,
            ok=success,
            grounded=success,
            matched_id=article.get("law_id") if article else None,
            matched_title=article.get("content") if article else None,
            status="grounded" if success else "not_found",
            latency_ms=latency_ms,
            note="law/get" if success else "article not returned",
            failure_reason=None if success else "law_get_miss",
        )

    ok, payload = _fetch_json(f"{BEOPMANG_BASE}/case?action=search&q={quote(case.query)}")
    latency_ms = (time.perf_counter() - t0) * 1000
    if not ok:
        return SystemResult(
            system="beopmang",
            query=case.query,
            category=case.category,
            ok=False,
            grounded=False,
            matched_id=None,
            matched_title=None,
            status="error",
            latency_ms=latency_ms,
            note=str(payload),
            failure_reason="fetch_error",
        )
    results = payload.get("data", {}).get("results", [])
    matched = None
    for row in results:
        if row.get("case_no") == case.query:
            matched = row
            break
    matched_id = None
    if matched:
        for key in ("prec_id", "detc_id", "decc_id", "expc_id", "case_id"):
            if key in matched:
                matched_id = str(matched[key])
                break
    success = matched is not None and matched_id == case.expected_external_id
    return SystemResult(
        system="beopmang",
        query=case.query,
        category=case.category,
        ok=success,
        grounded=success,
        matched_id=matched_id,
        matched_title=matched.get("case_name") if matched else None,
        status="grounded" if success else "not_found",
        latency_ms=latency_ms,
        note="case/search exact case_no" if success else f"results={len(results)}",
        failure_reason=None if success else "case_search_miss",
    )


def build_summary(results: list[SystemResult]) -> dict[str, Any]:
    systems = sorted({row.system for row in results})
    summary: dict[str, Any] = {"systems": {}, "results": [asdict(row) for row in results]}
    for system in systems:
        rows = [row for row in results if row.system == system]
        passed = sum(1 for row in rows if row.ok)
        summary["systems"][system] = {
            "passed": passed,
            "total": len(rows),
            "pass_rate": round((passed / len(rows)) * 100, 1) if rows else 0.0,
            "avg_latency_ms": round(sum(row.latency_ms for row in rows) / len(rows), 1) if rows else 0.0,
        }
    law_rows = [row for row in results if row.system == "law_open_data"]
    by_category: dict[str, dict[str, float | int]] = {}
    for category in sorted({row.category for row in law_rows}):
        category_rows = [row for row in law_rows if row.category == category]
        passed = sum(1 for row in category_rows if row.ok)
        by_category[category] = {
            "passed": passed,
            "total": len(category_rows),
            "pass_rate": round((passed / len(category_rows)) * 100, 1)
            if category_rows
            else 0.0,
        }
    summary["law_open_data_only"] = {
        "passed": sum(1 for row in law_rows if row.ok),
        "total": len(law_rows),
        "pass_rate": round((sum(1 for row in law_rows if row.ok) / len(law_rows)) * 100, 1)
        if law_rows
        else 0.0,
        "by_category": by_category,
    }
    return summary


def print_report(summary: dict[str, Any]) -> None:
    print("| system | passed | total | pass_rate | avg_latency_ms |")
    print("|---|---:|---:|---:|---:|")
    for system, row in summary["systems"].items():
        print(
            f"| {system} | {row['passed']} | {row['total']} | {row['pass_rate']}% | {row['avg_latency_ms']} |"
        )
    print()
    print("| law_open_data_only | passed | total | pass_rate |")
    print("|---|---:|---:|---:|")
    overall = summary["law_open_data_only"]
    print(f"| overall | {overall['passed']} | {overall['total']} | {overall['pass_rate']}% |")
    for category, row in overall["by_category"].items():
        print(f"| {category} | {row['passed']} | {row['total']} | {row['pass_rate']}% |")
    print()
    print("| query | category | law_open_data | korean-law-mcp | beopmang |")
    print("|---|---|---|---|---|")
    queries = [case.query for case in CASES]
    for query in queries:
        query_rows = {row["system"]: row for row in summary["results"] if row["query"] == query}
        print(
            f"| {query} | {query_rows['law_open_data']['category']} | "
            f"{'OK' if query_rows['law_open_data']['ok'] else 'MISS'} ({query_rows['law_open_data']['status']}) | "
            f"{'OK' if query_rows['korean-law-mcp']['ok'] else 'MISS'} ({query_rows['korean-law-mcp']['status']}) | "
            f"{'OK' if query_rows['beopmang']['ok'] else 'MISS'} ({query_rows['beopmang']['status']}) |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-way exact lookup benchmark.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results: list[SystemResult] = []
    for case in CASES:
        results.append(run_law_open_data(case, args.root))
        results.append(run_korean_law(case))
        results.append(run_beopmang(case))

    summary = build_summary(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(summary)


if __name__ == "__main__":
    main()
