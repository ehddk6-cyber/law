from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.answering import GroundedAnswerer


@dataclass
class EvalCase:
    name: str
    query: str
    expected_status: str
    expected_grounded: bool
    expected_doc_id: str | None = None
    expected_recommended: list[str] | None = None


CASES = [
    EvalCase("law_exact", "행정심판법 제18조", "grounded", True, "law:001363"),
    EvalCase("acr_exact", "결정문일련번호 23", "grounded", True, "acr:23"),
    EvalCase("prec_exact", "84누180", "grounded", True, "prec:100006"),
    EvalCase("decc_exact", "2000-04033", "grounded", True, "decc:17109"),
    EvalCase("detc_exact", "2004헌마275", "grounded", True, "detc:10026"),
    EvalCase("expc_exact", "05-0096", "grounded", True, "expc:313107"),
    EvalCase("review_only", "기소유예처분취소", "needs_review", False),
    EvalCase(
        "ox_positive",
        "행정심판법 제18조는 대리인의 선임에 관한 조항이다. OX",
        "candidate_answer",
        True,
        expected_recommended=["O"],
    ),
    EvalCase(
        "ox_negative",
        "행정심판법 제18조는 대리인의 선임에 관한 조항이 아니다. OX",
        "candidate_answer",
        True,
        expected_recommended=["X"],
    ),
    EvalCase(
        "wrong_count",
        """다음 중 틀린 것의 개수는?
ㄱ. 행정심판법 제18조
ㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.
ㄷ. 2004헌마275
ㄹ. 84누180
1) 1개
2) 2개
3) 3개
4) 4개""",
        "candidate_answer",
        True,
        expected_recommended=["1)"],
    ),
    EvalCase("prec_fts", "양도소득세부과처분취소", "needs_review", False),
    EvalCase("decc_fts", "산업재해보상보험", "needs_review", False),
    EvalCase("detc_fts", "헌법소원", "needs_review", False),
    EvalCase("expc_fts", "산지관리법", "needs_review", False),
    EvalCase("law_fts", "재외국민등록", "needs_review", False),
]


def run_case(answerer: GroundedAnswerer, case: EvalCase, response_mode: str) -> dict:
    packet = answerer.answer(case.query, limit=3, response_mode=response_mode)
    result = {
        "name": case.name,
        "query": case.query,
        "status": packet.status,
        "grounded": packet.grounded,
        "doc_id": packet.doc_id,
        "final_answer_source": packet.final_answer_source,
        "llm_provider": packet.llm_provider,
        "recommended_options": (packet.exam_assessment or {}).get(
            "recommended_options"
        ),
    }
    ok = (
        packet.status == case.expected_status
        and packet.grounded is case.expected_grounded
        and (case.expected_doc_id is None or packet.doc_id == case.expected_doc_id)
        and (
            case.expected_recommended is None
            or (
                (packet.exam_assessment or {}).get("recommended_options")
                == case.expected_recommended
            )
        )
    )
    result["ok"] = ok
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a compact regression report for legal QA."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--response-mode",
        choices=["grounded", "llm_preferred", "llm_only"],
        default="grounded",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    answerer = GroundedAnswerer(args.root)
    rows = [run_case(answerer, case, args.response_mode) for case in CASES]
    passed = sum(1 for row in rows if row["ok"])
    summary = {
        "response_mode": args.response_mode,
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round((passed / len(rows)) * 100, 2) if rows else 0.0,
        "cases": rows,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"response_mode={summary['response_mode']}")
    print(f"passed={summary['passed']}/{summary['total']}")
    print(f"pass_rate={summary['pass_rate']}")
    for row in rows:
        print(
            f"{'OK' if row['ok'] else 'FAIL'} "
            f"name={row['name']} status={row['status']} grounded={row['grounded']} "
            f"doc_id={row['doc_id']} final_answer_source={row['final_answer_source']}"
        )

    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
