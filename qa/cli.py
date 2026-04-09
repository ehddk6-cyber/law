from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import (
    AcrRetriever,
    DeccRetriever,
    DetcRetriever,
    ExpcRetriever,
    LawRetriever,
    PrecRetriever,
)
from qa.router import route_query
from qa.legal_analysis import (
    analyze_law_query,
    build_quasi_provision_chain,
    detect_erroneous_omission,
    format_analysis_text,
    format_analysis_json,
    generate_ox_quiz,
)


ARTIFACTS = Path(__file__).resolve().parents[1] / ".artifacts"


def format_article_no(article_no: str) -> str:
    if "의" in article_no:
        main, sub = article_no.split("의", 1)
        return f"제{main}조의{sub}"
    return f"제{article_no}조"


def _handle_search(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / ".artifacts"

    retriever = AcrRetriever(args.db, args.vector_dir)
    law_retriever = LawRetriever(args.law_db, artifacts / "law" / "vector")
    prec_retriever = PrecRetriever(args.prec_db, artifacts / "prec" / "vector")
    decc_retriever = DeccRetriever(args.decc_db, artifacts / "decc" / "vector")
    detc_retriever = DetcRetriever(args.detc_db, artifacts / "detc" / "vector")
    expc_retriever = ExpcRetriever(args.expc_db, artifacts / "expc" / "vector")
    decision = route_query(args.query)

    if (
        decision.strategy == "exact_law_article"
        and decision.law_name
        and decision.article_no
    ):
        results = law_retriever.exact_article(
            decision.law_name, decision.article_no, limit=args.limit
        )
        print(f"route={decision.strategy}")
        if not results:
            print("[NOT_FOUND]")
            return
        for index, result in enumerate(results, start=1):
            row = result.payload
            print(
                f"[{index}] search_path={result.search_path} score={result.score:.4f}"
            )
            print(f"doc_id={result.doc_id}")
            print(f"law_id={row['law_id']}")
            print(f"law_name={row['law_name']}")
            print(f"article_no={format_article_no(row['article_no'])}")
            print(f"article_title={row['article_title']}")
            print(f"text={row['article_text'][:500]}")
        return

    if decision.strategy == "exact_precedent_case" and decision.precedent_case_no:
        results = prec_retriever.exact_case(
            decision.precedent_case_no, limit=args.limit
        )
        print(f"route={decision.strategy}")
        if not results:
            print("[NOT_FOUND]")
            return
        for index, result in enumerate(results, start=1):
            row = result.payload
            print(
                f"[{index}] search_path={result.search_path} score={result.score:.4f}"
            )
            print(f"doc_id={result.doc_id}")
            print(f"serial_no={row['serial_no']}")
            print(f"case_no={row['case_no']}")
            print(f"case_name={row['case_name']}")
            print(f"court_name={row['court_name']}")
            print(f"decision_date={row['decision_date']}")
            print(f"issue={row['issue'][:300]}")
            print(f"holding={row['holding'][:300]}")
        return

    if decision.strategy == "exact_decc_case" and decision.decc_case_no:
        results = decc_retriever.exact_case(decision.decc_case_no, limit=args.limit)
        print(f"route={decision.strategy}")
        if not results:
            print("[NOT_FOUND]")
            return
        for index, result in enumerate(results, start=1):
            row = result.payload
            print(
                f"[{index}] search_path={result.search_path} score={result.score:.4f}"
            )
            print(f"doc_id={result.doc_id}")
            print(f"serial_no={row['serial_no']}")
            print(f"case_no={row['case_no']}")
            print(f"case_name={row['case_name']}")
            print(f"agency={row['agency']}")
            print(f"decision_date={row['decision_date']}")
            print(f"order={row['order_text'][:300]}")
            print(f"claim={row['claim_text'][:300]}")
        return

    if decision.strategy == "exact_detc_case" and decision.detc_case_no:
        results = detc_retriever.exact_case(decision.detc_case_no, limit=args.limit)
        print(f"route={decision.strategy}")
        if not results:
            print("[NOT_FOUND]")
            return
        for index, result in enumerate(results, start=1):
            row = result.payload
            print(
                f"[{index}] search_path={result.search_path} score={result.score:.4f}"
            )
            print(f"doc_id={result.doc_id}")
            print(f"serial_no={row['serial_no']}")
            print(f"case_no={row['case_no']}")
            print(f"case_name={row['case_name']}")
            print(f"case_type={row['case_type']}")
            print(f"decision_date={row['decision_date']}")
            print(f"summary={row['decision_summary'][:300]}")
            print(f"content={row['content'][:300]}")
        return

    if decision.strategy == "exact_expc_issue" and decision.expc_issue_no:
        results = expc_retriever.exact_issue(decision.expc_issue_no, limit=args.limit)
        print(f"route={decision.strategy}")
        if not results:
            print("[NOT_FOUND]")
            return
        for index, result in enumerate(results, start=1):
            row = result.payload
            print(
                f"[{index}] search_path={result.search_path} score={result.score:.4f}"
            )
            print(f"doc_id={result.doc_id}")
            print(f"serial_no={row['serial_no']}")
            print(f"issue_no={row['issue_no']}")
            print(f"title={row['title']}")
            print(f"agency={row['agency']}")
            print(f"decision_date={row['decision_date']}")
            print(f"query_summary={row['query_summary'][:300]}")
            print(f"answer={row['answer_text'][:300]}")
        return

    results = retriever.search(decision, limit=args.limit)

    print(f"route={decision.strategy}")
    if not results:
        print("[NOT_FOUND]")
        return

    for index, result in enumerate(results, start=1):
        print(f"[{index}] search_path={result.search_path} score={result.score:.4f}")
        print(f"doc_id={result.doc_id}")
        print(f"source_id={result.source_id}")
        print(f"title={result.title}")
        print(f"case_no={result.case_no}")
        print(f"decision_date={result.decision_date}")
        payload = result.payload
        for chunk in payload.get("chunks", []):
            print(
                f"  - {chunk.get('chunk_type', '')} | {chunk.get('heading', '')}: {chunk.get('text', '')[:200]}"
            )


def _handle_compare(args: argparse.Namespace) -> None:
    law_db = args.law_db or (ARTIFACTS / "law" / "law.sqlite3")
    vector_dir = ARTIFACTS / "law" / "vector"
    law_retriever = LawRetriever(law_db, vector_dir)

    result = analyze_law_query(args.query, law_retriever)

    if args.format == "json":
        print(json.dumps(format_analysis_json(result), ensure_ascii=False, indent=2))
    else:
        print(format_analysis_text(result))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified legal retrieval CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # search (default behavior)
    search_parser = subparsers.add_parser("search", help="Search legal documents.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument(
        "--db", type=Path, default=ARTIFACTS / "acr" / "acr.sqlite3"
    )
    search_parser.add_argument(
        "--vector-dir", type=Path, default=ARTIFACTS / "acr" / "vector"
    )
    search_parser.add_argument(
        "--law-db", type=Path, default=ARTIFACTS / "law" / "law.sqlite3"
    )
    search_parser.add_argument(
        "--prec-db", type=Path, default=ARTIFACTS / "prec" / "prec.sqlite3"
    )
    search_parser.add_argument(
        "--decc-db", type=Path, default=ARTIFACTS / "decc" / "decc.sqlite3"
    )
    search_parser.add_argument(
        "--detc-db", type=Path, default=ARTIFACTS / "detc" / "detc.sqlite3"
    )
    search_parser.add_argument(
        "--expc-db", type=Path, default=ARTIFACTS / "expc" / "expc.sqlite3"
    )
    search_parser.add_argument("--limit", type=int, default=5)
    search_parser.set_defaults(handler=_handle_search)

    # compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare related law articles. Query must contain law name + article ref, "
        "e.g. '행정소송법 제28조'. Neighboring articles are auto-fetched.",
    )
    compare_parser.add_argument("--query", required=True)
    compare_parser.add_argument("--law-db", type=Path, default=None)
    compare_parser.add_argument("--format", choices=["text", "json"], default="text")
    compare_parser.set_defaults(handler=_handle_compare)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
