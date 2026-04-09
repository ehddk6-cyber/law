from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.cli import format_article_no
from qa.unified import UnifiedSearcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified legal search across acr/law/prec/decc.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    searcher = UnifiedSearcher(args.root)
    decision, results = searcher.search(args.query, limit=args.limit)
    print(f"route={decision.strategy}")
    if not results:
        print("[NOT_FOUND]")
        return

    for index, result in enumerate(results, start=1):
        print(f"[{index}] source_type={result.source_type} search_path={result.search_path} score={result.score:.4f}")
        payload = result.payload
        if result.source_type == "law":
            print(f"doc_id={payload['doc_id']}")
            print(f"law_name={payload['law_name']}")
            print(f"article_no={format_article_no(payload['article_no'])}")
            print(f"article_title={payload['article_title']}")
            print(f"text={payload['article_text'][:300]}")
            continue
        if result.source_type == "prec":
            print(f"doc_id={payload['doc_id']}")
            print(f"case_no={payload['case_no']}")
            print(f"case_name={payload['case_name']}")
            print(f"court_name={payload['court_name']}")
            print(f"holding={payload['holding'][:300]}")
            continue
        if result.source_type == "decc":
            print(f"doc_id={payload['doc_id']}")
            print(f"case_no={payload['case_no']}")
            print(f"case_name={payload['case_name']}")
            print(f"agency={payload['agency']}")
            print(f"order={payload['order_text'][:300]}")
            continue
        if result.source_type == "detc":
            print(f"doc_id={payload['doc_id']}")
            print(f"case_no={payload['case_no']}")
            print(f"case_name={payload['case_name']}")
            print(f"case_type={payload['case_type']}")
            print(f"summary={payload['decision_summary'][:300]}")
            continue
        if result.source_type == "expc":
            print(f"doc_id={payload['doc_id']}")
            print(f"issue_no={payload['issue_no']}")
            print(f"title={payload['title']}")
            print(f"agency={payload['agency']}")
            print(f"answer={payload['answer_text'][:300]}")
            continue

        print(f"doc_id={payload['doc_id']}")
        print(f"title={payload['title']}")
        print(f"case_no={payload['case_no']}")
        print(f"decision_date={payload['decision_date']}")
        for chunk in payload["chunks"][:2]:
            print(f"  - {chunk['chunk_type']} | {chunk['heading']}: {chunk['text'][:160]}")


if __name__ == "__main__":
    main()
