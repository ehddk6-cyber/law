from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import AcrRetriever
from qa.router import route_query


EXPECTED = {
    "결정문일련번호 23": "acr:23",
    "제2022-5소위11-경02호": "acr:23",
    "수사 진행상황 미통지 등 이의": "acr:23",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal retrieval self-check.")
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "acr" / "acr.sqlite3")
    parser.add_argument("--vector-dir", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "acr" / "vector")
    args = parser.parse_args()

    retriever = AcrRetriever(args.db, args.vector_dir)
    failures = 0
    for query, expected_doc_id in EXPECTED.items():
        results = retriever.search(route_query(query), limit=3)
        actual = results[0].doc_id if results else None
        status = "OK" if actual == expected_doc_id else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"{status} query={query} expected={expected_doc_id} actual={actual}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
