from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.unified import UnifiedSearcher


EXPECTED = {
    "결정문일련번호 23": ("acr", "acr:23"),
    "행정심판법 제18조": ("law", "law:001363"),
    "84누180": ("prec", "prec:100006"),
    "2000-04033": ("decc", "decc:17109"),
    "2004헌마275": ("detc", "detc:10026"),
    "05-0096": ("expc", "expc:313107"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified search self-check.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    searcher = UnifiedSearcher(args.root)
    failures = 0
    for query, expected in EXPECTED.items():
        _, results = searcher.search(query, limit=3)
        actual = None
        if results:
            actual = (results[0].source_type, results[0].payload["doc_id"])
        status = "OK" if actual == expected else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"{status} query={query} expected={expected} actual={actual}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
