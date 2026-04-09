from __future__ import annotations

import argparse
from pathlib import Path


def run_retriever_check(
    description: str,
    retriever_class,
    db_flag: str,
    db_default: Path,
    expected: dict,
    extract_key,
    format_query,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(db_flag, type=Path, default=db_default)
    args = parser.parse_args()
    retriever = retriever_class(getattr(args, db_flag.lstrip("-").replace("-", "_")))
    failures = 0
    for query, exp in expected.items():
        rows = extract_key(retriever, query)
        actual = None
        if rows:
            row = rows[0]
            actual = format_query(row)
        status = "OK" if actual == exp else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"{status} query={query} expected={exp} actual={actual}")
    raise SystemExit(1 if failures else 0)
