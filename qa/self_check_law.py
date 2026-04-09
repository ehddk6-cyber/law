from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import LawRetriever
from qa.self_check_utils import run_retriever_check


EXPECTED = {
    ("행정심판법", "18"): ("행정심판법", "18", "대리인의 선임"),
    ("행정심판법", "18의2"): ("행정심판법", "18의2", "국선대리인"),
    ("행정소송법", "20"): ("행정소송법", "20", "제소기간"),
}


def main() -> None:
    run_retriever_check(
        description="Run a minimal law article lookup self-check.",
        retriever_class=LawRetriever,
        db_flag="--law-db",
        db_default=Path(__file__).resolve().parents[1] / ".artifacts" / "law" / "law.sqlite3",
        expected=EXPECTED,
        extract_key=lambda r, q: r.exact_article(q[0], q[1]),
        format_query=lambda row: (row["law_name"], row["article_no"], row["article_title"]),
    )


if __name__ == "__main__":
    main()
