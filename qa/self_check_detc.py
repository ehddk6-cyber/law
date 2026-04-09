from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import DetcRetriever
from qa.self_check_utils import run_retriever_check


EXPECTED = {
    "2004헌마275": ("detc:10026", "2004헌마275"),
    "95헌사164": ("detc:10049", "95헌사164"),
}


def main() -> None:
    run_retriever_check(
        description="Run a minimal detc lookup self-check.",
        retriever_class=DetcRetriever,
        db_flag="--detc-db",
        db_default=Path(__file__).resolve().parents[1] / ".artifacts" / "detc" / "detc.sqlite3",
        expected=EXPECTED,
        extract_key=lambda r, q: r.exact_case(q),
        format_query=lambda row: (row["doc_id"], row["case_no"]),
    )


if __name__ == "__main__":
    main()
