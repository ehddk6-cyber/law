from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import ExpcRetriever
from qa.self_check_utils import run_retriever_check


EXPECTED = {
    "05-0096": ("expc:313107", "05-0096"),
    "08-0204": ("expc:311671", "08-0204"),
}


def main() -> None:
    run_retriever_check(
        description="Run a minimal expc lookup self-check.",
        retriever_class=ExpcRetriever,
        db_flag="--expc-db",
        db_default=Path(__file__).resolve().parents[1] / ".artifacts" / "expc" / "expc.sqlite3",
        expected=EXPECTED,
        extract_key=lambda r, q: r.exact_issue(q),
        format_query=lambda row: (row["doc_id"], row["issue_no"]),
    )


if __name__ == "__main__":
    main()
