from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import PrecRetriever
from qa.self_check_utils import run_retriever_check


EXPECTED = {
    "84누180": ("prec:100006", "84누180", "양도소득세부과처분취소"),
    "84누323": ("prec:100008", "84누323", "부가가치세부과처분취소"),
}


def main() -> None:
    run_retriever_check(
        description="Run a minimal precedent lookup self-check.",
        retriever_class=PrecRetriever,
        db_flag="--prec-db",
        db_default=Path(__file__).resolve().parents[1] / ".artifacts" / "prec" / "prec.sqlite3",
        expected=EXPECTED,
        extract_key=lambda r, q: r.exact_case(q),
        format_query=lambda row: (row["doc_id"], row["case_no"], row["case_name"]),
    )


if __name__ == "__main__":
    main()
