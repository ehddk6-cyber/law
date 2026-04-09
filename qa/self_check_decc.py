from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.retrievers import DeccRetriever
from qa.self_check_utils import run_retriever_check


EXPECTED = {
    "2000-04033": ("decc:17109", "2000-04033", "재확인신체검사등외판정처분취소청구"),
    "2001-05221": ("decc:9858", "2001-05221", "산업재해보상보험료납부독촉처분등취소청구"),
}


def main() -> None:
    run_retriever_check(
        description="Run a minimal decc lookup self-check.",
        retriever_class=DeccRetriever,
        db_flag="--decc-db",
        db_default=Path(__file__).resolve().parents[1] / ".artifacts" / "decc" / "decc.sqlite3",
        expected=EXPECTED,
        extract_key=lambda r, q: r.exact_case(q),
        format_query=lambda row: (row["doc_id"], row["case_no"], row["case_name"]),
    )


if __name__ == "__main__":
    main()
