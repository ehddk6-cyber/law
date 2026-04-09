from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.answering import GroundedAnswerer
from qa.exact_benchmark_three_way import CASES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact verified self-check for law_open_data only."
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    answerer = GroundedAnswerer(args.root)
    failures = 0
    for case in CASES:
        packet = answerer.answer(case.query, limit=3, response_mode="grounded")
        official_ref = next(
            (
                citation.replace("[공식검증] ", "")
                for citation in packet.citations
                if citation.startswith("[공식검증] ")
            ),
            None,
        )
        ok = (
            packet.doc_id == case.expected_doc_id
            and packet.grounded is True
            and packet.status == "grounded"
            and official_ref is not None
        )
        print(
            f"{'OK' if ok else 'FAIL'} query={case.query} "
            f"status={packet.status} grounded={packet.grounded} "
            f"doc_id={packet.doc_id} official_ref={official_ref}"
        )
        if not ok:
            failures += 1

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
