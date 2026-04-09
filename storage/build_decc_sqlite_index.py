from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS decc_cases (
    doc_id TEXT PRIMARY KEY,
    serial_no TEXT NOT NULL,
    case_no TEXT NOT NULL,
    case_no_norm TEXT NOT NULL,
    case_name TEXT NOT NULL,
    case_name_norm TEXT NOT NULL,
    decision_date TEXT,
    agency TEXT NOT NULL,
    disposition_agency TEXT NOT NULL,
    decision_type_name TEXT NOT NULL,
    order_text TEXT NOT NULL,
    claim_text TEXT NOT NULL,
    reason_text TEXT NOT NULL,
    source_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decc_serial_no ON decc_cases(serial_no);
CREATE INDEX IF NOT EXISTS idx_decc_case_no_norm ON decc_cases(case_no_norm);
CREATE INDEX IF NOT EXISTS idx_decc_case_name_norm ON decc_cases(case_name_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS decc_cases_fts USING fts5(
    doc_id UNINDEXED,
    case_no,
    case_name,
    order_text,
    claim_text,
    reason_text,
    content=''
);
"""


def build_decc_sqlite_index(documents_path: Path, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM decc_cases")
        conn.execute("DELETE FROM decc_cases_fts")
        count = 0
        with documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                conn.execute(
                    """
                    INSERT INTO decc_cases (
                        doc_id, serial_no, case_no, case_no_norm, case_name, case_name_norm,
                        decision_date, agency, disposition_agency, decision_type_name,
                        order_text, claim_text, reason_text, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["doc_id"],
                        record["serial_no"],
                        record["case_no"],
                        record["case_no_norm"],
                        record["case_name"],
                        record["case_name_norm"],
                        record["decision_date"],
                        record["agency"],
                        record["disposition_agency"],
                        record["decision_type_name"],
                        record["order_text"],
                        record["claim_text"],
                        record["reason_text"],
                        record["source_path"],
                    ),
                )
                conn.execute(
                    "INSERT INTO decc_cases_fts (doc_id, case_no, case_name, order_text, claim_text, reason_text) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record["doc_id"],
                        record["case_no"],
                        record["case_name"],
                        record["order_text"],
                        record["claim_text"],
                        record["reason_text"],
                    ),
                )
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build decc exact lookup index.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "decc"
        / "documents.jsonl",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "decc"
        / "decc.sqlite3",
    )
    args = parser.parse_args()
    count = build_decc_sqlite_index(args.documents, args.db)
    print(f"decc_cases={count}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
