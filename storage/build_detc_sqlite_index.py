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

CREATE TABLE IF NOT EXISTS detc_cases (
    doc_id TEXT PRIMARY KEY,
    serial_no TEXT NOT NULL,
    case_no TEXT NOT NULL,
    case_no_norm TEXT NOT NULL,
    case_name TEXT NOT NULL,
    case_name_norm TEXT NOT NULL,
    decision_date TEXT,
    case_type TEXT NOT NULL,
    decision_summary TEXT NOT NULL,
    issue TEXT NOT NULL,
    content TEXT NOT NULL,
    cited_laws TEXT NOT NULL,
    cited_cases TEXT NOT NULL,
    target_provisions TEXT NOT NULL,
    source_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detc_serial_no ON detc_cases(serial_no);
CREATE INDEX IF NOT EXISTS idx_detc_case_no_norm ON detc_cases(case_no_norm);
CREATE INDEX IF NOT EXISTS idx_detc_case_name_norm ON detc_cases(case_name_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS detc_cases_fts USING fts5(
    doc_id UNINDEXED,
    case_no,
    case_name,
    issue,
    decision_summary,
    content=''
);
"""


def build_detc_sqlite_index(documents_path: Path, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM detc_cases")
        conn.execute("DELETE FROM detc_cases_fts")
        count = 0
        with documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                conn.execute(
                    """
                    INSERT INTO detc_cases (
                        doc_id, serial_no, case_no, case_no_norm, case_name, case_name_norm,
                        decision_date, case_type, decision_summary, issue, content,
                        cited_laws, cited_cases, target_provisions, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["doc_id"],
                        record["serial_no"],
                        record["case_no"],
                        record["case_no_norm"],
                        record["case_name"],
                        record["case_name_norm"],
                        record["decision_date"],
                        record["case_type"],
                        record["decision_summary"],
                        record["issue"],
                        record["content"],
                        record["cited_laws"],
                        record["cited_cases"],
                        record["target_provisions"],
                        record["source_path"],
                    ),
                )
                conn.execute(
                    "INSERT INTO detc_cases_fts (doc_id, case_no, case_name, issue, decision_summary) VALUES (?, ?, ?, ?, ?)",
                    (
                        record["doc_id"],
                        record["case_no"],
                        record["case_name"],
                        record["issue"],
                        record["decision_summary"],
                    ),
                )
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build constitutional court exact/fts index.")
    parser.add_argument("--documents", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "detc" / "documents.jsonl")
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "detc" / "detc.sqlite3")
    args = parser.parse_args()
    count = build_detc_sqlite_index(args.documents, args.db)
    print(f"detc_cases={count}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
