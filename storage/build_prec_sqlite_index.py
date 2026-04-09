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

CREATE TABLE IF NOT EXISTS precedents (
    doc_id TEXT PRIMARY KEY,
    serial_no TEXT NOT NULL,
    case_no TEXT NOT NULL,
    case_no_norm TEXT NOT NULL,
    case_name TEXT NOT NULL,
    case_name_norm TEXT NOT NULL,
    court_name TEXT NOT NULL,
    case_type TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    decision_date TEXT,
    issue TEXT NOT NULL,
    holding TEXT NOT NULL,
    cited_laws TEXT NOT NULL,
    cited_cases TEXT NOT NULL,
    content TEXT NOT NULL,
    source_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prec_serial_no ON precedents(serial_no);
CREATE INDEX IF NOT EXISTS idx_prec_case_no_norm ON precedents(case_no_norm);
CREATE INDEX IF NOT EXISTS idx_prec_case_name_norm ON precedents(case_name_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS precedents_fts USING fts5(
    doc_id UNINDEXED,
    case_no,
    case_name,
    issue,
    holding,
    content=''
);
"""


def build_prec_sqlite_index(documents_path: Path, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM precedents")
        conn.execute("DELETE FROM precedents_fts")
        count = 0
        with documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                conn.execute(
                    """
                    INSERT INTO precedents (
                        doc_id, serial_no, case_no, case_no_norm, case_name, case_name_norm,
                        court_name, case_type, decision_type, decision_date, issue, holding,
                        cited_laws, cited_cases, content, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["doc_id"],
                        record["serial_no"],
                        record["case_no"],
                        record["case_no_norm"],
                        record["case_name"],
                        record["case_name_norm"],
                        record["court_name"],
                        record["case_type"],
                        record["decision_type"],
                        record["decision_date"],
                        record["issue"],
                        record["holding"],
                        record["cited_laws"],
                        record["cited_cases"],
                        record["content"],
                        record["source_path"],
                    ),
                )
                conn.execute(
                    "INSERT INTO precedents_fts (doc_id, case_no, case_name, issue, holding) VALUES (?, ?, ?, ?, ?)",
                    (
                        record["doc_id"],
                        record["case_no"],
                        record["case_name"],
                        record["issue"],
                        record["holding"],
                    ),
                )
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build precedent exact/fts index.")
    parser.add_argument("--documents", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "prec" / "documents.jsonl")
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "prec" / "prec.sqlite3")
    args = parser.parse_args()
    count = build_prec_sqlite_index(args.documents, args.db)
    print(f"precedents={count}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
