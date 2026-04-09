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

CREATE TABLE IF NOT EXISTS expc_cases (
    doc_id TEXT PRIMARY KEY,
    serial_no TEXT NOT NULL,
    issue_no TEXT NOT NULL,
    issue_no_norm TEXT NOT NULL,
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    decision_date TEXT,
    agency TEXT NOT NULL,
    query_agency TEXT NOT NULL,
    query_summary TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    reason_text TEXT NOT NULL,
    source_path TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_expc_serial_no ON expc_cases(serial_no);
CREATE INDEX IF NOT EXISTS idx_expc_issue_no_norm ON expc_cases(issue_no_norm);
CREATE INDEX IF NOT EXISTS idx_expc_title_norm ON expc_cases(title_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS expc_cases_fts USING fts5(
    doc_id UNINDEXED,
    issue_no,
    title,
    query_summary,
    answer_text,
    content=''
);
"""


def build_expc_sqlite_index(documents_path: Path, db_path: Path) -> int:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM expc_cases")
        conn.execute("DELETE FROM expc_cases_fts")
        count = 0
        with documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                conn.execute(
                    """
                    INSERT INTO expc_cases (
                        doc_id, serial_no, issue_no, issue_no_norm, title, title_norm,
                        decision_date, agency, query_agency, query_summary, answer_text,
                        reason_text, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["doc_id"],
                        record["serial_no"],
                        record["issue_no"],
                        record["issue_no_norm"],
                        record["title"],
                        record["title_norm"],
                        record["decision_date"],
                        record["agency"],
                        record["query_agency"],
                        record["query_summary"],
                        record["answer_text"],
                        record["reason_text"],
                        record["source_path"],
                    ),
                )
                conn.execute(
                    "INSERT INTO expc_cases_fts (doc_id, issue_no, title, query_summary, answer_text) VALUES (?, ?, ?, ?, ?)",
                    (
                        record["doc_id"],
                        record["issue_no"],
                        record["title"],
                        record["query_summary"],
                        record["answer_text"],
                    ),
                )
                count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expc exact lookup index.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "expc"
        / "documents.jsonl",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "expc"
        / "expc.sqlite3",
    )
    args = parser.parse_args()
    count = build_expc_sqlite_index(args.documents, args.db)
    print(f"expc_cases={count}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
