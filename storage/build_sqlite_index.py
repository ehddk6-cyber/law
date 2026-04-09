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

CREATE TABLE IF NOT EXISTS docs (
    doc_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    complaint_title TEXT NOT NULL,
    complaint_title_norm TEXT NOT NULL,
    case_no TEXT NOT NULL,
    case_no_norm TEXT NOT NULL,
    decision_date TEXT,
    agency TEXT NOT NULL,
    agency_norm TEXT NOT NULL,
    meeting_type TEXT NOT NULL,
    decision_type TEXT NOT NULL,
    applicant_masked TEXT NOT NULL,
    respondent_masked TEXT NOT NULL,
    source_path TEXT NOT NULL,
    list_page INTEGER NOT NULL,
    raw_meta_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_type TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    text_clean TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES docs(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_docs_source_id ON docs(source_id);
CREATE INDEX IF NOT EXISTS idx_docs_case_no ON docs(case_no);
CREATE INDEX IF NOT EXISTS idx_docs_decision_date ON docs(decision_date);
CREATE INDEX IF NOT EXISTS idx_docs_title_norm ON docs(title_norm);
CREATE INDEX IF NOT EXISTS idx_docs_case_no_norm ON docs(case_no_norm);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    doc_id UNINDEXED,
    title,
    complaint_title,
    case_no,
    agency,
    content=''
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    heading,
    text_clean,
    content=''
);
"""


def build_sqlite_index(documents_path: Path, db_path: Path) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM docs")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM docs_fts")
        conn.execute("DELETE FROM chunks_fts")

        doc_count = 0
        chunk_count = 0
        with documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                doc = json.loads(line)
                conn.execute(
                    """
                    INSERT INTO docs (
                        doc_id, category, doc_type, source_id, title, title_norm,
                        complaint_title, complaint_title_norm, case_no, case_no_norm,
                        decision_date, agency, agency_norm, meeting_type, decision_type,
                        applicant_masked, respondent_masked, source_path, list_page, raw_meta_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc["doc_id"],
                        doc["category"],
                        doc["doc_type"],
                        doc["source_id"],
                        doc["title"],
                        doc["title_norm"],
                        doc["complaint_title"],
                        doc["complaint_title_norm"],
                        doc["case_no"],
                        doc["case_no_norm"],
                        doc["decision_date"],
                        doc["agency"],
                        doc["agency_norm"],
                        doc["meeting_type"],
                        doc["decision_type"],
                        doc["applicant_masked"],
                        doc["respondent_masked"],
                        doc["source_path"],
                        doc["list_page"],
                        json.dumps(doc["raw_meta"], ensure_ascii=False),
                    ),
                )
                conn.execute(
                    "INSERT INTO docs_fts (doc_id, title, complaint_title, case_no, agency) VALUES (?, ?, ?, ?, ?)",
                    (doc["doc_id"], doc["title"], doc["complaint_title"], doc["case_no"], doc["agency"]),
                )
                doc_count += 1
                for chunk in doc["chunks"]:
                    conn.execute(
                        """
                        INSERT INTO chunks (
                            chunk_id, doc_id, chunk_type, chunk_order, heading, text, text_clean, citations_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk["chunk_id"],
                            chunk["doc_id"],
                            chunk["chunk_type"],
                            chunk["chunk_order"],
                            chunk["heading"],
                            chunk["text"],
                            chunk["text_clean"],
                            json.dumps(chunk["citations"], ensure_ascii=False),
                        ),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts (chunk_id, doc_id, heading, text_clean) VALUES (?, ?, ?, ?)",
                        (chunk["chunk_id"], chunk["doc_id"], chunk["heading"], chunk["text_clean"]),
                    )
                    chunk_count += 1
        conn.commit()
        return doc_count, chunk_count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SQLite exact and FTS indexes.")
    parser.add_argument("--documents", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "acr" / "documents.jsonl")
    parser.add_argument("--db", type=Path, default=Path(__file__).resolve().parents[1] / ".artifacts" / "acr" / "acr.sqlite3")
    args = parser.parse_args()

    docs, chunks = build_sqlite_index(args.documents, args.db)
    print(f"docs={docs}")
    print(f"chunks={chunks}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
