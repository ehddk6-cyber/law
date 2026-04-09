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

CREATE TABLE IF NOT EXISTS laws (
    doc_id TEXT PRIMARY KEY,
    law_id TEXT NOT NULL,
    law_name TEXT NOT NULL,
    law_name_norm TEXT NOT NULL,
    short_name TEXT NOT NULL,
    short_name_norm TEXT NOT NULL,
    law_type TEXT NOT NULL,
    ministry TEXT NOT NULL,
    promulgation_date TEXT,
    effective_date TEXT,
    source_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS law_articles (
    article_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    law_id TEXT NOT NULL,
    law_name TEXT NOT NULL,
    law_name_norm TEXT NOT NULL,
    short_name_norm TEXT NOT NULL,
    article_no TEXT NOT NULL,
    article_no_norm TEXT NOT NULL,
    article_key TEXT NOT NULL,
    article_title TEXT NOT NULL,
    article_text TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES laws(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_law_name_norm ON laws(law_name_norm);
CREATE INDEX IF NOT EXISTS idx_law_short_name_norm ON laws(short_name_norm);
CREATE INDEX IF NOT EXISTS idx_article_lookup ON law_articles(law_name_norm, article_no_norm);
CREATE INDEX IF NOT EXISTS idx_article_short_lookup ON law_articles(short_name_norm, article_no_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS law_articles_fts USING fts5(
    article_id UNINDEXED,
    doc_id UNINDEXED,
    law_name,
    article_no,
    article_title,
    article_text,
    content=''
);
"""


def build_law_sqlite_index(documents_path: Path, db_path: Path) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM laws")
        conn.execute("DELETE FROM law_articles")
        conn.execute("INSERT INTO law_articles_fts(law_articles_fts) VALUES ('delete-all')")
        law_count = 0
        article_count = 0
        with documents_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                law = json.loads(line)
                conn.execute(
                    """
                    INSERT INTO laws (
                        doc_id, law_id, law_name, law_name_norm, short_name, short_name_norm,
                        law_type, ministry, promulgation_date, effective_date, source_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        law["doc_id"],
                        law["law_id"],
                        law["law_name"],
                        law["law_name_norm"],
                        law["short_name"],
                        law["short_name_norm"],
                        law["law_type"],
                        law["ministry"],
                        law["promulgation_date"],
                        law["effective_date"],
                        law["source_path"],
                    ),
                )
                law_count += 1
                for article in law["articles"]:
                    article_id = f"{law['doc_id']}:{article['article_key'] or article['article_no']}:{article_count + 1}"
                    conn.execute(
                        """
                        INSERT INTO law_articles (
                            article_id, doc_id, law_id, law_name, law_name_norm, short_name_norm,
                            article_no, article_no_norm, article_key, article_title, article_text
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            article_id,
                            law["doc_id"],
                            law["law_id"],
                            law["law_name"],
                            law["law_name_norm"],
                            law["short_name_norm"],
                            article["article_no"],
                            article["article_no_norm"],
                            article["article_key"],
                            article["article_title"],
                            article["article_text"],
                        ),
                    )
                    conn.execute(
                        "INSERT INTO law_articles_fts (article_id, doc_id, law_name, article_no, article_title, article_text) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            article_id,
                            law["doc_id"],
                            law["law_name"],
                            article["article_no"],
                            article["article_title"],
                            article["article_text"],
                        ),
                    )
                    article_count += 1
        conn.commit()
        return law_count, article_count
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build law exact article lookup index."
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "law"
        / "documents.jsonl",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "law"
        / "law.sqlite3",
    )
    args = parser.parse_args()
    laws, articles = build_law_sqlite_index(args.documents, args.db)
    print(f"laws={laws}")
    print(f"articles={articles}")
    print(f"db={args.db}")


if __name__ == "__main__":
    main()
