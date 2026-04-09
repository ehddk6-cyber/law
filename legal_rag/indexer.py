from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from .corpus import ChunkRecord, chunk_acr_document, inspect_category_structures, load_acr_documents
from .text import canonical_text, hash_embedding, semantic_terms


DEFAULT_INDEX_DIR = Path("artifacts/legal_rag")


def build_acr_indexes(base_dir: Path, index_dir: Path = DEFAULT_INDEX_DIR) -> dict:
    documents = load_acr_documents(base_dir)
    chunks: List[ChunkRecord] = []
    for document in documents:
        chunks.extend(chunk_acr_document(document))

    index_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(index_dir / "acr_documents.jsonl", (document.to_dict() for document in documents))
    _write_jsonl(index_dir / "acr_chunks.jsonl", (chunk.to_dict() for chunk in chunks))

    exact_index = {}
    for document in documents:
        for value in document.exact_fields:
            key = canonical_text(value)
            if key:
                exact_index.setdefault(key, []).append(document.doc_id)
    (index_dir / "acr_exact_index.json").write_text(
        json.dumps(exact_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _build_keyword_index(index_dir / "acr_keyword.sqlite3", documents)
    vector_manifest = _build_vector_index(index_dir / "acr_vector_index.json", chunks)

    structure_report = inspect_category_structures(base_dir)
    (index_dir / "category_structure_report.json").write_text(
        json.dumps(structure_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "category": "acr",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedding_backend": vector_manifest["backend"],
        "embedding_dims": vector_manifest["dims"],
        "files": {
            "documents": "acr_documents.jsonl",
            "chunks": "acr_chunks.jsonl",
            "exact_index": "acr_exact_index.json",
            "keyword_index": "acr_keyword.sqlite3",
            "vector_index": "acr_vector_index.json",
            "category_report": "category_structure_report.json",
        },
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _build_keyword_index(path: Path, documents) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE VIRTUAL TABLE acr_fts USING fts5(
            doc_id UNINDEXED,
            serial_number UNINDEXED,
            title,
            agenda_number,
            display_name,
            summary_text,
            order_text,
            reason_text,
            search_text,
            tokenize = 'unicode61'
        )
        """
    )
    for document in documents:
        con.execute(
            """
            INSERT INTO acr_fts (
                doc_id, serial_number, title, agenda_number, display_name,
                summary_text, order_text, reason_text, search_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document.doc_id,
                document.serial_number,
                document.title,
                document.agenda_number,
                document.display_name,
                document.summary_text,
                document.order_text,
                document.reason_text,
                document.search_text,
            ),
        )
    con.commit()
    con.close()


def _build_vector_index(path: Path, chunks: List[ChunkRecord]) -> dict:
    idf_counter = Counter()
    for chunk in chunks:
        idf_counter.update(set(semantic_terms(chunk.text)))
    total = max(len(chunks), 1)
    idf = {
        term: math.log(1.0 + total / (1.0 + frequency))
        for term, frequency in idf_counter.items()
    }
    payload = {
        "backend": "hashing-fallback",
        "dims": 768,
        "idf": idf,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "serial_number": chunk.serial_number,
                "chunk_index": chunk.chunk_index,
                "label": chunk.label,
                "text": chunk.text,
                "embedding": hash_embedding(chunk.text, dims=768, idf=idf),
            }
            for chunk in chunks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"backend": payload["backend"], "dims": payload["dims"]}

