from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from normalizers.text import normalize_display_text


VECTOR_DIM = 512


def _tokenize(text: str) -> list[str]:
    normalized = normalize_display_text(text).lower()
    return re.findall(r"[0-9a-z가-힣]+", normalized)


def _hash_vector(text: str, dim: int = VECTOR_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in _tokenize(text):
        vector[hash(token) % dim] += 1.0
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    return vector


def _extract_text(record: dict) -> str:
    if "chunks" in record and record["chunks"]:
        parts = [c.get("text_clean", "") or c.get("text", "") for c in record["chunks"]]
        return " ".join(p for p in parts if p)
    if "articles" in record and record["articles"]:
        parts = [
            a.get("article_text", "") or a.get("article_title", "")
            for a in record["articles"]
        ]
        return " ".join(p for p in parts if p)
    for key in (
        "article_text",
        "holding",
        "issue",
        "content",
        "order_text",
        "claim_text",
        "reason_text",
        "answer_text",
        "query_summary",
        "decision_summary",
        "article_title",
        "title",
        "case_name",
    ):
        val = record.get(key, "")
        if val:
            return str(val)
    return ""


def build_vector_artifacts(
    documents_path: Path, output_dir: Path, category: str | None = None
) -> tuple[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path = output_dir / "chunks.jsonl"
    vectors_path = output_dir / "vectors.npy"
    meta_path = output_dir / "metadata.json"

    rows: list[dict] = []
    vectors: list[np.ndarray] = []
    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            doc = json.loads(line)

            if "chunks" in doc and doc["chunks"]:
                for chunk in doc["chunks"]:
                    text = chunk.get("text_clean", "") or chunk.get("text", "")
                    if not text:
                        continue
                    row = {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "doc_id": doc.get("doc_id", ""),
                        "source_id": doc.get("source_id", doc.get("serial_no", "")),
                        "case_no": doc.get("case_no", doc.get("issue_no", "")),
                        "title": doc.get(
                            "title", doc.get("case_name", doc.get("law_name", ""))
                        ),
                        "decision_date": doc.get("decision_date"),
                        "agency": doc.get("agency", ""),
                        "meeting_type": doc.get("meeting_type", ""),
                        "chunk_type": chunk.get("chunk_type", ""),
                        "heading": chunk.get("heading", ""),
                        "text": text,
                    }
                    rows.append(row)
                    vectors.append(_hash_vector(text))
            elif "articles" in doc and doc["articles"]:
                for article in doc["articles"]:
                    text = article.get("article_text", "")
                    if not text:
                        continue
                    row = {
                        "chunk_id": f"{doc.get('doc_id', '')}:{article.get('article_no', '')}",
                        "doc_id": doc.get("doc_id", ""),
                        "source_id": doc.get("law_id", ""),
                        "case_no": article.get("article_no", ""),
                        "title": doc.get("law_name", ""),
                        "decision_date": doc.get("effective_date"),
                        "agency": doc.get("ministry", ""),
                        "meeting_type": doc.get("law_type", ""),
                        "chunk_type": "article",
                        "heading": article.get("article_title", ""),
                        "text": text,
                    }
                    rows.append(row)
                    vectors.append(_hash_vector(text))
            else:
                text = _extract_text(doc)
                if not text:
                    continue
                row = {
                    "chunk_id": doc.get("doc_id", ""),
                    "doc_id": doc.get("doc_id", ""),
                    "source_id": doc.get("source_id", doc.get("serial_no", "")),
                    "case_no": doc.get("case_no", doc.get("issue_no", "")),
                    "title": doc.get(
                        "title", doc.get("case_name", doc.get("law_name", ""))
                    ),
                    "decision_date": doc.get("decision_date"),
                    "agency": doc.get("agency", ""),
                    "meeting_type": doc.get("meeting_type", ""),
                    "chunk_type": "body",
                    "heading": "",
                    "text": text,
                }
                rows.append(row)
                vectors.append(_hash_vector(text))

    with export_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if vectors:
        np.save(vectors_path, np.vstack(vectors))
    else:
        np.save(vectors_path, np.zeros((0, VECTOR_DIM), dtype=np.float32))

    collection_name = f"{category or 'unknown'}_documents"
    meta_path.write_text(
        json.dumps(
            {
                "backend": "local_hashed_vector",
                "collection_name": collection_name,
                "dimension": VECTOR_DIM,
                "rows": len(rows),
                "note": "Local fallback vector index. Swap to Chroma when chromadb is available.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return len(rows), vectors_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk-level vector artifacts.")
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".artifacts"
        / "acr"
        / "documents.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / ".artifacts" / "acr" / "vector",
    )
    parser.add_argument("--category", type=str, default="acr")
    args = parser.parse_args()

    rows, vectors_path = build_vector_artifacts(
        args.documents, args.output_dir, category=args.category
    )
    print(f"vector_rows={rows}")
    print(f"vectors={vectors_path}")


if __name__ == "__main__":
    main()
