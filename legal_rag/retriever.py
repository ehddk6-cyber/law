from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .text import canonical_text, cosine_similarity, hash_embedding, normalize_text, tokenize


@dataclass
class Candidate:
    doc_id: str
    serial_number: str
    title: str
    stage: str
    score: float
    reason: str
    extra: dict


class AcrHybridRetriever:
    def __init__(self, base_dir: Path, index_dir: Path) -> None:
        self.base_dir = base_dir
        self.index_dir = index_dir
        self.documents = self._load_jsonl(index_dir / "acr_documents.jsonl")
        self.document_map = {row["doc_id"]: row for row in self.documents}
        self.exact_index = json.loads((index_dir / "acr_exact_index.json").read_text(encoding="utf-8"))
        self.vector_index = json.loads((index_dir / "acr_vector_index.json").read_text(encoding="utf-8"))

    @staticmethod
    def _load_jsonl(path: Path) -> List[dict]:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
        return rows

    def search(self, query: str, *, limit: int = 3) -> dict:
        exact_hits = self._exact_search(query)
        keyword_hits = self._keyword_search(query, limit=20)
        vector_hits = self._vector_search(query, limit=20)
        reranked = self._rerank(query, exact_hits, keyword_hits, vector_hits)
        top_results = reranked[:limit]
        if not top_results:
            return {"query": query, "results": [], "status": "[NOT_FOUND]"}
        return {"query": query, "results": top_results, "status": "ok"}

    def _exact_search(self, query: str) -> List[Candidate]:
        key = canonical_text(query)
        doc_ids = self.exact_index.get(key, [])
        candidates = []
        for rank, doc_id in enumerate(doc_ids, start=1):
            document = self.document_map[doc_id]
            candidates.append(
                Candidate(
                    doc_id=doc_id,
                    serial_number=document["serial_number"],
                    title=document["title"],
                    stage="exact",
                    score=1.0 - (rank - 1) * 0.01,
                    reason=f"exact:{query}",
                    extra={"matched_key": key},
                )
            )
        return candidates

    def _keyword_search(self, query: str, *, limit: int) -> List[Candidate]:
        db_path = self.index_dir / "acr_keyword.sqlite3"
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        match_query = " OR ".join(tokenize(query)) or normalize_text(query)
        rows = con.execute(
            """
            SELECT doc_id, serial_number, title, bm25(acr_fts) AS rank
            FROM acr_fts
            WHERE acr_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
        con.close()
        results = []
        for row in rows:
            score = 1.0 / (1.0 + abs(row["rank"]))
            results.append(
                Candidate(
                    doc_id=row["doc_id"],
                    serial_number=row["serial_number"],
                    title=row["title"],
                    stage="keyword",
                    score=score,
                    reason=f"fts5:{match_query}",
                    extra={"bm25_rank": row["rank"]},
                )
            )
        return results

    def _vector_search(self, query: str, *, limit: int) -> List[Candidate]:
        idf = self.vector_index.get("idf", {})
        query_embedding = hash_embedding(query, dims=self.vector_index["dims"], idf=idf)
        chunk_scores = []
        for chunk in self.vector_index["chunks"]:
            score = cosine_similarity(query_embedding, {int(k): v for k, v in chunk["embedding"].items()})
            if score > 0:
                chunk_scores.append((score, chunk))
        chunk_scores.sort(key=lambda item: item[0], reverse=True)
        doc_best: Dict[str, tuple] = {}
        for score, chunk in chunk_scores:
            current = doc_best.get(chunk["doc_id"])
            if current is None or score > current[0]:
                doc_best[chunk["doc_id"]] = (score, chunk)
        results = []
        for score, chunk in sorted(doc_best.values(), key=lambda item: item[0], reverse=True)[:limit]:
            document = self.document_map[chunk["doc_id"]]
            results.append(
                Candidate(
                    doc_id=chunk["doc_id"],
                    serial_number=document["serial_number"],
                    title=document["title"],
                    stage="vector",
                    score=score,
                    reason=f"vector:{chunk['label']}",
                    extra={"chunk_id": chunk["chunk_id"], "chunk_label": chunk["label"]},
                )
            )
        return results

    def _rerank(
        self,
        query: str,
        exact_hits: List[Candidate],
        keyword_hits: List[Candidate],
        vector_hits: List[Candidate],
    ) -> List[dict]:
        merged: Dict[str, dict] = {}
        query_key = canonical_text(query)
        for candidate in exact_hits + keyword_hits + vector_hits:
            row = merged.setdefault(
                candidate.doc_id,
                {
                    "doc_id": candidate.doc_id,
                    "serial_number": candidate.serial_number,
                    "title": candidate.title,
                    "score_breakdown": {"exact": 0.0, "keyword": 0.0, "vector": 0.0, "boost": 0.0},
                    "stages": [],
                    "reasons": [],
                },
            )
            row["score_breakdown"][candidate.stage] = max(row["score_breakdown"][candidate.stage], candidate.score)
            if candidate.stage not in row["stages"]:
                row["stages"].append(candidate.stage)
            row["reasons"].append(candidate.reason)

        query_tokens = set(tokenize(query))
        for row in merged.values():
            document = self.document_map[row["doc_id"]]
            title_key = canonical_text(document["title"])
            if query_key and (query_key == title_key or query_key == canonical_text(document["serial_number"])):
                row["score_breakdown"]["boost"] += 1.0
            title_tokens = set(tokenize(document["title"]))
            overlap = len(query_tokens & title_tokens)
            if overlap:
                row["score_breakdown"]["boost"] += min(0.45, overlap * 0.08)
            row["score"] = (
                row["score_breakdown"]["exact"] * 5.0
                + row["score_breakdown"]["keyword"] * 2.5
                + row["score_breakdown"]["vector"] * 1.5
                + row["score_breakdown"]["boost"]
            )

        ordered = sorted(
            merged.values(),
            key=lambda row: (row["score"], row["score_breakdown"]["exact"], row["score_breakdown"]["keyword"]),
            reverse=True,
        )
        return ordered

