from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from normalizers.text import normalize_lookup_text
from qa.router import RouteDecision
from storage.build_vector_index import VECTOR_DIM, _hash_vector

try:
    import faiss  # type: ignore[import-untyped]

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


@dataclass
class SearchResult:
    search_path: str
    doc_id: str
    source_id: str
    title: str
    case_no: str
    decision_date: str | None
    score: float
    payload: dict


LEGAL_SYNONYMS: dict[str, str] = {
    "취소": "취소소송",
    "취소소송": "취소",
    "무효": "무효확인",
    "무효확인": "무효",
    "의무확인": "의무확인소송",
    "부작위": "부작위위법확인",
    "작위": "작위위법확인",
    "대리인": "법정대리인",
    "법정대리인": "대리인",
    "처분": "행정처분",
    "행정처분": "처분",
    "심결": "심판결정",
    "항소": "항소심",
    "상고": "상고심",
    "재심": "재심사유",
    "확정": "확정판결",
    "가처분": "가처분신청",
    "집행정지": "집행정지신청",
    "이의신청": "이의",
    "심사청구": "심사",
    "소송": "행정소송",
    "행정소송": "소송",
    "사정판결": "사정",
    "참가": "보조참가",
    "보조참가": "참가",
}


def _sanitize_fts_query(query: str) -> str | None:
    terms = re.findall(r"[0-9A-Za-z가-힣]+", query)
    if not terms:
        return None
    expanded: list[str] = []
    for t in terms:
        if t in LEGAL_SYNONYMS:
            expanded.append(f'("{t}" OR "{LEGAL_SYNONYMS[t]}")')
        elif re.match(r"^[가-힣]+$", t) and len(t) >= 3:
            expanded.append(t + "*")
        else:
            expanded.append(t)
    return " ".join(expanded)


@lru_cache(maxsize=16)
def _cached_load_vector_artifacts(vector_dir: str) -> tuple[np.ndarray | None, tuple[dict, ...]]:
    """Memory-cached loader keyed by vector_dir string path."""
    vpath = Path(vector_dir)
    vectors_path = vpath / "vectors.npy"
    chunks_path = vpath / "chunks.jsonl"
    if not vectors_path.exists() or not chunks_path.exists():
        return None, ()
    vectors = np.load(vectors_path)
    if vectors.shape[0] == 0:
        return None, ()
    rows = tuple(
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return vectors, rows


def _load_vector_artifacts(vector_dir: Path) -> tuple[np.ndarray | None, list[dict]]:
    vectors, rows = _cached_load_vector_artifacts(str(vector_dir))
    if vectors is None:
        return None, []
    return vectors, list(rows)


@lru_cache(maxsize=16)
def _cached_faiss_index(vector_dir: str) -> "faiss.IndexFlatIP | None":
    """Build and cache a FAISS IndexFlatIP for the given vector directory."""
    if not _HAS_FAISS:
        return None
    vpath = Path(vector_dir)
    vectors_path = vpath / "vectors.npy"
    if not vectors_path.exists():
        return None
    vectors = np.load(vectors_path)
    if vectors.shape[0] == 0:
        return None
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # type: ignore[name-defined]
    index.add(vectors)
    return index


def _vector_search_core(
    query: str, vectors: np.ndarray, rows: list[dict], limit: int = 5
) -> list[tuple[float, dict]]:
    query_vector = _hash_vector(query, dim=VECTOR_DIM)
    scores = vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:limit]
    return [(float(scores[int(i)]), rows[int(i)]) for i in top_indices]


def _vector_search_faiss(
    query: str, vector_dir: Path, rows: list[dict], limit: int = 5
) -> list[tuple[float, dict]]:
    """FAISS-accelerated inner-product search."""
    index = _cached_faiss_index(str(vector_dir))
    if index is None:
        # Fallback to numpy brute-force
        vectors, _ = _cached_load_vector_artifacts(str(vector_dir))
        if vectors is None:
            return []
        return _vector_search_core(query, vectors, rows, limit)
    query_vector = _hash_vector(query, dim=VECTOR_DIM).reshape(1, -1)
    scores, indices = index.search(query_vector, limit)
    results: list[tuple[float, dict]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((float(score), rows[int(idx)]))
    return results


# ---------------------------------------------------------------------------
# BaseRetriever – shared logic for Prec / Decc / Detc / Expc
# ---------------------------------------------------------------------------


class BaseRetriever:
    """Base class for case-type retrievers with shared FTS, vector, and name search."""

    TABLE: str = ""
    FTS_TABLE: str = ""
    FTS_JOIN_COL: str = "doc_id"
    FTS_TABLE_ALIAS: str = "d"
    NAME_COL: str = "case_name_norm"
    ORDER_BY: str = "decision_date DESC, serial_no DESC"
    SOURCE_ID_COL: str = "serial_no"
    TITLE_COL: str = "case_name"
    CASE_NO_COL: str = "case_no"
    EXACT_COL: str = "case_no_norm"

    def __init__(self, db_path: Path, vector_dir: Path | None = None):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _result(self, row: sqlite3.Row, search_path: str, score: float) -> SearchResult:
        return SearchResult(
            search_path=search_path,
            doc_id=row["doc_id"],
            source_id=row[self.SOURCE_ID_COL],
            title=row[self.TITLE_COL],
            case_no=row[self.CASE_NO_COL],
            decision_date=row["decision_date"],
            score=score,
            payload=dict(row),
        )

    def _exact_case(self, case_no: str, limit: int = 5) -> list[SearchResult]:
        case_no_norm = normalize_lookup_text(case_no)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM {self.TABLE}
                WHERE {self.EXACT_COL} = ?
                  OR {self.EXACT_COL} LIKE ?
                  OR {self.EXACT_COL} LIKE ?
                ORDER BY {self.ORDER_BY}
                LIMIT ?
                """,
                (case_no_norm, f"%{case_no_norm}", f"{case_no_norm}%", limit),
            ).fetchall()
            return [self._result(row, "exact", 1.0) for row in rows]

    def fts_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    f"""
                    SELECT {self.FTS_TABLE_ALIAS}.*, bm25({self.FTS_TABLE}) AS score
                    FROM {self.FTS_TABLE}
                    JOIN {self.TABLE} {self.FTS_TABLE_ALIAS}
                      ON {self.FTS_TABLE_ALIAS}.{self.FTS_JOIN_COL} = {self.FTS_TABLE}.{self.FTS_JOIN_COL}
                    WHERE {self.FTS_TABLE} MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (safe_query, limit),
                ).fetchall()
                return [self._result(row, "fts", float(-row["score"])) for row in rows]
        except sqlite3.OperationalError:
            return []

    def name_search(self, normalized_query: str, limit: int = 5) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM {self.TABLE}
                WHERE {self.NAME_COL} LIKE ?
                ORDER BY {self.ORDER_BY}
                LIMIT ?
                """,
                (f"%{normalized_query}%", limit),
            ).fetchall()
            return [self._result(row, "name", 0.5) for row in rows]

    def vector_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if self.vector_dir is None:
            return []
        vectors, rows = _load_vector_artifacts(self.vector_dir)
        if vectors is None:
            return []
        top = _vector_search_faiss(query, self.vector_dir, rows, limit)
        results: list[SearchResult] = []
        seen: set[str] = set()
        for score, row in top:
            doc_id = row.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(
                SearchResult(
                    search_path="vector",
                    doc_id=doc_id,
                    source_id=row.get("source_id", ""),
                    title=row.get("title", ""),
                    case_no=row.get("case_no", ""),
                    decision_date=row.get("decision_date"),
                    score=score,
                    payload=row,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Concrete case retrievers
# ---------------------------------------------------------------------------


class PrecRetriever(BaseRetriever):
    TABLE = "precedents"
    FTS_TABLE = "precedents_fts"
    FTS_JOIN_COL = "doc_id"
    FTS_TABLE_ALIAS = "p"
    NAME_COL = "case_name_norm"
    ORDER_BY = "decision_date DESC, serial_no DESC"
    SOURCE_ID_COL = "serial_no"
    TITLE_COL = "case_name"
    CASE_NO_COL = "case_no"

    def exact_case(self, case_no: str, limit: int = 5) -> list[SearchResult]:
        return self._exact_case(case_no, limit)


class DeccRetriever(BaseRetriever):
    TABLE = "decc_cases"
    FTS_TABLE = "decc_cases_fts"
    FTS_JOIN_COL = "doc_id"
    FTS_TABLE_ALIAS = "d"
    NAME_COL = "case_name_norm"
    ORDER_BY = "decision_date DESC, serial_no DESC"
    SOURCE_ID_COL = "serial_no"
    TITLE_COL = "case_name"
    CASE_NO_COL = "case_no"

    def exact_case(self, case_no: str, limit: int = 5) -> list[SearchResult]:
        return self._exact_case(case_no, limit)


class DetcRetriever(BaseRetriever):
    TABLE = "detc_cases"
    FTS_TABLE = "detc_cases_fts"
    FTS_JOIN_COL = "doc_id"
    FTS_TABLE_ALIAS = "d"
    NAME_COL = "case_name_norm"
    ORDER_BY = "decision_date DESC, serial_no DESC"
    SOURCE_ID_COL = "serial_no"
    TITLE_COL = "case_name"
    CASE_NO_COL = "case_no"

    def exact_case(self, case_no: str, limit: int = 5) -> list[SearchResult]:
        return self._exact_case(case_no, limit)


class ExpcRetriever(BaseRetriever):
    TABLE = "expc_cases"
    FTS_TABLE = "expc_cases_fts"
    FTS_JOIN_COL = "doc_id"
    FTS_TABLE_ALIAS = "e"
    NAME_COL = "title_norm"
    ORDER_BY = "decision_date DESC, serial_no DESC"
    SOURCE_ID_COL = "serial_no"
    TITLE_COL = "title"
    CASE_NO_COL = "issue_no"
    EXACT_COL = "issue_no_norm"

    def exact_issue(self, issue_no: str, limit: int = 5) -> list[SearchResult]:
        return self._exact_case(issue_no, limit)

    def title_search(self, normalized_query: str, limit: int = 5) -> list[SearchResult]:
        return self.name_search(normalized_query, limit)

    def name_search(self, normalized_query: str, limit: int = 5) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM {self.TABLE}
                WHERE {self.NAME_COL} LIKE ?
                ORDER BY {self.ORDER_BY}
                LIMIT ?
                """,
                (f"%{normalized_query}%", limit),
            ).fetchall()
            return [self._result(row, "name", 0.5) for row in rows]


# ---------------------------------------------------------------------------
# Unique-pattern retrievers (kept as-is)
# ---------------------------------------------------------------------------


class AcrRetriever:
    def __init__(self, db_path: Path, vector_dir: Path):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _result(self, row: sqlite3.Row, search_path: str, score: float) -> SearchResult:
        return SearchResult(
            search_path=search_path,
            doc_id=row["doc_id"],
            source_id=row["source_id"],
            title=row["title"],
            case_no=row["case_no"],
            decision_date=row["decision_date"],
            score=score,
            payload=dict(row),
        )

    def exact_by_source_id(self, source_id: str) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM docs WHERE source_id = ?", (source_id,)).fetchall()
            return [self._result(row, "exact", 1.0) for row in rows]

    def exact_by_case_no(self, case_no: str) -> list[SearchResult]:
        case_no_norm = normalize_lookup_text(case_no)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM docs WHERE case_no_norm = ?", (case_no_norm,)
            ).fetchall()
            return [self._result(row, "exact", 1.0) for row in rows]

    def fts_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT d.*, bm25(docs_fts) AS score
                    FROM docs_fts
                    JOIN docs d ON d.doc_id = docs_fts.doc_id
                    WHERE docs_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (safe_query, limit),
                ).fetchall()
                return [self._result(row, "fts", float(-row["score"])) for row in rows]
        except sqlite3.OperationalError:
            return []

    def title_search(self, normalized_query: str, limit: int = 5) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM docs
                WHERE title_norm LIKE ? OR complaint_title_norm LIKE ?
                """,
                (f"%{normalized_query}%", f"%{normalized_query}%"),
            ).fetchall()
            ranked: list[tuple[tuple[int, int, int], sqlite3.Row]] = []
            for row in rows:
                exact_title = 1 if row["title_norm"] == normalized_query else 0
                exact_complaint = 1 if row["complaint_title_norm"] == normalized_query else 0
                contains_title = 1 if normalized_query in row["title_norm"] else 0
                try:
                    source_rank = int(row["source_id"])
                except ValueError:
                    source_rank = 10**9
                ranked.append(
                    (
                        (
                            -(exact_title + exact_complaint),
                            -contains_title,
                            source_rank,
                        ),
                        row,
                    )
                )
            ranked.sort(key=lambda item: item[0])
            return [self._result(row, "title", 0.5) for _, row in ranked[:limit]]

    def vector_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        vectors, rows = _load_vector_artifacts(self.vector_dir)
        if vectors is None:
            return []
        top = _vector_search_faiss(query, self.vector_dir, rows, limit)

        doc_best: dict[str, tuple[float, dict]] = {}
        for score, row in top:
            current = doc_best.get(row["doc_id"])
            if current is None or score > current[0]:
                doc_best[row["doc_id"]] = (score, row)

        with self._connect() as conn:
            results: list[SearchResult] = []
            for doc_id, (score, _) in sorted(
                doc_best.items(), key=lambda item: item[1][0], reverse=True
            ):
                row = conn.execute("SELECT * FROM docs WHERE doc_id = ?", (doc_id,)).fetchone()
                if row is None:
                    continue
                results.append(self._result(row, "vector", score))
            return results

    def search(self, decision: RouteDecision, limit: int = 5) -> list[SearchResult]:
        if decision.strategy == "exact_source_id" and decision.serial_no:
            exact = self.exact_by_source_id(decision.serial_no)
            if exact:
                return exact

        if decision.strategy == "exact_case_no" and decision.case_no:
            exact = self.exact_by_case_no(decision.case_no)
            if exact:
                return exact

        normalized_query = decision.normalized_query
        if decision.strategy == "fts":
            exactish = self.title_search(normalized_query, limit=limit)
            if exactish:
                return exactish
            terms = " ".join(re.findall(r"[0-9A-Za-z가-힣]+", decision.raw_query))
            if terms:
                fts = self.fts_search(terms, limit=limit)
                if fts:
                    return fts
            return self.vector_search(decision.raw_query, limit=limit)

        if decision.strategy == "vector":
            results = self.vector_search(decision.raw_query, limit=limit)
            if results:
                return results
            exactish = self.title_search(normalized_query, limit=limit)
            if exactish:
                return exactish
            return []

        alternate = self.title_search(normalized_query, limit=limit)
        if alternate:
            return alternate
        return self.vector_search(decision.raw_query, limit=limit)


class LawRetriever:
    def __init__(self, db_path: Path, vector_dir: Path | None = None):
        self.db_path = db_path
        self.vector_dir = vector_dir
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _result(self, row: sqlite3.Row, search_path: str, score: float) -> SearchResult:
        return SearchResult(
            search_path=search_path,
            doc_id=row["doc_id"],
            source_id=row["law_id"],
            title=row["law_name"],
            case_no=row["article_no"],
            decision_date=None,
            score=score,
            payload=dict(row),
        )

    def exact_article(self, law_name: str, article_no: str, limit: int = 5) -> list[SearchResult]:
        law_name_norm = normalize_lookup_text(law_name)
        article_no_norm = normalize_lookup_text(article_no)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM law_articles
                WHERE article_no_norm = ?
                  AND (
                    law_name_norm = ?
                    OR short_name_norm = ?
                    OR law_name_norm LIKE ?
                    OR short_name_norm LIKE ?
                  )
                ORDER BY LENGTH(law_name_norm), law_name
                LIMIT ?
                """,
                (
                    article_no_norm,
                    law_name_norm,
                    law_name_norm,
                    f"%{law_name_norm}%",
                    f"%{law_name_norm}%",
                    limit,
                ),
            ).fetchall()
            return [self._result(row, "exact_law", 1.0) for row in rows]

    def fts_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        safe_query = _sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT la.*, bm25(law_articles_fts) AS score
                    FROM law_articles_fts
                    JOIN law_articles la ON la.article_id = law_articles_fts.article_id
                    WHERE law_articles_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                    """,
                    (safe_query, limit),
                ).fetchall()
                return [self._result(row, "fts", float(-row["score"])) for row in rows]
        except sqlite3.OperationalError:
            return []

    def name_search(self, normalized_query: str, limit: int = 5) -> list[SearchResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM law_articles
                WHERE law_name_norm LIKE ? OR short_name_norm LIKE ?
                ORDER BY LENGTH(law_name_norm), law_name, article_no_norm
                LIMIT ?
                """,
                (f"%{normalized_query}%", f"%{normalized_query}%", limit),
            ).fetchall()
            return [self._result(row, "name", 0.5) for row in rows]

    def vector_search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if self.vector_dir is None:
            return []
        vectors, rows = _load_vector_artifacts(self.vector_dir)
        if vectors is None:
            return []
        top = _vector_search_faiss(query, self.vector_dir, rows, limit)
        results: list[SearchResult] = []
        seen: set[str] = set()
        for score, row in top:
            doc_id = row.get("doc_id", "")
            if doc_id in seen:
                continue
            seen.add(doc_id)
            results.append(
                SearchResult(
                    search_path="vector",
                    doc_id=doc_id,
                    source_id=row.get("source_id", ""),
                    title=row.get("title", ""),
                    case_no=row.get("case_no", ""),
                    decision_date=row.get("decision_date"),
                    score=score,
                    payload=row,
                )
            )
        return results
