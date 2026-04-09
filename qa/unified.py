from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from qa.retrievers import (
    AcrRetriever,
    DeccRetriever,
    DetcRetriever,
    ExpcRetriever,
    LawRetriever,
    PrecRetriever,
    SearchResult,
)
from qa.router import RouteDecision, route_query

SOURCE_WEIGHTS: dict[str, float] = {
    "law": 1.0,
    "prec": 0.95,
    "decc": 0.8,
    "detc": 0.8,
    "acr": 0.8,
    "expc": 0.8,
}

EXACT_STRATEGIES = {
    "exact_source_id",
    "exact_case_no",
    "exact_law_article",
    "exact_precedent_case",
    "exact_decc_case",
    "exact_detc_case",
    "exact_expc_issue",
}


@dataclass
class UnifiedResult:
    source_type: str
    search_path: str
    score: float
    payload: dict


def _normalize_score(score: float, search_path: str) -> float:
    if search_path.startswith("exact"):
        return 1.0
    if search_path == "fts":
        # BM25 negated: positive, higher=better. Map to 0~1.
        if score <= 0:
            return 0.0
        return min(score / 10.0, 0.99)
    if search_path == "vector":
        # Cosine similarity: 0~1 already.
        return max(0.0, min(score, 1.0))
    if search_path in ("name", "title"):
        return 0.5
    return min(max(score, 0.0), 1.0)


def _to_unified(source_type: str, result: SearchResult) -> UnifiedResult:
    normalized = _normalize_score(result.score, result.search_path)
    weight = SOURCE_WEIGHTS.get(source_type, 0.8)
    return UnifiedResult(
        source_type=source_type,
        search_path=result.search_path,
        score=normalized * weight,
        payload=result.payload,
    )


def _deduplicate(results: list[UnifiedResult]) -> list[UnifiedResult]:
    """Remove cross-source duplicates by doc_id, keeping the highest-scored entry."""
    seen: dict[str, UnifiedResult] = {}
    for r in results:
        doc_id = r.payload.get("doc_id", "")
        if not doc_id:
            # Keep results without doc_id (shouldn't happen, but be safe)
            key = f"__nodocid__{id(r)}"
        else:
            key = doc_id
        if key not in seen or r.score > seen[key].score:
            seen[key] = r
    return list(seen.values())


class UnifiedSearcher:
    def __init__(self, root: Path):
        artifacts = root / ".artifacts"
        self.acr = AcrRetriever(artifacts / "acr" / "acr.sqlite3", artifacts / "acr" / "vector")
        self.law = LawRetriever(artifacts / "law" / "law.sqlite3", artifacts / "law" / "vector")
        self.prec = PrecRetriever(
            artifacts / "prec" / "prec.sqlite3", artifacts / "prec" / "vector"
        )
        self.decc = DeccRetriever(
            artifacts / "decc" / "decc.sqlite3", artifacts / "decc" / "vector"
        )
        self.detc = DetcRetriever(
            artifacts / "detc" / "detc.sqlite3", artifacts / "detc" / "vector"
        )
        self.expc = ExpcRetriever(
            artifacts / "expc" / "expc.sqlite3", artifacts / "expc" / "vector"
        )

    def search(self, query: str, limit: int = 5) -> tuple[RouteDecision, list[UnifiedResult]]:
        decision = route_query(query)

        if decision.strategy == "exact_source_id" and decision.serial_no:
            results = self.acr.exact_by_source_id(decision.serial_no)
            if results:
                return decision, [_to_unified("acr", r) for r in results]
            return decision, []

        if decision.strategy == "exact_case_no" and decision.case_no:
            results = self.acr.exact_by_case_no(decision.case_no)
            if results:
                return decision, [_to_unified("acr", r) for r in results]
            return decision, []

        if decision.strategy == "exact_law_article" and decision.law_name and decision.article_no:
            results = self.law.exact_article(decision.law_name, decision.article_no, limit=limit)
            if results:
                return decision, [_to_unified("law", r) for r in results]
            return decision, []

        if decision.strategy == "exact_precedent_case" and decision.precedent_case_no:
            results = self.prec.exact_case(decision.precedent_case_no, limit=limit)
            if results:
                return decision, [_to_unified("prec", r) for r in results]
            return decision, []

        if decision.strategy == "exact_decc_case" and decision.decc_case_no:
            results = self.decc.exact_case(decision.decc_case_no, limit=limit)
            if results:
                return decision, [_to_unified("decc", r) for r in results]
            return decision, []

        if decision.strategy == "exact_detc_case" and decision.detc_case_no:
            results = self.detc.exact_case(decision.detc_case_no, limit=limit)
            if results:
                return decision, [_to_unified("detc", r) for r in results]
            return decision, []

        if decision.strategy == "exact_expc_issue" and decision.expc_issue_no:
            results = self.expc.exact_issue(decision.expc_issue_no, limit=limit)
            if results:
                return decision, [_to_unified("expc", r) for r in results]
            return decision, []

        if decision.strategy == "law_name_search" and decision.law_name:
            results = self.law.name_search(decision.normalized_query, limit=limit)
            if results:
                return decision, [_to_unified("law", r) for r in results]
            return self._fallback_search(decision, limit)

        return self._do_multi_search(decision, limit)

    def _fallback_search(
        self, decision: RouteDecision, limit: int
    ) -> tuple[RouteDecision, list[UnifiedResult]]:
        if not decision.fallback_strategies:
            return decision, []
        for strategy in decision.fallback_strategies:
            fallback = RouteDecision(
                strategy=strategy,
                raw_query=decision.raw_query,
                normalized_query=decision.normalized_query,
                fallback_strategies=None,
            )
            _, results = self._do_multi_search(fallback, limit)
            if results:
                return decision, results
        return decision, []

    def _do_multi_search(
        self, decision: RouteDecision, limit: int
    ) -> tuple[RouteDecision, list[UnifiedResult]]:
        # Build (source_type, callable) pairs for parallel execution
        tasks: list[tuple[str, callable]] = []  # type: ignore[type-arg]
        tasks.append(("acr", lambda: self.acr.search(decision, limit=limit)))

        if decision.strategy == "fts":
            q = decision.raw_query
            tasks.extend(
                [
                    ("prec", lambda: self.prec.fts_search(q, limit=limit)),
                    ("decc", lambda: self.decc.fts_search(q, limit=limit)),
                    ("detc", lambda: self.detc.fts_search(q, limit=limit)),
                    ("expc", lambda: self.expc.fts_search(q, limit=limit)),
                    ("law", lambda: self.law.fts_search(q, limit=limit)),
                ]
            )
        elif decision.strategy == "vector":
            q = decision.raw_query
            tasks.extend(
                [
                    ("prec", lambda: self.prec.vector_search(q, limit=limit)),
                    ("decc", lambda: self.decc.vector_search(q, limit=limit)),
                    ("detc", lambda: self.detc.vector_search(q, limit=limit)),
                    ("expc", lambda: self.expc.vector_search(q, limit=limit)),
                    ("law", lambda: self.law.vector_search(q, limit=limit)),
                ]
            )
        else:
            nq = decision.normalized_query
            tasks.extend(
                [
                    ("prec", lambda: self.prec.name_search(nq, limit=limit)),
                    ("decc", lambda: self.decc.name_search(nq, limit=limit)),
                    ("detc", lambda: self.detc.name_search(nq, limit=limit)),
                    ("expc", lambda: self.expc.title_search(nq, limit=limit)),
                    ("law", lambda: self.law.name_search(nq, limit=limit)),
                ]
            )

        # Execute all retrievers in parallel
        all_results: list[UnifiedResult] = []
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            future_map = {executor.submit(fn): source_type for source_type, fn in tasks}
            for future in as_completed(future_map):
                source_type = future_map[future]
                try:
                    raw_results: list[SearchResult] = future.result()
                    all_results.extend(_to_unified(source_type, r) for r in raw_results)
                except Exception:
                    # Silently skip failed retrievers
                    continue

        # Deduplicate across sources, then sort and truncate
        all_results = _deduplicate(all_results)
        all_results.sort(key=lambda item: item.score, reverse=True)
        return decision, all_results[:limit]
