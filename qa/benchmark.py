from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.unified import UnifiedSearcher


BENCHMARK_QUERIES = [
    {"query": "행정심판법 제18조", "category": "law", "type": "exact"},
    {"query": "84누180", "category": "prec", "type": "exact"},
    {"query": "2000-04033", "category": "decc", "type": "exact"},
    {"query": "2004헌마275", "category": "detc", "type": "exact"},
    {"query": "05-0096", "category": "expc", "type": "exact"},
    {"query": "결정문일련번호 23", "category": "acr", "type": "exact"},
    {"query": "행정심판 대리인", "category": "acr", "type": "search"},
    {"query": "양도소득세부과처분취소", "category": "prec", "type": "search"},
    {"query": "산업재해보상보험", "category": "decc", "type": "search"},
    {"query": "헌법소원 심판", "category": "detc", "type": "search"},
    {"query": "법령해석 회신", "category": "expc", "type": "search"},
    {"query": "법인세 부과", "category": "mixed", "type": "search"},
]


@dataclass
class BenchmarkResult:
    query: str
    expected_category: str
    type: str
    strategy: str
    result_count: int
    top_source_type: str | None
    top_score: float
    latency_ms: float
    found_expected: bool


def run_benchmark(root: Path, iterations: int = 1) -> dict:
    searcher = UnifiedSearcher(root)
    results: list[BenchmarkResult] = []

    for entry in BENCHMARK_QUERIES:
        query = entry["query"]
        latencies: list[float] = []
        last_result = None

        for _ in range(iterations):
            t0 = time.perf_counter()
            decision, unified = searcher.search(query, limit=3)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed)
            last_result = (decision, unified)

        decision, unified = last_result
        top = unified[0] if unified else None
        found_expected = False
        if entry["type"] == "exact" and top:
            found_expected = top.source_type == entry["category"]
        elif entry["category"] != "mixed" and top:
            found_expected = top.source_type == entry["category"]

        results.append(
            BenchmarkResult(
                query=query,
                expected_category=entry["category"],
                type=entry["type"],
                strategy=decision.strategy,
                result_count=len(unified),
                top_source_type=top.source_type if top else None,
                top_score=top.score if top else 0.0,
                latency_ms=statistics.mean(latencies),
                found_expected=found_expected,
            )
        )

    exact_results = [r for r in results if r.type == "exact"]
    search_results = [r for r in results if r.type == "search"]
    all_latencies = [r.latency_ms for r in results]

    summary = {
        "iterations": iterations,
        "total_queries": len(results),
        "exact_accuracy": (
            round(
                sum(1 for r in exact_results if r.found_expected)
                / len(exact_results)
                * 100,
                1,
            )
            if exact_results
            else 0
        ),
        "search_relevance": (
            round(
                sum(1 for r in search_results if r.found_expected)
                / len(search_results)
                * 100,
                1,
            )
            if search_results
            else 0
        ),
        "latency_ms": {
            "mean": round(statistics.mean(all_latencies), 1),
            "median": round(statistics.median(all_latencies), 1),
            "p95": round(sorted(all_latencies)[int(len(all_latencies) * 0.95)], 1)
            if len(all_latencies) >= 2
            else round(max(all_latencies), 1),
            "min": round(min(all_latencies), 1),
            "max": round(max(all_latencies), 1),
        },
        "results": [asdict(r) for r in results],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark retrieval latency and accuracy."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = run_benchmark(args.root, iterations=args.iterations)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"iterations={summary['iterations']}")
    print(f"total_queries={summary['total_queries']}")
    print(f"exact_accuracy={summary['exact_accuracy']}%")
    print(f"search_relevance={summary['search_relevance']}%")
    print(f"latency_mean={summary['latency_ms']['mean']}ms")
    print(f"latency_median={summary['latency_ms']['median']}ms")
    print(f"latency_p95={summary['latency_ms']['p95']}ms")
    for r in summary["results"]:
        status = "OK" if r["found_expected"] else "MISS"
        print(
            f"  {status} [{r['type']}] {r['query'][:30]:30s} "
            f"strategy={r['strategy']:20s} top={r['top_source_type'] or 'none':6s} "
            f"score={r['top_score']:.4f} latency={r['latency_ms']:.1f}ms"
        )


if __name__ == "__main__":
    main()
