from __future__ import annotations

import argparse
import json
from pathlib import Path

from .retriever import AcrHybridRetriever


def evaluate(query_file: Path, base_dir: Path, index_dir: Path) -> dict:
    retriever = AcrHybridRetriever(base_dir, index_dir)
    queries = json.loads(query_file.read_text(encoding="utf-8"))
    rows = []
    correct = 0
    for entry in queries:
        result = retriever.search(entry["query"], limit=3)
        top = result["results"][0] if result["results"] else None
        is_correct = bool(top and top["serial_number"] == entry["expected_serial_number"])
        correct += 1 if is_correct else 0
        rows.append(
            {
                "query": entry["query"],
                "expected_serial_number": entry["expected_serial_number"],
                "predicted_serial_number": top["serial_number"] if top else None,
                "predicted_title": top["title"] if top else None,
                "is_correct": is_correct,
                "status": result["status"],
            }
        )
    precision = correct / len(rows) if rows else 0.0
    return {
        "query_count": len(rows),
        "correct_top1": correct,
        "top1_precision": precision,
        "passed": precision >= 0.95,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate acr retrieval precision")
    parser.add_argument("--query-file", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--index-dir", default="artifacts/legal_rag")
    args = parser.parse_args()

    report = evaluate(Path(args.query_file), Path(args.base_dir), Path(args.index_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
