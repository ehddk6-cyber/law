from __future__ import annotations

import argparse
import json
from pathlib import Path

from .indexer import DEFAULT_INDEX_DIR, build_acr_indexes
from .retriever import AcrHybridRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="law_open_data hybrid legal RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser_cmd = subparsers.add_parser("build-index", help="Build acr indexes")
    build_parser_cmd.add_argument("--base-dir", default=".", help="Repository root")
    build_parser_cmd.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Index output directory")

    search_parser = subparsers.add_parser("search", help="Run hybrid retrieval")
    search_parser.add_argument("query", help="User query")
    search_parser.add_argument("--base-dir", default=".", help="Repository root")
    search_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Index directory")
    search_parser.add_argument("--limit", type=int, default=3, help="Maximum results")

    verify_parser = subparsers.add_parser("verify", help="Verify retrieval over a query file")
    verify_parser.add_argument("--base-dir", default=".", help="Repository root")
    verify_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Index directory")
    verify_parser.add_argument("--query-file", required=True, help="JSON file containing queries")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build-index":
        manifest = build_acr_indexes(Path(args.base_dir), Path(args.index_dir))
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    retriever = AcrHybridRetriever(Path(args.base_dir), Path(args.index_dir))

    if args.command == "search":
        result = retriever.search(args.query, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "verify":
        query_path = Path(args.query_file)
        queries = json.loads(query_path.read_text(encoding="utf-8"))
        outputs = []
        for entry in queries:
            outputs.append(
                {
                    "query": entry["query"],
                    "expected_serial_number": entry.get("expected_serial_number"),
                    "result": retriever.search(entry["query"], limit=entry.get("limit", 3)),
                }
            )
        print(json.dumps(outputs, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

