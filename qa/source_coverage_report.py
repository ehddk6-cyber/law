from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


ERROR_MARKERS = {
    "prec": "일치하는 판례가 없습니다",
    "decc": "일치하는 행정심판례가 없습니다",
}


@dataclass
class CoverageRow:
    category: str
    saved_items: int | None
    reported_total: int | None
    body_xml_files: int
    normalized_docs: int
    error_bodies: int
    parse_gap: int


def _read_summary(summary_path: Path) -> dict:
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _count_error_bodies(xml_root: Path, marker: str | None) -> int:
    if marker is None or not xml_root.exists():
        return 0

    count = 0
    for xml_path in xml_root.glob("ID_*.xml"):
        with xml_path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read(256)
        if marker in text:
            count += 1
    return count


def build_row(root: Path, category: str) -> CoverageRow:
    category_root = root / category
    summary = _read_summary(category_root / "summary.json")
    xml_root = category_root / "body" / "xml"
    artifacts_jsonl = root / ".artifacts" / category / "documents.jsonl"

    body_xml_files = len(list(xml_root.glob("ID_*.xml"))) if xml_root.exists() else 0
    normalized_docs = _count_jsonl_rows(artifacts_jsonl)
    error_bodies = _count_error_bodies(xml_root, ERROR_MARKERS.get(category))
    parse_gap = body_xml_files - normalized_docs - error_bodies

    return CoverageRow(
        category=category,
        saved_items=summary.get("saved_items"),
        reported_total=summary.get("reported_total"),
        body_xml_files=body_xml_files,
        normalized_docs=normalized_docs,
        error_bodies=error_bodies,
        parse_gap=parse_gap,
    )


def format_row(row: CoverageRow) -> str:
    return (
        f"{row.category:>5} | "
        f"saved={row.saved_items!s:>7} | "
        f"reported={row.reported_total!s:>7} | "
        f"xml={row.body_xml_files:>7} | "
        f"normalized={row.normalized_docs:>7} | "
        f"error_bodies={row.error_bodies:>7} | "
        f"parse_gap={row.parse_gap:>7}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report raw-to-normalized coverage for selected legal sources.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root that contains category folders and .artifacts/",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["prec", "decc", "detc", "expc"],
        help="Categories to inspect.",
    )
    args = parser.parse_args()

    for category in args.categories:
        row = build_row(args.root, category)
        print(format_row(row))


if __name__ == "__main__":
    main()
