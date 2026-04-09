from __future__ import annotations

import argparse
import json
from pathlib import Path

from parsers.acr import iter_acr_records
from parsers.decc import iter_decc_records
from parsers.detc import iter_detc_records
from parsers.expc import iter_expc_records
from parsers.law import iter_law_records
from parsers.prec import iter_prec_records


def build_acr_corpus(data_root: Path, output_path: Path) -> int:
    records = iter_acr_records(data_root / "acr")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def build_law_corpus(data_root: Path, output_path: Path) -> int:
    records = iter_law_records(data_root / "law")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def build_prec_corpus(data_root: Path, output_path: Path) -> int:
    records = iter_prec_records(data_root / "prec")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def build_decc_corpus(data_root: Path, output_path: Path) -> int:
    records = iter_decc_records(data_root / "decc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def build_expc_corpus(data_root: Path, output_path: Path) -> int:
    records = iter_expc_records(data_root / "expc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def build_detc_corpus(data_root: Path, output_path: Path) -> int:
    records = iter_detc_records(data_root / "detc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized corpus files from raw law_open_data.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / ".artifacts" / "acr" / "documents.jsonl")
    parser.add_argument("--law-output", type=Path, default=Path(__file__).resolve().parent / ".artifacts" / "law" / "documents.jsonl")
    parser.add_argument("--prec-output", type=Path, default=Path(__file__).resolve().parent / ".artifacts" / "prec" / "documents.jsonl")
    parser.add_argument("--decc-output", type=Path, default=Path(__file__).resolve().parent / ".artifacts" / "decc" / "documents.jsonl")
    parser.add_argument("--expc-output", type=Path, default=Path(__file__).resolve().parent / ".artifacts" / "expc" / "documents.jsonl")
    parser.add_argument("--detc-output", type=Path, default=Path(__file__).resolve().parent / ".artifacts" / "detc" / "documents.jsonl")
    args = parser.parse_args()

    count = build_acr_corpus(args.data_root, args.output)
    print(f"normalized_docs={count}")
    print(f"output={args.output}")
    law_count = build_law_corpus(args.data_root, args.law_output)
    print(f"law_normalized_docs={law_count}")
    print(f"law_output={args.law_output}")
    prec_count = build_prec_corpus(args.data_root, args.prec_output)
    print(f"prec_normalized_docs={prec_count}")
    print(f"prec_output={args.prec_output}")
    decc_count = build_decc_corpus(args.data_root, args.decc_output)
    print(f"decc_normalized_docs={decc_count}")
    print(f"decc_output={args.decc_output}")
    expc_count = build_expc_corpus(args.data_root, args.expc_output)
    print(f"expc_normalized_docs={expc_count}")
    print(f"expc_output={args.expc_output}")
    detc_count = build_detc_corpus(args.data_root, args.detc_output)
    print(f"detc_normalized_docs={detc_count}")
    print(f"detc_output={args.detc_output}")


if __name__ == "__main__":
    main()
