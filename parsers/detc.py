from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from normalizers.text import clean_html_text, normalize_lookup_text
from parsers.base import xml_text as _text


def parse_detc_xml(xml_path: Path) -> dict | None:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    serial_no = _text(root, "헌재결정례일련번호")
    case_no = _text(root, "사건번호")
    case_name = _text(root, "사건명")
    if not serial_no or not case_no or not case_name:
        return None

    return {
        "doc_id": f"detc:{serial_no}",
        "serial_no": serial_no,
        "case_no": case_no,
        "case_no_norm": normalize_lookup_text(case_no),
        "case_name": case_name,
        "case_name_norm": normalize_lookup_text(case_name),
        "decision_date": _text(root, "종국일자"),
        "case_type": _text(root, "사건종류명"),
        "decision_summary": clean_html_text(_text(root, "결정요지")),
        "issue": clean_html_text(_text(root, "판시사항")),
        "content": clean_html_text(_text(root, "전문")),
        "cited_laws": clean_html_text(_text(root, "참조조문")),
        "cited_cases": clean_html_text(_text(root, "참조판례")),
        "target_provisions": clean_html_text(_text(root, "심판대상조문")),
        "source_path": str(xml_path),
    }


def iter_detc_records(detc_root: Path) -> list[dict]:
    records: list[dict] = []
    for xml_path in sorted((detc_root / "body" / "xml").glob("ID_*.xml")):
        if xml_path.name.endswith(":Zone.Identifier"):
            continue
        record = parse_detc_xml(xml_path)
        if record is not None:
            records.append(record)
    return records
