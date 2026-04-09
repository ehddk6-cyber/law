from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from normalizers.text import clean_html_text, normalize_lookup_text
from parsers.base import xml_text as _text


def parse_decc_xml(xml_path: Path) -> dict | None:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    serial_no = _text(root, "행정심판례일련번호")
    case_no = _text(root, "사건번호")
    case_name = _text(root, "사건명")
    if not serial_no or not case_no or not case_name:
        return None

    return {
        "doc_id": f"decc:{serial_no}",
        "serial_no": serial_no,
        "case_no": case_no,
        "case_no_norm": normalize_lookup_text(case_no),
        "case_name": case_name,
        "case_name_norm": normalize_lookup_text(case_name),
        "decision_date": _text(root, "의결일자"),
        "agency": _text(root, "재결청"),
        "disposition_agency": _text(root, "처분청"),
        "decision_type_name": _text(root, "재결례유형명"),
        "order_text": clean_html_text(_text(root, "주문")),
        "claim_text": clean_html_text(_text(root, "청구취지")),
        "reason_text": clean_html_text(_text(root, "이유")),
        "source_path": str(xml_path),
    }


def iter_decc_records(decc_root: Path) -> list[dict]:
    records: list[dict] = []
    for xml_path in sorted((decc_root / "body" / "xml").glob("ID_*.xml")):
        if xml_path.name.endswith(":Zone.Identifier"):
            continue
        record = parse_decc_xml(xml_path)
        if record is not None:
            records.append(record)
    return records
