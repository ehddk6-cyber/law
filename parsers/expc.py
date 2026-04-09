from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from normalizers.text import clean_html_text, normalize_lookup_text
from parsers.base import xml_text as _text


def parse_expc_xml(xml_path: Path) -> dict | None:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    serial_no = _text(root, "법령해석례일련번호")
    issue_no = _text(root, "안건번호")
    title = _text(root, "안건명")
    if not serial_no or not issue_no or not title:
        return None

    return {
        "doc_id": f"expc:{serial_no}",
        "serial_no": serial_no,
        "issue_no": issue_no,
        "issue_no_norm": normalize_lookup_text(issue_no),
        "title": title,
        "title_norm": normalize_lookup_text(title),
        "decision_date": _text(root, "해석일자"),
        "agency": _text(root, "해석기관명"),
        "query_agency": _text(root, "질의기관명"),
        "query_summary": clean_html_text(_text(root, "질의요지")),
        "answer_text": clean_html_text(_text(root, "회답")),
        "reason_text": clean_html_text(_text(root, "이유")),
        "source_path": str(xml_path),
    }


def iter_expc_records(expc_root: Path) -> list[dict]:
    records: list[dict] = []
    for xml_path in sorted((expc_root / "body" / "xml").glob("ID_*.xml")):
        if xml_path.name.endswith(":Zone.Identifier"):
            continue
        record = parse_expc_xml(xml_path)
        if record is not None:
            records.append(record)
    return records
