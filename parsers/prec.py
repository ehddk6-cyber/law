from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from normalizers.text import clean_html_text, normalize_lookup_text
from parsers.base import xml_text as _text


def parse_prec_xml(xml_path: Path) -> dict | None:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    serial_no = _text(root, "판례정보일련번호")
    case_no = _text(root, "사건번호")
    case_name = _text(root, "사건명")
    if not serial_no or not case_no or not case_name:
        return None

    holding = clean_html_text(_text(root, "판결요지"))
    issue = clean_html_text(_text(root, "판시사항"))
    content = clean_html_text(_text(root, "판례내용"))
    cited_laws = clean_html_text(_text(root, "참조조문"))
    cited_cases = clean_html_text(_text(root, "참조판례"))

    return {
        "doc_id": f"prec:{serial_no}",
        "serial_no": serial_no,
        "case_no": case_no,
        "case_no_norm": normalize_lookup_text(case_no),
        "case_name": case_name,
        "case_name_norm": normalize_lookup_text(case_name),
        "court_name": _text(root, "법원명"),
        "case_type": _text(root, "사건종류명"),
        "decision_type": _text(root, "판결유형"),
        "decision_date": _text(root, "선고일자"),
        "issue": issue,
        "holding": holding,
        "cited_laws": cited_laws,
        "cited_cases": cited_cases,
        "content": content,
        "source_path": str(xml_path),
    }


def iter_prec_records(prec_root: Path) -> list[dict]:
    records: list[dict] = []
    for xml_path in sorted((prec_root / "body" / "xml").glob("ID_*.xml")):
        if xml_path.name.endswith(":Zone.Identifier"):
            continue
        record = parse_prec_xml(xml_path)
        if record is not None:
            records.append(record)
    return records
