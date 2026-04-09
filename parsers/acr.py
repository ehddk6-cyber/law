from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from normalizers.date import parse_korean_date
from normalizers.text import clean_html_text, compact_spaces, normalize_display_text, normalize_lookup_text
from parsers.base import ChunkRecord, DocumentRecord


SECTION_PATTERNS = [
    ("reason_1", "신청취지", re.compile(r"1\.\s*신청취지(.*?)(?=2\.\s*피신청인의 주장|2\.\s*피신청인 등의 주장|3\.\s*사실관계|4\.\s*판단|5\.\s*결론|$)", re.S)),
    ("reason_1", "피신청인의 주장", re.compile(r"2\.\s*피신청인(?: 등의)? 주장(.*?)(?=3\.\s*사실관계|4\.\s*판단|5\.\s*결론|$)", re.S)),
    ("reason_2", "사실관계", re.compile(r"3\.\s*사실관계(.*?)(?=4\.\s*판단|5\.\s*결론|$)", re.S)),
    ("reason_3", "판단", re.compile(r"4\.\s*판단(.*?)(?=5\.\s*결론|$)", re.S)),
    ("reason_3", "결론", re.compile(r"5\.\s*결론(.*)$", re.S)),
]

CITATION_RE = re.compile(r"「([^」]+)」\s*제(\d+)조(?:\s*제(\d+)항)?(?:\s*제(\d+)호)?")


def _first_text(root: ET.Element, tag: str) -> str:
    node = root.find(f".//{tag}")
    if node is None:
        return ""
    return normalize_display_text("".join(node.itertext()))


def _split_reason_sections(reason_text: str) -> list[tuple[str, str, str]]:
    sections: list[tuple[str, str, str]] = []
    for chunk_type, heading, pattern in SECTION_PATTERNS:
        match = pattern.search(reason_text)
        if not match:
            continue
        body = clean_html_text(match.group(1))
        if body:
            sections.append((chunk_type, heading, body))
    if sections:
        return sections
    fallback = clean_html_text(reason_text)
    return [("reason_full", "이유", fallback)] if fallback else []


def _extract_citations(text: str) -> list[str]:
    citations: list[str] = []
    for law_name, article_no, paragraph_no, item_no in CITATION_RE.findall(text):
        parts = [f"「{law_name}」", f"제{article_no}조"]
        if paragraph_no:
            parts.append(f"제{paragraph_no}항")
        if item_no:
            parts.append(f"제{item_no}호")
        citations.append(" ".join(parts))
    return sorted(set(citations))


def _build_chunks(doc_id: str, root: ET.Element, title: str, case_no: str, decision_date: str | None) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    meta_text = clean_html_text(
        "\n".join(
            [
                f"제목: {title}",
                f"의안번호: {case_no}",
                f"기관명: {_first_text(root, '기관명')}",
                f"회의종류: {_first_text(root, '회의종류')}",
                f"의결일: {decision_date or ''}",
            ]
        )
    )
    base_sections: list[tuple[str, str, str]] = [("meta", "결정 기본정보", meta_text)]

    holding_text = clean_html_text(_first_text(root, "주문"))
    if holding_text:
        base_sections.append(("holding", "주문", holding_text))

    reason_text = normalize_display_text(_first_text(root, "이유"))
    base_sections.extend(_split_reason_sections(reason_text))

    summary_text = clean_html_text(_first_text(root, "결정요지"))
    if summary_text:
        base_sections.append(("summary", "결정요지", summary_text))

    for order, (chunk_type, heading, text) in enumerate(base_sections, start=1):
        citations = _extract_citations(text)
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_id}:{chunk_type}:{order}",
                doc_id=doc_id,
                chunk_type=chunk_type,
                chunk_order=order,
                heading=heading,
                text=text,
                text_clean=clean_html_text(text),
                citations=citations,
            )
        )
    return chunks


def parse_acr_document(list_item: dict, xml_path: Path, list_page: int) -> DocumentRecord:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    source_id = _first_text(root, "결정문일련번호")
    title = _first_text(root, "제목") or normalize_display_text(list_item.get("제목", ""))
    complaint_title = _first_text(root, "민원표시") or normalize_display_text(list_item.get("민원표시명", ""))
    case_no = _first_text(root, "의안번호") or normalize_display_text(list_item.get("의안번호", ""))
    decision_date = parse_korean_date(_first_text(root, "의결일") or list_item.get("의결일", ""))
    agency = compact_spaces(_first_text(root, "기관명"))
    meeting_type = compact_spaces(_first_text(root, "회의종류"))
    decision_type = compact_spaces(_first_text(root, "결정구분"))
    doc_id = f"acr:{source_id}"
    chunks = _build_chunks(doc_id, root, title, case_no, decision_date)

    return DocumentRecord(
        doc_id=doc_id,
        category="acr",
        doc_type="committee_decision",
        source_id=source_id,
        title=title,
        title_norm=normalize_lookup_text(title),
        complaint_title=complaint_title,
        complaint_title_norm=normalize_lookup_text(complaint_title),
        case_no=case_no,
        case_no_norm=normalize_lookup_text(case_no),
        decision_date=decision_date,
        agency=agency,
        agency_norm=normalize_lookup_text(agency),
        meeting_type=meeting_type,
        decision_type=decision_type,
        applicant_masked=_first_text(root, "신청인"),
        respondent_masked=_first_text(root, "피신청인"),
        source_path=str(xml_path),
        list_page=list_page,
        raw_meta=list_item,
        chunks=chunks,
    )


def iter_acr_records(acr_root: Path) -> list[DocumentRecord]:
    page_paths = sorted((acr_root / "list" / "json").glob("page_*.json"))
    records: list[DocumentRecord] = []
    for page_path in page_paths:
        if page_path.name.endswith(":Zone.Identifier"):
            continue
        page_no = int(page_path.stem.split("_")[1])
        payload = json.loads(page_path.read_text(encoding="utf-8"))
        items = payload.get("Acr", {}).get("acr", [])
        for item in items:
            source_id = normalize_display_text(item.get("결정문일련번호", ""))
            if not source_id:
                continue
            xml_path = acr_root / "body" / "xml" / f"ID_{source_id}.xml"
            if not xml_path.exists():
                continue
            records.append(parse_acr_document(item, xml_path, page_no))
    return records

