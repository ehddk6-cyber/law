from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from xml.etree import ElementTree as ET

from .text import normalize_text


@dataclass
class AcrDocument:
    doc_id: str
    serial_number: str
    title: str
    agenda_number: str
    display_name: str
    decision_type: str
    meeting_type: str
    decision_date: str
    institution: str
    petitioner: str
    respondent: str
    related_agency: str
    order_text: str
    reason_text: str
    summary_text: str
    detail_path: str
    source_page: str

    @property
    def exact_fields(self) -> List[str]:
        return [
            self.serial_number,
            f"결정문일련번호 {self.serial_number}",
            self.title,
            self.agenda_number,
            self.display_name,
        ]

    @property
    def search_text(self) -> str:
        parts = [
            self.title,
            self.display_name,
            self.agenda_number,
            self.serial_number,
            self.decision_type,
            self.meeting_type,
            self.decision_date,
            self.petitioner,
            self.respondent,
            self.related_agency,
            self.summary_text,
            self.order_text,
            self.reason_text,
        ]
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    serial_number: str
    chunk_index: int
    label: str
    text: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


def _text(element: ET.Element | None, tag: str) -> str:
    if element is None:
        return ""
    child = element.find(tag)
    if child is None or child.text is None:
        return ""
    return normalize_text(child.text)


def _load_acr_list_items(corpus_dir: Path) -> Iterable[Tuple[dict, str]]:
    list_dir = corpus_dir / "list" / "json"
    for page_path in sorted(list_dir.glob("page_*.json")):
        with page_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for item in payload["Acr"]["acr"]:
            yield item, page_path.name


def _parse_acr_xml(path: Path) -> ET.Element:
    raw = path.read_text(encoding="utf-8")
    return ET.fromstring(raw)


def load_acr_documents(base_dir: Path) -> List[AcrDocument]:
    corpus_dir = base_dir / "acr"
    documents: List[AcrDocument] = []
    for item, page_name in _load_acr_list_items(corpus_dir):
        serial_number = normalize_text(item.get("결정문일련번호", ""))
        detail_path = corpus_dir / "body" / "xml" / f"ID_{serial_number}.xml"
        root = _parse_acr_xml(detail_path)
        decision = root.find("의결서")
        documents.append(
            AcrDocument(
                doc_id=f"acr:{serial_number}",
                serial_number=serial_number,
                title=normalize_text(item.get("제목", "")),
                agenda_number=normalize_text(item.get("의안번호", "")),
                display_name=normalize_text(item.get("민원표시명", "")),
                decision_type=normalize_text(item.get("결정구분", "")),
                meeting_type=normalize_text(item.get("회의종류", "")),
                decision_date=normalize_text(item.get("의결일", "")),
                institution=_text(decision, "기관명"),
                petitioner=_text(decision, "신청인"),
                respondent=_text(decision, "피신청인"),
                related_agency=_text(decision, "관계기관"),
                order_text=_text(decision, "주문"),
                reason_text=_text(decision, "이유"),
                summary_text=_text(decision, "결정요지"),
                detail_path=str(detail_path.relative_to(base_dir)),
                source_page=page_name,
            )
        )
    return documents


def chunk_acr_document(document: AcrDocument, *, chunk_size: int = 900, overlap: int = 150) -> List[ChunkRecord]:
    seed_sections = [
        ("summary", document.summary_text),
        ("order", document.order_text),
        ("reason", document.reason_text),
    ]
    chunks: List[ChunkRecord] = []
    chunk_index = 0
    for label, section_text in seed_sections:
        text = normalize_text(section_text)
        if not text:
            continue
        if len(text) <= chunk_size:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}:{chunk_index}",
                    doc_id=document.doc_id,
                    serial_number=document.serial_number,
                    chunk_index=chunk_index,
                    label=label,
                    text=text,
                )
            )
            chunk_index += 1
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            window = text[start:end]
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document.doc_id}:{chunk_index}",
                    doc_id=document.doc_id,
                    serial_number=document.serial_number,
                    chunk_index=chunk_index,
                    label=label,
                    text=window,
                )
            )
            chunk_index += 1
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def inspect_category_structures(base_dir: Path) -> Dict[str, dict]:
    report: Dict[str, dict] = {}

    def read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    for category in ("prec", "decc", "expc"):
        meta = read_json(base_dir / category / "meta.json")
        list_payload = read_json(sorted((base_dir / category / "list" / "json").glob("page_*.json"))[0])
        top_key = next(iter(list_payload))
        section = list_payload[top_key]
        list_key = next(key for key, value in section.items() if isinstance(value, list))
        item = section[list_key][0]
        sample_files = sorted((base_dir / category / "body" / "xml").glob("ID_*.xml"))
        sample_path = next(path for path in sample_files if not path.name.endswith(".Zone.Identifier"))
        sample_text = sample_path.read_text(encoding="utf-8")
        report[category] = {
            "meta": meta,
            "list_root_key": top_key,
            "list_items_key": list_key,
            "list_item_fields": sorted(item.keys()),
            "sample_body_path": str(sample_path.relative_to(base_dir)),
            "sample_body_prefix": sample_text[:400],
            "notes": _category_notes(category, sample_text),
        }
    return report


def _category_notes(category: str, sample_text: str) -> List[str]:
    if category == "prec":
        return ["판례 XML fields are stable and directly indexable."]
    if category == "decc":
        notes = ["행정심판 XML uses fields compatible with precedent-style parsing."]
        if "일치하는 행정심판례가 없습니다" in sample_text:
            notes.append("Some body files are error payloads and must be filtered while indexing.")
        else:
            notes.append("List identifier key is 행정심판재결례일련번호, not the generic ID field.")
        return notes
    return ["법령해석례 XML contains 질의요지, 회답, 이유 sections suitable for chunking."]

