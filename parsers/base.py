from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from normalizers.text import normalize_display_text


def xml_text(parent: ET.Element | None, tag: str) -> str:
    if parent is None:
        return ""
    child = parent.find(tag)
    if child is None:
        return ""
    return normalize_display_text("".join(child.itertext()))


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_type: str
    chunk_order: int
    heading: str
    text: str
    text_clean: str
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentRecord:
    doc_id: str
    category: str
    doc_type: str
    source_id: str
    title: str
    title_norm: str
    complaint_title: str
    complaint_title_norm: str
    case_no: str
    case_no_norm: str
    decision_date: str | None
    agency: str
    agency_norm: str
    meeting_type: str
    decision_type: str
    applicant_masked: str
    respondent_masked: str
    source_path: str
    list_page: int
    raw_meta: dict[str, Any]
    chunks: list[ChunkRecord]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        return payload


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
