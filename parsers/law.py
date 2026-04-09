from __future__ import annotations

from pathlib import Path
import re
from xml.etree import ElementTree as ET

from normalizers.text import clean_html_text, normalize_lookup_text
from parsers.base import xml_text as _text


HEADING_ONLY_RE = re.compile(r"^제\s*\d+\s*(?:편|장|절)\s+.+$")


def _merge_numbered_text(number: str, text: str) -> str:
    number = (number or "").strip()
    text = (text or "").strip()
    if not number:
        return text
    if not text:
        return number
    normalized_number = re.sub(r"\s+", "", number)
    normalized_text = re.sub(r"\s+", "", text)
    if normalized_text.startswith(normalized_number):
        return text
    return f"{number} {text}".strip()


def _is_heading_only_text(text: str) -> bool:
    return bool(HEADING_ONLY_RE.match(" ".join((text or "").split())))


def _article_quality(article: dict) -> tuple[int, int, int]:
    title = (article.get("article_title") or "").strip()
    text = (article.get("article_text") or "").strip()
    return (
        0 if _is_heading_only_text(text) else 1,
        1 if title else 0,
        len(text),
    )


def _article_text(article_node: ET.Element) -> str:
    parts: list[str] = []
    main_text = _text(article_node, "조문내용")
    if main_text:
        parts.append(main_text)
    for paragraph in article_node.findall("항"):
        paragraph_no = _text(paragraph, "항번호")
        paragraph_text = _text(paragraph, "항내용")
        if paragraph_no or paragraph_text:
            parts.append(_merge_numbered_text(paragraph_no, paragraph_text))
        for item in paragraph.findall("호"):
            item_no = _text(item, "호번호")
            item_text = _text(item, "호내용")
            if item_no or item_text:
                parts.append(_merge_numbered_text(item_no, item_text))
    return clean_html_text("\n".join(parts))


def parse_law_xml(xml_path: Path) -> dict | None:
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    basic = root.find("기본정보")
    if basic is None:
        return None

    law_id = _text(basic, "법령ID")
    law_name = _text(basic, "법령명_한글")
    if not law_id or not law_name:
        return None

    short_name = _text(basic, "법령명약칭")
    record = {
        "doc_id": f"law:{law_id}",
        "law_id": law_id,
        "law_name": law_name,
        "law_name_norm": normalize_lookup_text(law_name),
        "short_name": short_name,
        "short_name_norm": normalize_lookup_text(short_name),
        "law_type": _text(basic, "법종구분"),
        "ministry": _text(basic, "소관부처"),
        "promulgation_date": _text(basic, "공포일자"),
        "effective_date": _text(basic, "시행일자"),
        "source_path": str(xml_path),
        "articles": [],
    }

    articles_by_no: dict[str, dict] = {}

    for article_node in root.findall("./조문/조문단위"):
        base_article_no = _text(article_node, "조문번호")
        branch_article_no = _text(article_node, "조문가지번호")
        if not base_article_no:
            continue
        article_no = (
            base_article_no
            if not branch_article_no
            else f"{base_article_no}의{branch_article_no}"
        )
        article_text = _article_text(article_node)
        if not article_text:
            continue
        article = {
            "article_no": article_no,
            "article_no_norm": normalize_lookup_text(article_no),
            "article_key": _text(article_node, "조문키"),
            "article_title": _text(article_node, "조문제목"),
            "article_text": article_text,
        }
        if _is_heading_only_text(article_text) and not article["article_title"]:
            continue
        existing = articles_by_no.get(article_no)
        if existing is None or _article_quality(article) > _article_quality(existing):
            articles_by_no[article_no] = article

    record["articles"] = list(articles_by_no.values())

    return record


def iter_law_records(law_root: Path) -> list[dict]:
    records: list[dict] = []
    for xml_path in sorted((law_root / "body" / "xml").glob("ID_*.xml")):
        if xml_path.name.endswith(":Zone.Identifier"):
            continue
        record = parse_law_xml(xml_path)
        if record is not None:
            records.append(record)
    return records
