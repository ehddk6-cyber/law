from __future__ import annotations

import re
import unicodedata


WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")


def collapse_whitespace(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value).strip()


def clean_html_text(value: str) -> str:
    without_tags = HTML_TAG_RE.sub(" ", value or "")
    return collapse_whitespace(without_tags)


def normalize_display_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    return collapse_whitespace(normalized)


def normalize_lookup_text(value: str) -> str:
    normalized = normalize_display_text(value).lower()
    normalized = PUNCT_RE.sub("", normalized)
    return normalized.strip()


def compact_spaces(value: str) -> str:
    return normalize_display_text(value).replace(" ", "")

