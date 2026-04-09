from __future__ import annotations

import re
from dataclasses import dataclass

from normalizers.text import normalize_lookup_text


CASE_NO_RE = re.compile(r"제?\d{4}-\d+소위\d+-[가-힣A-Za-z0-9]+호")
SERIAL_RE = re.compile(r"(?:결정문일련번호|id)\s*(\d+)|^\s*(\d+)\s*$", re.I)
LAW_ARTICLE_RE = re.compile(r"^\s*(.+?)\s*제\s*(\d+)(?:조의\s*(\d+)|의\s*(\d+)조|조)\s*$")
PREC_CASE_RE = re.compile(
    r"^\s*(?:[가-힣]+법원[- ]?)?\d{2,4}[가-힣]{1,4}\d+\s*$|^\s*[가-힣]+법원-\d{4}-[가-힣]{1,4}-\d+\s*$"
)
DECC_CASE_RE = re.compile(r"^\s*(?:\d{4}-\d{4,6}|\d{4}[가-힣]+행심\d+)\s*$")
DETC_CASE_RE = re.compile(r"^\s*\d{2,4}헌[가-힣]{1,4}\d+\s*$")
EXPC_ISSUE_RE = re.compile(r"^\s*\d{2}-\d{3,4}\s*$")
SEMANTIC_HINTS = ("왜", "이유", "판단", "요약", "비슷", "유사")
LAW_NAME_SUFFIX_RE = re.compile(r"[가-힣]{2,}(?:법|령|규칙|시행령|시행규칙)")
EXAM_PREFIX_PATTERNS = (
    r"^\s*다음\s*중\s*옳은\s*것은[?:\s]*",
    r"^\s*다음\s*중\s*틀린\s*것은[?:\s]*",
    r"^\s*다음\s*설명\s*중\s*옳은\s*것은[?:\s]*",
    r"^\s*다음\s*설명\s*중\s*틀린\s*것은[?:\s]*",
    r"^\s*옳은\s*것은[?:\s]*",
    r"^\s*틀린\s*것은[?:\s]*",
)


@dataclass
class RouteDecision:
    strategy: str
    raw_query: str
    normalized_query: str
    serial_no: str | None = None
    case_no: str | None = None
    law_name: str | None = None
    article_no: str | None = None
    precedent_case_no: str | None = None
    decc_case_no: str | None = None
    detc_case_no: str | None = None
    expc_issue_no: str | None = None
    fallback_strategies: list[str] | None = None


def route_query(query: str) -> RouteDecision:
    query = query.strip()
    stripped_query = query
    for pattern in EXAM_PREFIX_PATTERNS:
        stripped_query = re.sub(pattern, "", stripped_query, flags=re.I)
    stripped_query = stripped_query.strip() or query
    normalized = normalize_lookup_text(stripped_query)

    fallback_exact = ["fts", "vector"]
    fallback_vector = ["fts"]
    fallback_fts = ["vector"]

    serial_match = SERIAL_RE.search(stripped_query)
    if serial_match:
        serial = serial_match.group(1) or serial_match.group(2)
        return RouteDecision(
            strategy="exact_source_id",
            raw_query=stripped_query,
            normalized_query=normalized,
            serial_no=serial,
            fallback_strategies=fallback_exact,
        )

    case_match = CASE_NO_RE.search(stripped_query.replace(" ", ""))
    if case_match:
        return RouteDecision(
            strategy="exact_case_no",
            raw_query=stripped_query,
            normalized_query=normalized,
            case_no=case_match.group(0),
            fallback_strategies=fallback_exact,
        )

    law_match = LAW_ARTICLE_RE.match(stripped_query)
    if law_match:
        article_main = law_match.group(2)
        article_sub = law_match.group(3) or law_match.group(4)
        article_no = article_main if not article_sub else f"{article_main}의{article_sub}"
        return RouteDecision(
            strategy="exact_law_article",
            raw_query=stripped_query,
            normalized_query=normalized,
            law_name=law_match.group(1).strip(),
            article_no=article_no,
            fallback_strategies=fallback_exact,
        )

    if DECC_CASE_RE.match(stripped_query.replace(" ", "")):
        return RouteDecision(
            strategy="exact_decc_case",
            raw_query=stripped_query,
            normalized_query=normalized,
            decc_case_no=stripped_query.strip(),
            fallback_strategies=fallback_exact,
        )

    if DETC_CASE_RE.match(stripped_query.replace(" ", "")):
        return RouteDecision(
            strategy="exact_detc_case",
            raw_query=stripped_query,
            normalized_query=normalized,
            detc_case_no=stripped_query.strip(),
            fallback_strategies=fallback_exact,
        )

    if PREC_CASE_RE.match(stripped_query.replace(" ", "")):
        return RouteDecision(
            strategy="exact_precedent_case",
            raw_query=stripped_query,
            normalized_query=normalized,
            precedent_case_no=stripped_query.strip(),
            fallback_strategies=fallback_exact,
        )

    if EXPC_ISSUE_RE.match(stripped_query.replace(" ", "")):
        return RouteDecision(
            strategy="exact_expc_issue",
            raw_query=stripped_query,
            normalized_query=normalized,
            expc_issue_no=stripped_query.strip(),
            fallback_strategies=fallback_exact,
        )

    law_name_match = LAW_NAME_SUFFIX_RE.search(stripped_query)
    if law_name_match and not re.search(r"제\s*\d+", stripped_query):
        return RouteDecision(
            strategy="law_name_search",
            raw_query=stripped_query,
            normalized_query=normalized,
            law_name=law_name_match.group(0),
            fallback_strategies=fallback_exact,
        )

    if any(hint in stripped_query for hint in SEMANTIC_HINTS):
        return RouteDecision(
            strategy="vector",
            raw_query=stripped_query,
            normalized_query=normalized,
            fallback_strategies=fallback_vector,
        )

    return RouteDecision(
        strategy="fts",
        raw_query=stripped_query,
        normalized_query=normalized,
        fallback_strategies=fallback_fts,
    )
