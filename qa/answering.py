from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import json
from pathlib import Path
import re

from qa.agent_exec import run_agent
from qa.cli import format_article_no
from qa.exact_verifier import KoreanLawExactVerifier
from qa.llm_prompting import (
    JUDGE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_agent_prompt,
    build_judge_prompt,
    build_llm_user_prompt,
)
from qa.response_schema import SCHEMA_VERSION
from normalizers.text import normalize_lookup_text
from providers import create_provider, resolve_provider_options
from qa.unified import UnifiedResult, UnifiedSearcher


@dataclass
class AnswerPacket:
    query: str
    scope: str
    question_type: str
    route: str
    status: str
    grounded: bool
    answer: str
    evidence: list[str]
    citations: list[str]
    warnings: list[str]
    source_type: str | None
    doc_id: str | None
    explain_style: str | None = None
    teaching_explanation: dict | None = None
    option_reviews: list[dict] | None = None
    exam_assessment: dict | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_answer: str | None = None
    llm_error: str | None = None
    response_mode: str | None = None
    final_answer_source: str | None = None
    judge_mode: str | None = None
    judge_result: dict | None = None
    judge_error: str | None = None
    judge_trace: list[str] | None = None
    judge_runs: dict[str, dict] | None = None
    agent_prompts: dict[str, str] | None = None
    agent_runs: dict[str, dict] | None = None
    fallback_trace: list[str] | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


def _clean_snippet(text: str, limit: int = 320) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _summarize_key_issue(query: str) -> str:
    stem = _extract_question_stem(query).strip()
    if _detect_question_type(query) == "exam_ox":
        stem = OX_SUFFIX_RE.sub("", stem).strip()
    if stem:
        return _clean_snippet(stem, limit=90)
    return "보기별 진술의 법적 정합성 판단"


def _format_ground_rule(citations: list[str], exam_assessment: dict | None) -> str:
    parts: list[str] = []
    if citations:
        parts.append(", ".join(citations[:2]))
    if exam_assessment and exam_assessment.get("reason"):
        parts.append(str(exam_assessment["reason"]))
    if not parts:
        return "직접 근거를 요약할 정보가 부족합니다."
    return " / ".join(parts)


def _review_result(review: dict) -> str:
    return str(review.get("effective_support_level") or review.get("support_level") or "indeterminate")


def _review_reason(review: dict) -> str:
    return str(review.get("effective_support_reason") or review.get("support_reason") or "근거가 부족합니다.")


def _build_wrong_point(option_reviews: list[dict] | None) -> str:
    if not option_reviews:
        return "오답 포인트를 요약할 정보가 부족합니다."
    unsupported = [review for review in option_reviews if _review_result(review) == "unsupported"]
    if unsupported:
        labels = ", ".join(str(review["label"]) for review in unsupported)
        reasons = _dedupe_texts([_review_reason(review) for review in unsupported])
        return f"{labels}는 근거와 충돌합니다. " + " / ".join(reasons[:2])
    indeterminate = [review for review in option_reviews if _review_result(review) == "indeterminate"]
    if indeterminate:
        return "근거 부족으로 오답 포인트를 확정하기 어렵습니다."
    supported = [review for review in option_reviews if _review_result(review) == "supported"]
    if supported:
        return "현재 grounded 근거와 직접 충돌하는 보기나 진술은 확인되지 않았습니다."
    return "오답 포인트를 요약할 정보가 부족합니다."


def _build_teaching_summary(packet: AnswerPacket) -> str:
    recommended = (packet.exam_assessment or {}).get("recommended_options") or []
    joined = ", ".join(recommended)
    if packet.status == "verification_failed":
        return "로컬 근거와 공식 검증이 충돌해 재검토가 우선입니다."
    if packet.status == "candidate_answer" and packet.grounded and joined:
        return f"현재 근거상 {joined}가 정답 후보입니다."
    if packet.status == "needs_review" and joined:
        return f"현재 근거상 {joined}가 후보지만 자동 확정은 보류됩니다."
    if packet.status == "needs_review":
        return "현재 근거만으로는 정답을 자동 확정하기 어렵습니다."
    if packet.status == "grounded" and joined:
        return f"현재 근거상 {joined}를 우선 검토할 수 있습니다."
    return _clean_snippet(packet.answer, limit=90)


def _build_caution(packet: AnswerPacket) -> str:
    warning_text = " ".join(packet.warnings or [])
    if packet.status == "verification_failed":
        return "로컬 근거와 공식 검증이 충돌했습니다."
    if packet.status == "needs_review":
        if "공식" in warning_text:
            return "일부 근거만 확인됐고 자동 확정은 보류됩니다."
        return "자동 확정 보류 상태입니다."
    if packet.grounded:
        return "공식 검증을 통과한 grounded 결과입니다."
    return "추가 검토가 필요합니다."


def _build_option_breakdown(option_reviews: list[dict] | None) -> list[dict]:
    if not option_reviews:
        return []
    rows: list[dict] = []
    for review in option_reviews:
        rows.append(
            {
                "label": review.get("label"),
                "text": review.get("text"),
                "result": _review_result(review),
                "reason": _review_reason(review),
                "citations": list(review.get("citations") or []),
                "grounded": bool(review.get("grounded")),
            }
        )
    return rows


def _build_admin_exam_explanation(packet: AnswerPacket) -> dict | None:
    if packet.question_type not in {"exam_ox", "exam_choice"}:
        return None
    return {
        "summary": _build_teaching_summary(packet),
        "answer_candidate": list((packet.exam_assessment or {}).get("recommended_options") or []),
        "key_issue": _summarize_key_issue(packet.query),
        "ground_rule": _format_ground_rule(packet.citations, packet.exam_assessment),
        "option_breakdown": _build_option_breakdown(packet.option_reviews),
        "wrong_point": _build_wrong_point(packet.option_reviews),
        "caution": _build_caution(packet),
    }


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _detect_question_type(query: str) -> str:
    compact = "".join(query.split())
    if any(token in query for token in ("옳은", "틀린", "고른", "다음 중", "설명으로")):
        return "exam_choice"
    if any(token in query for token in ("OX", "ox", "o/x")):
        return "exam_ox"
    if "사례" in query or "문제" in query:
        return "exam_case"
    if "판례" in query or "결정례" in query or "해석례" in query:
        return "authority_lookup"
    if "제" in query and "조" in query:
        return "statute_lookup"
    if (
        compact
        and any(marker in compact for marker in ("헌", "누", "도", "다", "마"))
        and any(char.isdigit() for char in compact)
    ):
        return "case_lookup"
    return "exam_general"


OPTION_SPLIT_RE = re.compile(
    r"(?:^|\n|\s)(①|②|③|④|⑤|1\.|2\.|3\.|4\.|5\.|1\)|2\)|3\)|4\)|5\))\s*"
)
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
NEGATION_MARKERS = ("아니다", "않다", "없다", "불가", "금지", "못한다", "배제", "제외")
ANTONYM_PAIRS = (
    ("할수있다", "할수없다"),
    ("할수있다", "못한다"),
    ("포함", "제외"),
    ("허용", "금지"),
    ("가능", "불가"),
    ("원칙", "예외"),
    ("인정", "불인정"),
    ("적법", "위법"),
    ("유효", "무효"),
)
INLINE_LAW_ARTICLE_RE = re.compile(
    r"([가-힣A-Za-z0-9·\s]+?법)\s*제\s*(\d+)(?:조의\s*(\d+)|의\s*(\d+)조|조)"
)
INLINE_CASE_NO_RE = re.compile(
    r"(\d{2,4}헌[가-힣]{1,4}\d+|\d{2,4}[가-힣]{1,4}\d+|\d{4}-\d{4,6}|\d{2}-\d{3,4})"
)
OX_SUFFIX_RE = re.compile(
    r"\(?\s*[OoXx]\s*/\s*[OoXx]\s*\)?$|\s*OX\s*$|\s*ox\s*$|\s*[OoXx]\s*$"
)
SUBSTATEMENT_RE = re.compile(r"(ㄱ\.|ㄴ\.|ㄷ\.|ㄹ\.)\s*")


def _extract_choice_options(query: str) -> list[tuple[str, str]]:
    matches = list(OPTION_SPLIT_RE.finditer(query))
    if len(matches) < 2:
        return []
    options: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        text = query[start:end].strip(" \n\t-:")
        if text:
            options.append((label, text))
    return options


def _extract_question_stem(query: str) -> str:
    matches = list(OPTION_SPLIT_RE.finditer(query))
    if not matches:
        return query.strip()
    return query[: matches[0].start()].strip()


def _extract_substatements(text: str) -> list[tuple[str, str]]:
    matches = list(SUBSTATEMENT_RE.finditer(text))
    if not matches:
        return []
    items: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        item_text = text[start:end].strip(" \n\t-:")
        if item_text:
            items.append((label, item_text))
    return items


def _extract_referenced_substatement_labels(text: str) -> list[str]:
    labels = re.findall(r"(ㄱ|ㄴ|ㄷ|ㄹ)\.?", text)
    normalized = []
    for label in labels:
        normalized_label = f"{label}."
        if normalized_label not in normalized:
            normalized.append(normalized_label)
    return normalized


def _is_all_correct_selection_text(text: str) -> bool:
    condensed = "".join(text.split())
    condensed = re.sub(r"[^\wㄱ-ㄹ]", "", condensed)
    markers = (
        "모두고른것",
        "모두고른것만",
        "옳은것만",
        "옳은것을모두",
        "옳은것만을모두",
        "전부고른것",
        "전부고른것만",
        "전부",
    )
    return any(marker in condensed for marker in markers)


def _is_combo_reference_text(text: str) -> bool:
    labels = _extract_referenced_substatement_labels(text)
    if labels:
        normalized = re.sub(r"[ㄱ-ㄹ\.,·와및또는그리고\s]", "", text)
        normalized = (
            normalized.replace("만을", "")
            .replace("만이", "")
            .replace("만이다", "")
            .replace("만인", "")
            .replace("만", "")
        )
        normalized = (
            normalized.replace("고른것", "")
            .replace("고른", "")
            .replace("것", "")
            .replace("모두", "")
            .replace("전부", "")
        )
        return not normalized
    return _is_all_correct_selection_text(text)


def _is_count_only_text(text: str) -> bool:
    normalized = re.sub(r"\s", "", text)
    return bool(
        re.fullmatch(r"(?:[0-5]|[0-5]개|한개|두개|세개|네개|다섯개)", normalized)
    )


def _extract_count_value(text: str) -> int | None:
    normalized = re.sub(r"\s", "", text)
    mapping = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "0개": 0,
        "1개": 1,
        "2개": 2,
        "3개": 3,
        "4개": 4,
        "5개": 5,
        "한개": 1,
        "두개": 2,
        "세개": 3,
        "네개": 4,
        "다섯개": 5,
    }
    return mapping.get(normalized)


def _is_all_correct_selection_query(query: str) -> bool:
    condensed = "".join(query.split())
    return (
        "옳은것만" in condensed
        or "옳은것은" in condensed
        or "옳은것을모두" in condensed
        or "옳은것만을모두" in condensed
    )


def _is_all_incorrect_selection_query(query: str) -> bool:
    condensed = "".join(query.split())
    return (
        "틀린것만" in condensed
        or "틀린것은" in condensed
        or "틀린것을모두" in condensed
        or "틀린것만을모두" in condensed
        or "오답만" in condensed
        or "오답을모두" in condensed
    )


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_lookup_text(text))


def _overlap_ratio(source_text: str, statement_text: str) -> float:
    source_tokens = set(_tokenize(source_text))
    statement_tokens = set(_tokenize(statement_text))
    if not statement_tokens:
        return 0.0
    overlap = len(source_tokens & statement_tokens)
    return overlap / max(len(statement_tokens), 1)


def _strip_anchor_text(option_text: str, packet: AnswerPacket) -> str:
    stripped = option_text
    for citation in packet.citations:
        cleaned = citation
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", cleaned)
        cleaned = cleaned.split(" / ")[0]
        if cleaned:
            stripped = stripped.replace(cleaned, " ")
    return " ".join(stripped.split())


def _normalize_statement_descriptor(text: str) -> str:
    cleaned = " ".join(text.split())
    cleaned = re.sub(r"^(은|는|이|가)\s*", "", cleaned)
    cleaned = re.sub(r"\s*에\s*관한\s*", " ", cleaned)
    cleaned = re.sub(
        r"(에\s*관한\s*조항이다|에\s*관한\s*조항|사건이다|사건이다\.|조항이다|조항이다\.|이다\.?|이다)$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(이다|이다\.|이다\?|다\.)$", "", cleaned)
    return " ".join(cleaned.split()).strip()


def _classify_option_support(option_text: str, packet: AnswerPacket) -> tuple[str, str]:
    if not packet.grounded:
        return ("indeterminate", "exact 근거를 확보하지 못했습니다.")

    evidence_text = " ".join(packet.evidence)
    statement_core = _strip_anchor_text(option_text, packet)
    if not statement_core:
        return ("supported", "보기 문장이 사실상 조문/사건번호 식별자 자체입니다.")

    descriptor = _normalize_statement_descriptor(statement_core)
    normalized_descriptor = normalize_lookup_text(descriptor)
    if normalized_descriptor:
        for evidence_line in packet.evidence:
            rhs = (
                evidence_line.split(":", 1)[1]
                if ":" in evidence_line
                else evidence_line
            )
            normalized_rhs = normalize_lookup_text(rhs)
            if normalized_descriptor and (
                normalized_descriptor in normalized_rhs
                or normalized_rhs in normalized_descriptor
            ):
                return (
                    "supported",
                    f"anchor 뒤 설명어가 근거 요지와 직접 일치합니다: {descriptor}",
                )

    normalized_statement = normalize_lookup_text(statement_core)
    for positive, negative in ANTONYM_PAIRS:
        statement_has_positive = positive in normalized_statement
        statement_has_negative = negative in normalized_statement
        evidence_has_positive = positive in normalize_lookup_text(evidence_text)
        evidence_has_negative = negative in normalize_lookup_text(evidence_text)
        if statement_has_positive and evidence_has_negative:
            return (
                "unsupported",
                f"보기 문장은 '{positive}'인데 근거는 '{negative}' 쪽에 가깝습니다.",
            )
        if statement_has_negative and evidence_has_positive:
            return (
                "unsupported",
                f"보기 문장은 '{negative}'인데 근거는 '{positive}' 쪽에 가깝습니다.",
            )
        if statement_has_positive and evidence_has_positive:
            return (
                "supported",
                f"보기 문장과 근거가 '{positive}' 표현에서 일치합니다.",
            )
        if statement_has_negative and evidence_has_negative:
            return (
                "supported",
                f"보기 문장과 근거가 '{negative}' 표현에서 일치합니다.",
            )

    title_hits = []
    for citation in packet.citations:
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", citation).split(" / ")[0].strip()
        normalized_cleaned = normalize_lookup_text(cleaned)
        if normalized_cleaned and (
            normalized_cleaned in normalized_statement
            or normalized_statement in normalized_cleaned
        ):
            title_hits.append(cleaned)
    for evidence_line in packet.evidence:
        if ":" in evidence_line:
            _, rhs = evidence_line.split(":", 1)
        else:
            rhs = evidence_line
        normalized_rhs = normalize_lookup_text(rhs)
        if normalized_rhs and (
            normalized_rhs in normalized_statement
            or normalized_statement in normalized_rhs
        ):
            title_hits.append(rhs.strip())
    if title_hits:
        return (
            "supported",
            f"보기 문장의 핵심 표현이 근거 제목/요지와 직접 일치합니다: {title_hits[0]}",
        )

    ratio = _overlap_ratio(evidence_text, statement_core)
    has_negation = any(marker in statement_core for marker in NEGATION_MARKERS)
    evidence_has_negation = any(marker in evidence_text for marker in NEGATION_MARKERS)

    if ratio >= 0.55 and (not has_negation or evidence_has_negation):
        return (
            "supported",
            f"보기 문장과 근거 원문 토큰 겹침 비율이 높습니다({ratio:.2f}).",
        )
    if ratio <= 0.15 and has_negation != evidence_has_negation:
        return (
            "unsupported",
            f"보기 문장과 근거 원문이 부정 표현 또는 핵심 토큰에서 충돌합니다({ratio:.2f}).",
        )
    if ratio <= 0.10:
        return (
            "indeterminate",
            f"보기 문장과 근거 원문 사이의 직접 겹침이 낮습니다({ratio:.2f}).",
        )
    return (
        "indeterminate",
        f"부분 겹침은 있으나 진위를 자동 확정하기엔 부족합니다({ratio:.2f}).",
    )


def _extract_anchor_query(option_text: str) -> str | None:
    law_match = INLINE_LAW_ARTICLE_RE.search(option_text)
    if law_match:
        article_main = law_match.group(2)
        article_sub = law_match.group(3) or law_match.group(4)
        article = article_main if not article_sub else f"{article_main}의{article_sub}"
        return (
            f"{law_match.group(1).strip()} 제{article}조"
            if "의" not in article
            else f"{law_match.group(1).strip()} 제{article_main}조의{article_sub}"
        )

    case_match = INLINE_CASE_NO_RE.search(option_text)
    if case_match:
        return case_match.group(1)
    return None


def _assess_choice_query(query: str, option_reviews: list[dict]) -> dict:
    supported_labels = [
        review["label"]
        for review in option_reviews
        if review.get("effective_support_level", review["support_level"]) == "supported"
    ]
    unsupported_labels = [
        review["label"]
        for review in option_reviews
        if review.get("effective_support_level", review["support_level"])
        == "unsupported"
    ]
    grounded_labels = [
        review["label"] for review in option_reviews if review["grounded"]
    ]
    all_correct_supported = []
    all_correct_indeterminate = []
    all_incorrect_supported = []
    all_incorrect_indeterminate = []
    for review in option_reviews:
        referenced = review.get("referenced_labels")
        sub_assessment = review.get("sub_assessment")
        if referenced and sub_assessment:
            supported_set = set(sub_assessment["supported_labels"])
            indeterminate_set = set(sub_assessment["indeterminate_labels"])
            unsupported_set = set(sub_assessment["unsupported_labels"])
            referenced_set = set(referenced)
            if referenced_set == supported_set and not indeterminate_set:
                all_correct_supported.append(review["label"])
            elif referenced_set & indeterminate_set:
                all_correct_indeterminate.append(review["label"])
            if referenced_set == unsupported_set and not indeterminate_set:
                all_incorrect_supported.append(review["label"])
            elif referenced_set & indeterminate_set:
                all_incorrect_indeterminate.append(review["label"])

    if _is_all_incorrect_selection_query(query) and any(
        review.get("referenced_labels") for review in option_reviews
    ):
        if len(all_incorrect_supported) == 1:
            return {
                "status": "candidate_answer",
                "recommended_options": all_incorrect_supported,
                "reason": "unsupported 하위 진술만 정확히 포함한 보기가 1개입니다.",
            }
        if len(all_incorrect_supported) > 1:
            return {
                "status": "multiple_candidates",
                "recommended_options": all_incorrect_supported,
                "reason": "unsupported 하위 진술만 포함한 보기가 여러 개라 자동 확정할 수 없습니다.",
            }
        return {
            "status": "insufficient_grounding",
            "recommended_options": all_incorrect_indeterminate,
            "reason": "unsupported 하위 진술 집합을 하나로 확정하지 못했습니다.",
        }

    if _is_all_correct_selection_query(query) and any(
        review.get("referenced_labels") for review in option_reviews
    ):
        if len(all_correct_supported) == 1:
            return {
                "status": "candidate_answer",
                "recommended_options": all_correct_supported,
                "reason": "supported 하위 진술만 정확히 포함한 보기가 1개입니다.",
            }
        if len(all_correct_supported) > 1:
            return {
                "status": "multiple_candidates",
                "recommended_options": all_correct_supported,
                "reason": "supported 하위 진술만 포함한 보기가 여러 개라 자동 확정할 수 없습니다.",
            }
        return {
            "status": "insufficient_grounding",
            "recommended_options": all_correct_indeterminate,
            "reason": "supported 하위 진술 집합을 하나로 확정하지 못했습니다.",
        }

    if any(
        token in query
        for token in (
            "몇 개",
            "몇개",
            "개수",
            "옳은 개수",
            "옳은 것의 개수",
            "옳은 것은 모두 몇",
            "모두 몇개",
            "모두 몇 개",
        )
    ):
        stem_sub_assessment = next(
            (
                review.get("sub_assessment")
                for review in option_reviews
                if review.get("sub_assessment")
            ),
            None,
        )
        if stem_sub_assessment:
            if "틀린" in query and "옳은" not in query:
                supported_count = len(stem_sub_assessment["unsupported_labels"])
            else:
                supported_count = len(stem_sub_assessment["supported_labels"])
            indeterminate_count = len(stem_sub_assessment["indeterminate_labels"])
        else:
            if "틀린" in query and "옳은" not in query:
                supported_count = len(unsupported_labels)
            else:
                supported_count = len(supported_labels)
            indeterminate_count = len(
                [
                    review
                    for review in option_reviews
                    if review.get("effective_support_level", review["support_level"])
                    == "indeterminate"
                ]
            )
        count_labels = [
            review["label"]
            for review in option_reviews
            if _extract_count_value(review["text"]) == supported_count
        ]
        if len(count_labels) == 1:
            return {
                "status": "candidate_answer",
                "count_answer": supported_count,
                "recommended_options": count_labels,
                "reason": (
                    f"{'unsupported' if '틀린' in query and '옳은' not in query else 'supported'} "
                    f"보기 수는 {supported_count}개이고, 이에 대응하는 보기가 1개입니다."
                ),
            }
        if len(count_labels) > 1:
            return {
                "status": "multiple_candidates",
                "count_answer": supported_count,
                "recommended_options": count_labels,
                "reason": (
                    f"{'unsupported' if '틀린' in query and '옳은' not in query else 'supported'} "
                    f"보기 수 {supported_count}개에 대응하는 보기가 여러 개입니다."
                ),
            }
        return {
            "status": "insufficient_grounding",
            "count_answer": supported_count,
            "recommended_options": [],
            "reason": (
                f"supported 보기 수 {supported_count}개에 대응하는 보기를 찾지 못했습니다."
                if indeterminate_count == 0
                else "하위 진술 중 exact 근거를 확보하지 못한 항목이 있어 개수를 확정할 수 없습니다."
            ),
        }

    if "옳은" in query:
        if len(supported_labels) == 1:
            return {
                "status": "candidate_answer",
                "recommended_options": supported_labels,
                "reason": "supported로 분류된 보기가 1개입니다.",
            }
        if len(supported_labels) > 1:
            return {
                "status": "multiple_candidates",
                "recommended_options": supported_labels,
                "reason": "supported로 분류된 보기가 여러 개라 자동 확정할 수 없습니다.",
            }
        return {
            "status": "insufficient_grounding",
            "recommended_options": [],
            "reason": "supported로 분류된 보기가 없습니다.",
        }

    if "틀린" in query:
        if len(unsupported_labels) == 1:
            return {
                "status": "candidate_answer",
                "recommended_options": unsupported_labels,
                "reason": "unsupported로 분류된 보기가 1개입니다.",
            }
        if len(unsupported_labels) > 1:
            return {
                "status": "multiple_candidates",
                "recommended_options": unsupported_labels,
                "reason": "unsupported로 분류된 보기가 여러 개라 자동 확정할 수 없습니다.",
            }
        return {
            "status": "insufficient_grounding",
            "recommended_options": [],
            "reason": "unsupported로 분류된 보기가 없어 틀린 보기를 자동 확정할 수 없습니다.",
        }

    return {
        "status": "review_only",
        "recommended_options": grounded_labels,
        "reason": "문항 유형상 grounded 보기 후보만 제시합니다.",
    }


def _law_answer(query: str, route: str, payload: dict) -> AnswerPacket:
    article_no = format_article_no(payload["article_no"])
    answer = f"결론: {payload['law_name']} {article_no} {payload['article_title']}의 내용을 근거로 답변합니다."
    evidence = [
        f"근거 조문: {payload['law_name']} {article_no} {payload['article_title']}",
        f"원문 요약: {_clean_snippet(payload['article_text'], limit=500)}",
    ]
    citations = [f"[법령] {payload['law_name']} {article_no}"]
    warnings = ["주의: 해석을 덧붙이지 않고 검색된 조문 범위 안에서만 답변했습니다."]
    return AnswerPacket(
        query,
        "all-laws-exam",
        _detect_question_type(query),
        route,
        "grounded",
        True,
        answer,
        evidence,
        citations,
        warnings,
        "law",
        payload["doc_id"],
    )


def _acr_answer(query: str, route: str, payload: dict) -> AnswerPacket:
    holding = ""
    reasoning = ""
    for chunk in payload.get("chunks", []):
        if chunk["chunk_type"] == "holding" and not holding:
            holding = chunk["text"]
        if chunk["chunk_type"].startswith("reason") and not reasoning:
            reasoning = chunk["text"]
    parts = []
    if holding:
        parts.append(f"주문: {_clean_snippet(holding)}")
    if reasoning:
        parts.append(f"판단 근거: {_clean_snippet(reasoning)}")
    if not parts:
        parts.append(
            "검색된 결정문에서 주문 또는 판단 부분을 충분히 추출하지 못했습니다."
        )
    answer = f"결론: {payload['title']} 관련 결정문을 근거로 답변합니다."
    evidence = parts
    citations = [
        f"[결정문] {payload['title']} / {payload['case_no']} / {payload['decision_date'] or '일자 미상'}"
    ]
    warnings = ["주의: 결정문 원문 청크에서 직접 추출한 내용만 사용했습니다."]
    return AnswerPacket(
        query,
        "all-laws-exam",
        _detect_question_type(query),
        route,
        "grounded",
        True,
        answer,
        evidence,
        citations,
        warnings,
        "acr",
        payload["doc_id"],
    )


def _prec_answer(query: str, route: str, payload: dict) -> AnswerPacket:
    answer = (
        f"결론: {payload['case_no']} {payload['case_name']} 판례를 근거로 답변합니다."
    )
    evidence = [
        f"판결 요지: {_clean_snippet(payload['holding'])}",
        f"쟁점: {_clean_snippet(payload['issue'])}",
    ]
    citations = [
        f"[판례] {payload['court_name']} {payload['case_no']} / {payload['decision_date'] or '일자 미상'}"
    ]
    warnings = ["주의: 판례 원문에서 확보된 요지 범위만 반영했습니다."]
    return AnswerPacket(
        query,
        "all-laws-exam",
        _detect_question_type(query),
        route,
        "grounded",
        True,
        answer,
        evidence,
        citations,
        warnings,
        "prec",
        payload["doc_id"],
    )


def _decc_answer(query: str, route: str, payload: dict) -> AnswerPacket:
    answer = f"결론: {payload['case_no']} {payload['case_name']} 행정심판례를 근거로 답변합니다."
    evidence = [
        f"재결 주문: {_clean_snippet(payload['order_text'])}",
        f"청구 취지: {_clean_snippet(payload['claim_text'])}",
    ]
    citations = [
        f"[행정심판례] {payload['agency']} {payload['case_no']} / {payload['decision_date'] or '일자 미상'}"
    ]
    warnings = ["주의: 재결례 원문에 없는 추가 해석은 제외했습니다."]
    return AnswerPacket(
        query,
        "all-laws-exam",
        _detect_question_type(query),
        route,
        "grounded",
        True,
        answer,
        evidence,
        citations,
        warnings,
        "decc",
        payload["doc_id"],
    )


def _detc_answer(query: str, route: str, payload: dict) -> AnswerPacket:
    core = (
        payload.get("decision_summary")
        or payload.get("issue")
        or payload.get("content")
        or ""
    )
    answer = f"결론: {payload['case_no']} {payload['case_name']} 헌재결정례를 근거로 답변합니다."
    evidence = [f"핵심 부분: {_clean_snippet(core)}"]
    citations = [
        f"[헌재결정례] {payload['case_no']} / {payload['decision_date'] or '일자 미상'}"
    ]
    warnings = ["주의: 전문·결정요지 중 비어 있지 않은 원문 부분만 사용했습니다."]
    return AnswerPacket(
        query,
        "all-laws-exam",
        _detect_question_type(query),
        route,
        "grounded",
        True,
        answer,
        evidence,
        citations,
        warnings,
        "detc",
        payload["doc_id"],
    )


def _expc_answer(query: str, route: str, payload: dict) -> AnswerPacket:
    answer = f"결론: {payload['issue_no']} {payload['title']} 법령해석례를 근거로 답변합니다."
    evidence = [
        f"회답: {_clean_snippet(payload['answer_text'])}",
        f"질의 요지: {_clean_snippet(payload['query_summary'])}",
    ]
    citations = [
        f"[법령해석례] {payload['agency']} {payload['issue_no']} / {payload['decision_date'] or '일자 미상'}"
    ]
    warnings = ["주의: 법령해석례 원문 회답을 우선하고 별도 추론은 하지 않았습니다."]
    return AnswerPacket(
        query,
        "all-laws-exam",
        _detect_question_type(query),
        route,
        "grounded",
        True,
        answer,
        evidence,
        citations,
        warnings,
        "expc",
        payload["doc_id"],
    )


def _fallback_answer(
    query: str, route: str, results: list[UnifiedResult]
) -> AnswerPacket:
    lines = []
    citations: list[str] = []
    for result in results[:3]:
        payload = result.payload
        if result.source_type == "acr":
            lines.append(f"[결정문] {payload['title']} / {payload['case_no']}")
            citations.append(f"[결정문] {payload['doc_id']}")
        elif result.source_type == "prec":
            lines.append(f"[판례] {payload['case_no']} {payload['case_name']}")
            citations.append(f"[판례] {payload['doc_id']}")
        elif result.source_type == "decc":
            lines.append(f"[행정심판례] {payload['case_no']} {payload['case_name']}")
            citations.append(f"[행정심판례] {payload['doc_id']}")
        elif result.source_type == "detc":
            lines.append(f"[헌재결정례] {payload['case_no']} {payload['case_name']}")
            citations.append(f"[헌재결정례] {payload['doc_id']}")
        elif result.source_type == "expc":
            lines.append(f"[법령해석례] {payload['issue_no']} {payload['title']}")
            citations.append(f"[법령해석례] {payload['doc_id']}")
    answer = "결론 보류: LAW OPEN DATA 안에서 질문과 직접 매칭되는 단일 근거를 확정하지 못했습니다."
    evidence = ["후보 문서: " + " / ".join(lines)] if lines else []
    warnings = ["주의: exact match가 아니므로 단정 답변을 하지 않았습니다."]
    first = results[0] if results else None
    return AnswerPacket(
        query=query,
        scope="all-laws-exam",
        question_type=_detect_question_type(query),
        route=route,
        status="needs_review",
        grounded=False,
        answer=answer,
        evidence=evidence,
        citations=citations,
        warnings=warnings,
        source_type=first.source_type if first else None,
        doc_id=first.payload["doc_id"] if first else None,
    )


class GroundedAnswerer:
    def __init__(self, root: Path, verifier: KoreanLawExactVerifier | None = None):
        self.root = root
        self.searcher = UnifiedSearcher(root)
        self.verifier = verifier or KoreanLawExactVerifier()

    def _apply_exact_verification(
        self,
        packet: AnswerPacket,
        decision_strategy: str,
        top: UnifiedResult,
    ) -> AnswerPacket:
        if not decision_strategy.startswith("exact_"):
            return packet
        verification = self.verifier.verify(decision_strategy, top)
        if verification is None:
            return packet

        if verification.grounded:
            if top.source_type == "law" and verification.official_text:
                payload = top.payload
                article_no = format_article_no(payload["article_no"])
                article_title = (payload.get("article_title") or "").strip()
                packet.answer = (
                    f"결론: {payload['law_name']} {article_no}"
                    f"{(' ' + article_title) if article_title else ''} 조문을 공식 조회와 교차검증했습니다."
                )
                packet.evidence = [
                    f"공식 검증 조문: {payload['law_name']} {article_no}"
                    f"{(' ' + article_title) if article_title else ''}",
                    f"공식 조문 전문: {_clean_snippet(verification.official_text, limit=800)}",
                ]
            packet.citations = _dedupe_texts(
                packet.citations
                + ([f"[공식검증] {verification.official_ref}"] if verification.official_ref else [])
            )
            packet.warnings = _dedupe_texts(packet.warnings + verification.warnings)
            return packet

        packet.status = verification.status
        packet.grounded = False
        packet.answer = "결론 보류: 로컬 exact 결과를 공식 조회와 교차검증하지 못했습니다."
        evidence = []
        if verification.mismatch_reason:
            evidence.append(f"검증 상태: {verification.mismatch_reason}")
        if verification.official_text:
            evidence.append(f"공식 조회 본문: {_clean_snippet(verification.official_text, limit=800)}")
        packet.evidence = evidence
        packet.citations = _dedupe_texts(
            packet.citations
            + ([f"[공식검증] {verification.official_ref}"] if verification.official_ref else [])
        )
        packet.warnings = _dedupe_texts(packet.warnings + verification.warnings)
        return packet

    def _attach_agent_prompts(self, packet: AnswerPacket) -> None:
        packet.agent_prompts = {
            "codex": build_agent_prompt("codex", packet, self.root),
            "claude-code": build_agent_prompt("claude-code", packet, self.root),
            "claude-stepfree": build_agent_prompt("claude-code", packet, self.root),
            "qwen": build_agent_prompt("qwen", packet, self.root),
            "gemini": build_agent_prompt("gemini", packet, self.root),
        }

    def _reanchor_statement_packet(
        self,
        query: str,
        packet: AnswerPacket,
        limit: int,
    ) -> AnswerPacket:
        if packet.grounded:
            return packet
        anchor_query = _extract_anchor_query(query)
        if not anchor_query or anchor_query == query:
            return packet
        anchor_decision, anchor_results = self.searcher.search(anchor_query, limit=limit)
        if not anchor_results:
            return packet
        anchor_packet = self._answer_from_results(anchor_query, anchor_decision.strategy, anchor_results)
        if not anchor_packet.evidence and not anchor_packet.citations:
            return packet
        packet.status = "needs_review"
        packet.grounded = False
        packet.answer = "결론 보류: 관련 exact 근거는 확보했지만, 진술의 진위 자체는 자동 확정하지 않았습니다."
        packet.evidence = anchor_packet.evidence
        packet.citations = anchor_packet.citations
        packet.source_type = anchor_packet.source_type
        packet.doc_id = anchor_packet.doc_id
        packet.warnings = _dedupe_texts(
            packet.warnings
            + [
                f"주의: 진술형 질의라 exact anchor '{anchor_query}'만 확보했고, 진위 판단은 보수적으로 보류했습니다."
            ]
            + anchor_packet.warnings
        )
        return packet

    def _run_agent_answer(
        self,
        packet: AnswerPacket,
        agent_name: str,
        model: str | None = None,
        timeout: int = 180,
    ) -> None:
        if not packet.agent_prompts:
            self._attach_agent_prompts(packet)
        prompt = packet.agent_prompts.get(agent_name)
        if not prompt:
            packet.agent_runs = packet.agent_runs or {}
            packet.agent_runs[agent_name] = {
                "ok": False,
                "error": f"agent_prompt_not_found={agent_name}",
            }
            return
        result = run_agent(agent_name, prompt, self.root, model=model, timeout=timeout)
        packet.agent_runs = packet.agent_runs or {}
        packet.agent_runs[agent_name] = result
        if result.get("ok") and result.get("output"):
            packet.llm_provider = agent_name
            packet.llm_model = model
            packet.llm_answer = result["output"]
            packet.llm_error = None

    def _extract_json_object(self, raw: str) -> dict | None:
        text = (raw or "").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _validate_judge_result(
        self, packet: AnswerPacket, result: dict
    ) -> tuple[bool, str | None]:
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return False, "judge_answer_missing"
        if result.get("grounded_only") is not True:
            return False, "judge_grounded_only_required"
        if result.get("uses_only_citations") is not True:
            return False, "judge_citation_scope_required"
        citations = result.get("citations")
        if citations is not None:
            if not isinstance(citations, list) or not all(
                isinstance(item, str) for item in citations
            ):
                return False, "judge_citations_invalid"
            allowed = set(packet.citations)
            if any(item not in allowed for item in citations):
                return False, "judge_citations_out_of_scope"
        return True, None

    def _run_judge_agent(
        self,
        packet: AnswerPacket,
        role: str,
        agent_name: str,
        model: str | None,
        timeout: int,
        prior_result: dict | None = None,
    ) -> dict | None:
        prompt = build_judge_prompt(
            role, agent_name, packet, self.root, prior_result=prior_result
        )
        result = run_agent(agent_name, prompt, self.root, model=model, timeout=timeout)
        packet.judge_runs = packet.judge_runs or {}
        packet.judge_runs[f"{role}:{agent_name}"] = result
        packet.judge_trace = packet.judge_trace or []
        packet.judge_trace.append(
            f"{role}:{agent_name}:{'ok' if result.get('ok') else result.get('error')}"
        )
        if not result.get("ok") or not result.get("output"):
            packet.judge_error = str(result.get("error") or f"{role}_failed")
            return None
        parsed = self._extract_json_object(result["output"])
        if not parsed:
            packet.judge_error = f"{role}_invalid_json"
            return None
        valid, reason = self._validate_judge_result(packet, parsed)
        if not valid:
            packet.judge_error = reason
            return None
        return parsed

    def _run_judge_layer(
        self,
        packet: AnswerPacket,
        judge_mode: str,
        judge_agent: str,
        judge_model: str | None,
        judge_critic_agent: str | None,
        judge_critic_model: str | None,
        timeout: int,
    ) -> None:
        packet.judge_mode = judge_mode
        packet.judge_trace = []
        solver = self._run_judge_agent(
            packet, "solver", judge_agent, judge_model, timeout
        )
        if not solver:
            return
        final = solver
        if judge_mode == "debate" and judge_critic_agent:
            critic = self._run_judge_agent(
                packet,
                "critic",
                judge_critic_agent,
                judge_critic_model,
                timeout,
                prior_result=solver,
            )
            if critic:
                final = critic
        packet.judge_result = final
        packet.answer = final["answer"].strip()
        if final.get("warnings"):
            packet.warnings = list(
                dict.fromkeys(
                    packet.warnings
                    + [str(item) for item in final["warnings"] if isinstance(item, str)]
                )
            )
        if final.get("citations"):
            packet.citations = [
                item for item in final["citations"] if item in set(packet.citations)
            ]
        if isinstance(final.get("status"), str) and final["status"] in {
            "grounded",
            "candidate_answer",
            "needs_review",
            "not_found",
        }:
            packet.status = final["status"]
            packet.grounded = final["status"] in {"grounded", "candidate_answer"}
        if final.get("recommended_options") and packet.exam_assessment is not None:
            packet.exam_assessment["judge_recommended_options"] = list(
                final["recommended_options"]
            )
        packet.judge_error = None

    def _attach_llm_answer(
        self, packet: AnswerPacket, provider_name: str, model: str | None = None
    ) -> None:
        resolved = resolve_provider_options(self.root, provider_name, model)
        if not resolved["provider"]:
            packet.llm_error = (
                f"provider 미설정입니다. 설정 파일: {resolved['config_path']}"
            )
            return
        provider_options = dict(resolved["options"])
        provider_options.pop("model", None)
        provider = create_provider(
            resolved["provider"], model=resolved["model"], **provider_options
        )
        result = provider.generate(
            build_llm_user_prompt(packet), system_prompt=SYSTEM_PROMPT
        )
        packet.llm_provider = result.provider
        packet.llm_model = result.model
        packet.llm_answer = result.output_text
        packet.llm_error = result.error

    def _run_fallback_chain(
        self,
        packet: AnswerPacket,
        primary_provider: str | None,
        primary_model: str | None,
        fallback_chain: list[str],
        agent_timeout: int,
    ) -> None:
        packet.fallback_trace = []
        tried: set[str] = set()

        def try_provider(name: str, model: str | None) -> bool:
            self._attach_llm_answer(packet, name, model=model)
            packet.fallback_trace.append(
                f"provider:{name}:{'ok' if packet.llm_answer else packet.llm_error}"
            )
            tried.add(name)
            return bool(packet.llm_answer)

        def try_agent(name: str) -> bool:
            self._run_agent_answer(packet, name, model=None, timeout=agent_timeout)
            result = (packet.agent_runs or {}).get(name, {})
            packet.fallback_trace.append(
                f"agent:{name}:{'ok' if result.get('ok') else result.get('error')}"
            )
            tried.add(name)
            return bool(result.get("ok"))

        if primary_provider:
            if try_provider(primary_provider, primary_model):
                return

        for name in fallback_chain:
            if name in tried:
                continue
            if name in {"codex", "claude-code", "claude-stepfree", "qwen", "gemini"}:
                if try_agent(name):
                    return
            else:
                if try_provider(name, None):
                    return

    def _apply_response_mode(
        self, packet: AnswerPacket, response_mode: str | None
    ) -> None:
        packet.response_mode = response_mode or "grounded"
        packet.final_answer_source = "grounded"
        if packet.judge_result and packet.judge_result.get("answer"):
            packet.answer = str(packet.judge_result["answer"])
            packet.final_answer_source = "judge"
            packet.warnings = packet.warnings + [
                "주의: 최종 answer는 LAW OPEN DATA 근거만 사용하도록 judge layer에서 재검토되었습니다."
            ]
        elif packet.response_mode == "llm_preferred" and packet.llm_answer:
            packet.answer = packet.llm_answer
            packet.final_answer_source = "llm"
            packet.warnings = packet.warnings + [
                "주의: 최종 answer는 LLM 정리문이며, grounded 근거는 evidence/citations를 확인해야 합니다."
            ]
        elif packet.response_mode == "llm_only":
            if packet.llm_answer:
                packet.answer = packet.llm_answer
                packet.evidence = []
                packet.final_answer_source = "llm"
            elif packet.llm_error:
                packet.answer = f"LLM 응답 실패: {packet.llm_error}"
                packet.final_answer_source = "llm_error"

    def _answer_from_results(
        self, query: str, route: str, results: list[UnifiedResult]
    ) -> AnswerPacket:
        if not results:
            if route.startswith("exact_"):
                return AnswerPacket(
                    query=query,
                    scope="all-laws-exam",
                    question_type=_detect_question_type(query),
                    route=route,
                    status="needs_review",
                    grounded=False,
                    answer="결론 보류: exact lookup이 실패했고 broad fallback을 비활성화했습니다.",
                    evidence=[],
                    citations=[],
                    warnings=["주의: exact 질의에서는 관련 없는 FTS/벡터 후보로 단정 답변하지 않습니다."],
                    source_type=None,
                    doc_id=None,
                )
            return AnswerPacket(
                query=query,
                scope="all-laws-exam",
                question_type=_detect_question_type(query),
                route=route,
                status="not_found",
                grounded=False,
                answer="검색 결과가 없습니다. exact lookup 또는 제목/사건번호를 다시 확인해야 합니다.",
                evidence=[],
                citations=[],
                warnings=["주의: LAW OPEN DATA 내부에서 직접 근거를 찾지 못했습니다."],
                source_type=None,
                doc_id=None,
            )

        top = results[0]
        if not top.search_path.startswith("exact"):
            return _fallback_answer(query, route, results)

        payload = top.payload
        if top.source_type == "law":
            packet = _law_answer(query, route, payload)
            return self._apply_exact_verification(packet, route, top)
        if top.source_type == "acr":
            packet = _acr_answer(query, route, payload)
            return self._apply_exact_verification(packet, route, top)
        if top.source_type == "prec":
            packet = _prec_answer(query, route, payload)
            return self._apply_exact_verification(packet, route, top)
        if top.source_type == "decc":
            packet = _decc_answer(query, route, payload)
            return self._apply_exact_verification(packet, route, top)
        if top.source_type == "detc":
            packet = _detc_answer(query, route, payload)
            return self._apply_exact_verification(packet, route, top)
        if top.source_type == "expc":
            packet = _expc_answer(query, route, payload)
            return self._apply_exact_verification(packet, route, top)
        return _fallback_answer(query, route, results)

    def _review_options(
        self, query: str, options: list[tuple[str, str]], limit: int
    ) -> list[dict]:
        reviews: list[dict] = []
        for label, option_text in options:
            decision, results = self.searcher.search(option_text, limit=limit)
            packet = self._answer_from_results(option_text, decision.strategy, results)
            anchor_query = None
            if not packet.grounded:
                anchor_query = _extract_anchor_query(option_text)
                if anchor_query:
                    anchor_decision, anchor_results = self.searcher.search(
                        anchor_query, limit=limit
                    )
                    anchor_packet = self._answer_from_results(
                        anchor_query, anchor_decision.strategy, anchor_results
                    )
                    if anchor_packet.grounded:
                        packet = anchor_packet
            support_level, support_reason = _classify_option_support(
                option_text, packet
            )
            reviews.append(
                {
                    "label": label,
                    "text": option_text,
                    "anchor_query": anchor_query,
                    "status": packet.status,
                    "grounded": packet.grounded,
                    "support_level": support_level,
                    "support_reason": support_reason,
                    "source_type": packet.source_type,
                    "doc_id": packet.doc_id,
                    "answer": packet.answer,
                    "citations": packet.citations,
                }
            )
        return reviews

    def _review_single_statement(self, statement: str, limit: int) -> dict:
        decision, results = self.searcher.search(statement, limit=limit)
        packet = self._answer_from_results(statement, decision.strategy, results)
        anchor_query = None
        if not packet.grounded:
            anchor_query = _extract_anchor_query(statement)
            if anchor_query:
                anchor_decision, anchor_results = self.searcher.search(
                    anchor_query, limit=limit
                )
                anchor_packet = self._answer_from_results(
                    anchor_query, anchor_decision.strategy, anchor_results
                )
                if anchor_packet.grounded:
                    packet = anchor_packet
        support_level, support_reason = _classify_option_support(statement, packet)
        return {
            "label": "OX",
            "text": statement,
            "anchor_query": anchor_query,
            "status": packet.status,
            "grounded": packet.grounded,
            "support_level": support_level,
            "support_reason": support_reason,
            "source_type": packet.source_type,
            "doc_id": packet.doc_id,
            "answer": packet.answer,
            "citations": packet.citations,
        }

    def _review_substatements(self, option_text: str, limit: int) -> list[dict]:
        items = _extract_substatements(option_text)
        reviews: list[dict] = []
        for label, item_text in items:
            review = self._review_single_statement(item_text, limit=limit)
            review["label"] = label
            review["text"] = item_text
            reviews.append(review)
        return reviews

    def _assess_substatement_combo(self, sub_reviews: list[dict]) -> dict | None:
        if not sub_reviews:
            return None
        supported = [
            review["label"]
            for review in sub_reviews
            if review["support_level"] == "supported"
        ]
        unsupported = [
            review["label"]
            for review in sub_reviews
            if review["support_level"] == "unsupported"
        ]
        indeterminate = [
            review["label"]
            for review in sub_reviews
            if review["support_level"] == "indeterminate"
        ]
        labels = [review["label"] for review in sub_reviews]
        return {
            "all_labels": labels,
            "supported_labels": supported,
            "unsupported_labels": unsupported,
            "indeterminate_labels": indeterminate,
            "supported_combo": ", ".join(supported),
        }

    def _derive_combo_option_support(
        self, sub_assessment: dict | None
    ) -> tuple[str, str] | None:
        if not sub_assessment:
            return None

        total = len(sub_assessment["all_labels"])
        supported = len(sub_assessment["supported_labels"])
        unsupported = len(sub_assessment["unsupported_labels"])
        indeterminate = len(sub_assessment["indeterminate_labels"])

        if total and supported == total:
            return ("supported", "보기의 하위 항목이 모두 supported입니다.")
        if unsupported:
            return (
                "unsupported",
                "보기의 하위 항목 중 unsupported가 있어 전체 보기를 정답 후보로 보기 어렵습니다.",
            )
        if indeterminate:
            return (
                "indeterminate",
                "보기의 하위 항목 중 indeterminate가 있어 전체 보기 확정이 어렵습니다.",
            )
        return None

    def _review_selection_reference_options(
        self,
        stem: str,
        options: list[tuple[str, str]],
        limit: int,
        target_key: str,
        target_label: str,
    ) -> list[dict] | None:
        stem_sub_reviews = self._review_substatements(stem, limit=limit)
        if not stem_sub_reviews:
            return None

        review_map = {review["label"]: review for review in stem_sub_reviews}
        if not all(_is_combo_reference_text(option_text) for _, option_text in options):
            return None

        reviews: list[dict] = []
        for label, option_text in options:
            if _is_all_correct_selection_text(option_text):
                referenced = [review["label"] for review in stem_sub_reviews]
            else:
                referenced = _extract_referenced_substatement_labels(option_text)
            sub_reviews = [
                review_map[sub_label]
                for sub_label in referenced
                if sub_label in review_map
            ]
            missing = [
                sub_label for sub_label in referenced if sub_label not in review_map
            ]
            sub_assessment = self._assess_substatement_combo(sub_reviews)
            if sub_assessment:
                selected_labels = set(
                    sub_assessment["supported_labels"]
                    if target_key == "supported"
                    else sub_assessment["unsupported_labels"]
                )
                total_labels = set(sub_assessment["all_labels"])
                indeterminate_labels = set(sub_assessment["indeterminate_labels"])
                referenced_set = set(referenced)
                if referenced_set == selected_labels and not indeterminate_labels:
                    derived = (
                        "supported",
                        f"보기의 하위 항목이 모두 {target_label}입니다.",
                    )
                elif referenced_set & indeterminate_labels:
                    derived = (
                        "indeterminate",
                        f"보기의 하위 항목 중 indeterminate가 있어 전체 보기 확정이 어렵습니다.",
                    )
                else:
                    derived = (
                        "unsupported",
                        f"보기의 하위 항목이 {target_label}와 일치하지 않습니다.",
                    )
            else:
                derived = None
            effective_level = derived[0] if derived else "indeterminate"
            effective_reason = (
                derived[1]
                if derived
                else "조합 보기를 평가할 하위 항목 근거가 부족합니다."
            )
            if missing:
                effective_level = "indeterminate"
                effective_reason = f"문항 본문에 없는 하위 항목이 포함되어 있습니다: {', '.join(missing)}"
            citations = []
            for sub_review in sub_reviews:
                citations.extend(sub_review["citations"])
            citations = list(dict.fromkeys(citations))
            reviews.append(
                {
                    "label": label,
                    "text": option_text,
                    "anchor_query": None,
                    "status": "grounded"
                    if sub_reviews and not missing
                    else "needs_review",
                    "grounded": bool(sub_reviews) and not missing,
                    "support_level": effective_level,
                    "support_reason": effective_reason,
                    "effective_support_level": effective_level,
                    "effective_support_reason": effective_reason,
                    "source_type": "mixed"
                    if len(
                        {
                            sub["source_type"]
                            for sub in sub_reviews
                            if sub["source_type"]
                        }
                    )
                    > 1
                    else (sub_reviews[0]["source_type"] if sub_reviews else None),
                    "doc_id": None,
                    "answer": f"조합 보기 {label}는 문항 본문의 하위 진술 근거를 조합해 평가했습니다.",
                    "citations": citations,
                    "sub_reviews": sub_reviews,
                    "sub_assessment": sub_assessment,
                    "referenced_labels": referenced,
                }
            )
        return reviews

    def _review_combo_reference_options(
        self,
        stem: str,
        options: list[tuple[str, str]],
        limit: int,
    ) -> list[dict] | None:
        return self._review_selection_reference_options(
            stem, options, limit, "supported", "supported"
        )

    def _review_incorrect_reference_options(
        self,
        stem: str,
        options: list[tuple[str, str]],
        limit: int,
    ) -> list[dict] | None:
        return self._review_selection_reference_options(
            stem, options, limit, "unsupported", "unsupported"
        )

    def _review_count_reference_options(
        self,
        stem: str,
        options: list[tuple[str, str]],
        limit: int,
    ) -> list[dict] | None:
        stem_sub_reviews = self._review_substatements(stem, limit=limit)
        if not stem_sub_reviews or not all(
            _is_count_only_text(option_text) for _, option_text in options
        ):
            return None

        sub_assessment = self._assess_substatement_combo(stem_sub_reviews)
        supported_count = (
            len(sub_assessment["supported_labels"]) if sub_assessment else 0
        )
        indeterminate_count = (
            len(sub_assessment["indeterminate_labels"]) if sub_assessment else 0
        )
        citations = []
        for sub_review in stem_sub_reviews:
            citations.extend(sub_review["citations"])
        citations = list(dict.fromkeys(citations))

        reviews: list[dict] = []
        for label, option_text in options:
            count_value = _extract_count_value(option_text)
            if count_value is None:
                continue
            if indeterminate_count:
                effective_level = "indeterminate"
                effective_reason = (
                    "하위 진술 중 indeterminate가 있어 정확한 개수 확정이 어렵습니다."
                )
            elif count_value == supported_count:
                effective_level = "supported"
                effective_reason = (
                    f"supported 하위 진술 수가 {supported_count}개입니다."
                )
            else:
                effective_level = "unsupported"
                effective_reason = (
                    f"supported 하위 진술 수는 {supported_count}개입니다."
                )
            reviews.append(
                {
                    "label": label,
                    "text": option_text,
                    "anchor_query": None,
                    "status": "grounded" if not indeterminate_count else "needs_review",
                    "grounded": indeterminate_count == 0,
                    "support_level": effective_level,
                    "support_reason": effective_reason,
                    "effective_support_level": effective_level,
                    "effective_support_reason": effective_reason,
                    "source_type": "mixed"
                    if len(
                        {
                            sub["source_type"]
                            for sub in stem_sub_reviews
                            if sub["source_type"]
                        }
                    )
                    > 1
                    else (
                        stem_sub_reviews[0]["source_type"] if stem_sub_reviews else None
                    ),
                    "doc_id": None,
                    "answer": f"개수 보기 {label}는 supported 하위 진술 개수를 기준으로 평가했습니다.",
                    "citations": citations,
                    "sub_reviews": stem_sub_reviews,
                    "sub_assessment": sub_assessment,
                    "count_value": count_value,
                }
            )
        return reviews or None

    def answer(
        self,
        query: str,
        limit: int = 5,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        include_agent_prompts: bool | None = None,
        response_mode: str | None = None,
        run_agent_name: str | None = None,
        run_agent_model: str | None = None,
        judge_mode: str | None = None,
        judge_agent: str | None = None,
        judge_model: str | None = None,
        judge_critic_agent: str | None = None,
        judge_critic_model: str | None = None,
        explain_style: str | None = None,
    ) -> AnswerPacket:
        resolved = resolve_provider_options(self.root, llm_provider, llm_model)
        explicit_llm_request = bool(llm_provider or llm_model)
        decision, results = self.searcher.search(query, limit=limit)
        packet = self._answer_from_results(query, decision.strategy, results)
        packet = self._reanchor_statement_packet(query, packet, limit=limit)
        if packet.question_type == "exam_choice":
            options = _extract_choice_options(query)
            if options:
                stem = _extract_question_stem(query)
                combo_reference_reviews = None
                if _is_all_incorrect_selection_query(query):
                    combo_reference_reviews = self._review_incorrect_reference_options(
                        stem, options, limit=limit
                    )
                elif any(
                    _is_combo_reference_text(option_text) for _, option_text in options
                ):
                    combo_reference_reviews = self._review_combo_reference_options(
                        stem, options, limit=limit
                    )
                count_reference_reviews = (
                    None
                    if combo_reference_reviews
                    else self._review_count_reference_options(
                        stem, options, limit=limit
                    )
                )
                packet.option_reviews = (
                    combo_reference_reviews
                    or count_reference_reviews
                    or self._review_options(query, options, limit=limit)
                )
                if not combo_reference_reviews and not count_reference_reviews:
                    for review in packet.option_reviews:
                        sub_reviews = self._review_substatements(
                            review["text"], limit=limit
                        )
                        if sub_reviews:
                            review["sub_reviews"] = sub_reviews
                            review["sub_assessment"] = self._assess_substatement_combo(
                                sub_reviews
                            )
                            derived = self._derive_combo_option_support(
                                review["sub_assessment"]
                            )
                            if derived:
                                review["effective_support_level"] = derived[0]
                                review["effective_support_reason"] = derived[1]
                            else:
                                review["effective_support_level"] = review[
                                    "support_level"
                                ]
                                review["effective_support_reason"] = review[
                                    "support_reason"
                                ]
                        else:
                            review["effective_support_level"] = review["support_level"]
                            review["effective_support_reason"] = review[
                                "support_reason"
                            ]
                option_citations: list[str] = []
                grounded_sources = []
                grounded_doc_ids = []
                for review in packet.option_reviews:
                    option_citations.extend(review["citations"])
                    if review["grounded"] and review["source_type"]:
                        grounded_sources.append(review["source_type"])
                    if review["grounded"] and review["doc_id"]:
                        grounded_doc_ids.append(review["doc_id"])
                if option_citations:
                    packet.citations = list(dict.fromkeys(option_citations))
                if grounded_sources:
                    packet.source_type = (
                        grounded_sources[0]
                        if len(set(grounded_sources)) == 1
                        else "mixed"
                    )
                if grounded_doc_ids:
                    packet.doc_id = (
                        grounded_doc_ids[0] if len(set(grounded_doc_ids)) == 1 else None
                    )
                grounded_labels = [
                    review["label"]
                    for review in packet.option_reviews
                    if review["grounded"]
                ]
                packet.exam_assessment = _assess_choice_query(
                    query, packet.option_reviews
                )
                packet.warnings = [
                    "주의: 보기별 평가는 LAW OPEN DATA 원문과 하위 진술 근거를 기준으로 계산했습니다."
                ]
                if stem:
                    packet.evidence = [f"문항 본문: {stem}"]
                else:
                    packet.evidence = []
                if grounded_labels:
                    packet.evidence = packet.evidence + [
                        f"보기별 grounded 후보: {', '.join(grounded_labels)}"
                    ]
                else:
                    packet.evidence = packet.evidence + [
                        "보기별 grounded 후보를 확정하지 못했습니다."
                    ]
                effective_supported = [
                    review["label"]
                    for review in packet.option_reviews
                    if review.get("effective_support_level") == "supported"
                ]
                if effective_supported:
                    packet.evidence = packet.evidence + [
                        f"보기별 supported 후보: {', '.join(effective_supported)}"
                    ]
                if packet.exam_assessment.get("recommended_options"):
                    packet.evidence = packet.evidence + [
                        f"문항 기준 최종 정답 후보: {', '.join(packet.exam_assessment['recommended_options'])}"
                    ]
                packet.evidence = packet.evidence + [
                    f"문항 평가: {packet.exam_assessment['reason']}"
                ]
                if packet.exam_assessment["status"] == "candidate_answer":
                    packet.status = "candidate_answer"
                    packet.answer = (
                        "문항 판단: "
                        f"{', '.join(packet.exam_assessment['recommended_options'])}가 현재 근거상 정답 후보입니다."
                    )
                    packet.grounded = True
                elif packet.exam_assessment["status"] == "multiple_candidates":
                    packet.status = "needs_review"
                    packet.answer = (
                        "문항 판단 보류: "
                        f"{', '.join(packet.exam_assessment['recommended_options'])}가 후보지만 하나로 확정되지 않습니다."
                    )
                    packet.grounded = False
                else:
                    packet.status = "needs_review"
                    packet.answer = "문항 판단 보류: 보기별 근거는 일부 있으나 정답을 자동 확정하기 어렵습니다."
                    packet.grounded = False
        elif packet.question_type == "exam_ox":
            statement = OX_SUFFIX_RE.sub("", query).strip()
            review = self._review_single_statement(statement, limit=limit)
            packet.option_reviews = [review]
            if review["support_level"] == "supported":
                packet.exam_assessment = {
                    "status": "candidate_answer",
                    "recommended_options": ["O"],
                    "reason": "진술이 근거와 직접 일치해 O 후보로 분류했습니다.",
                }
            elif review["support_level"] == "unsupported":
                packet.exam_assessment = {
                    "status": "candidate_answer",
                    "recommended_options": ["X"],
                    "reason": "진술이 근거와 충돌해 X 후보로 분류했습니다.",
                }
            else:
                packet.exam_assessment = {
                    "status": "insufficient_grounding",
                    "recommended_options": [],
                    "reason": "진술의 O/X를 자동 확정하기엔 근거가 부족합니다.",
                }
            packet.evidence = packet.evidence + [
                f"OX 평가: {packet.exam_assessment['reason']}"
            ]
            packet.citations = review["citations"]
            packet.source_type = review["source_type"]
            packet.doc_id = review["doc_id"]
            packet.warnings = [
                "주의: OX 평가는 단일 진술문에 대한 근거 대조 결과입니다."
            ]
            if packet.exam_assessment["status"] == "candidate_answer":
                packet.status = "candidate_answer"
                packet.grounded = True
                packet.answer = (
                    "문항 판단: "
                    f"{', '.join(packet.exam_assessment['recommended_options'])}가 현재 근거상 정답 후보입니다."
                )
            else:
                packet.status = "needs_review"
                packet.grounded = False
                packet.answer = "문항 판단 보류: 단일 진술문의 O/X를 자동 확정하기엔 근거가 부족합니다."
        include_agent_prompts = (
            resolved["include_agent_prompts"]
            if include_agent_prompts is None
            else include_agent_prompts
        )
        final_response_mode = response_mode or resolved["response_mode"]
        final_judge_mode = judge_mode or resolved["judge_mode"]
        final_judge_agent = judge_agent
        final_judge_critic_agent = judge_critic_agent
        if (
            not final_judge_agent
            and final_judge_mode != "off"
            and resolved["judge_chain"]
        ):
            final_judge_agent = resolved["judge_chain"][0]
            if len(resolved["judge_chain"]) > 1:
                final_judge_critic_agent = (
                    final_judge_critic_agent or resolved["judge_chain"][1]
                )
        if include_agent_prompts:
            self._attach_agent_prompts(packet)
        if final_judge_agent:
            self._run_judge_layer(
                packet,
                judge_mode="debate"
                if final_judge_critic_agent
                else (final_judge_mode if final_judge_mode != "off" else "single"),
                judge_agent=final_judge_agent,
                judge_model=judge_model,
                judge_critic_agent=final_judge_critic_agent,
                judge_critic_model=judge_critic_model,
                timeout=resolved["judge_timeout_seconds"],
            )
        if run_agent_name:
            self._run_agent_answer(
                packet,
                run_agent_name,
                model=run_agent_model,
                timeout=resolved["agent_timeout_seconds"],
            )
        elif final_response_mode != "grounded" or explicit_llm_request:
            self._run_fallback_chain(
                packet,
                resolved["provider"],
                resolved["model"],
                resolved["fallback_chain"],
                resolved["agent_timeout_seconds"],
            )
        self._apply_response_mode(packet, final_response_mode)
        packet.explain_style = explain_style
        if explain_style == "admin_exam":
            packet.teaching_explanation = _build_admin_exam_explanation(packet)
        return packet
