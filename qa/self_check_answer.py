from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa.answering import GroundedAnswerer


EXPECTED = {
    "행정심판법 제18조": ("law", "law:001363", "[법령]", "grounded", True),
    "결정문일련번호 23": ("acr", "acr:23", "[결정문]", "grounded", True),
    "84누180": ("prec", "prec:100006", "[판례]", "grounded", True),
    "2000-04033": ("decc", "decc:17109", "[행정심판례]", "grounded", True),
    "2004헌마275": ("detc", "detc:10026", "[헌재결정례]", "grounded", True),
    "05-0096": ("expc", "expc:313107", "[법령해석례]", "grounded", True),
}


def _answer(answerer: GroundedAnswerer, query: str, limit: int = 3):
    return answerer.answer(query, limit=limit, response_mode="grounded")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run grounded-answer self-check.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    answerer = GroundedAnswerer(args.root)
    failures = 0
    for query, expected in EXPECTED.items():
        packet = _answer(answerer, query, limit=3)
        actual = (packet.source_type, packet.doc_id)
        citation_ok = bool(packet.citations) and expected[2] in packet.citations[0]
        status_ok = packet.status == expected[3]
        grounded_ok = packet.grounded is expected[4]
        evidence_ok = bool(packet.evidence)
        scope_ok = packet.scope == "all-laws-exam"
        status = "OK" if actual == expected[:2] and citation_ok and status_ok and grounded_ok and evidence_ok and scope_ok and packet.answer else "FAIL"
        if status == "FAIL":
            failures += 1
        print(
            f"{status} query={query} expected_source={expected[:2]} actual_source={actual} "
            f"citation_ok={citation_ok} status_ok={status_ok} grounded_ok={grounded_ok} scope_ok={scope_ok}"
        )

    review_packet = _answer(answerer, "기소유예처분취소", limit=3)
    review_ok = review_packet.status == "needs_review" and review_packet.grounded is False
    print(f"{'OK' if review_ok else 'FAIL'} query=기소유예처분취소 expected_review=True actual_status={review_packet.status}")
    if not review_ok:
        failures += 1

    choice_query = "다음 중 옳은 것은? ① 행정심판법 제18조 ② 2004헌마275"
    choice_packet = _answer(answerer, choice_query, limit=3)
    choice_ok = (
        choice_packet.question_type == "exam_choice"
        and bool(choice_packet.option_reviews)
        and len(choice_packet.option_reviews) == 2
        and choice_packet.exam_assessment is not None
        and choice_packet.exam_assessment["status"] == "multiple_candidates"
        and all(review["support_level"] == "supported" for review in choice_packet.option_reviews)
    )
    print(
        f"{'OK' if choice_ok else 'FAIL'} query=choice "
        f"options_detected={len(choice_packet.option_reviews or [])} "
        f"assessment={choice_packet.exam_assessment['status'] if choice_packet.exam_assessment else None}"
    )
    if not choice_ok:
        failures += 1

    ox_query = "행정심판법 제18조는 대리인의 선임에 관한 조항이다. OX"
    ox_packet = _answer(answerer, ox_query, limit=3)
    ox_ok = (
        ox_packet.question_type == "exam_ox"
        and ox_packet.exam_assessment is not None
        and ox_packet.exam_assessment["status"] == "candidate_answer"
        and ox_packet.exam_assessment["recommended_options"] == ["O"]
    )
    print(
        f"{'OK' if ox_ok else 'FAIL'} query=ox "
        f"assessment={ox_packet.exam_assessment['status'] if ox_packet.exam_assessment else None} "
        f"recommended={ox_packet.exam_assessment['recommended_options'] if ox_packet.exam_assessment else None}"
    )
    if not ox_ok:
        failures += 1

    ox_negative_query = "행정심판법 제18조는 대리인의 선임에 관한 조항이 아니다. OX"
    ox_negative_packet = _answer(answerer, ox_negative_query, limit=3)
    ox_negative_ok = (
        ox_negative_packet.question_type == "exam_ox"
        and ox_negative_packet.exam_assessment is not None
        and ox_negative_packet.exam_assessment["status"] == "candidate_answer"
        and ox_negative_packet.exam_assessment["recommended_options"] == ["X"]
    )
    print(
        f"{'OK' if ox_negative_ok else 'FAIL'} query=ox_negative "
        f"assessment={ox_negative_packet.exam_assessment['status'] if ox_negative_packet.exam_assessment else None} "
        f"recommended={ox_negative_packet.exam_assessment['recommended_options'] if ox_negative_packet.exam_assessment else None}"
    )
    if not ox_negative_ok:
        failures += 1

    ox_antonym_query = "행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다. OX"
    ox_antonym_packet = _answer(answerer, ox_antonym_query, limit=3)
    ox_antonym_ok = (
        ox_antonym_packet.question_type == "exam_ox"
        and ox_antonym_packet.exam_assessment is not None
        and ox_antonym_packet.exam_assessment["status"] == "candidate_answer"
        and ox_antonym_packet.exam_assessment["recommended_options"] == ["X"]
    )
    print(
        f"{'OK' if ox_antonym_ok else 'FAIL'} query=ox_antonym "
        f"assessment={ox_antonym_packet.exam_assessment['status'] if ox_antonym_packet.exam_assessment else None} "
        f"recommended={ox_antonym_packet.exam_assessment['recommended_options'] if ox_antonym_packet.exam_assessment else None}"
    )
    if not ox_antonym_ok:
        failures += 1

    long_choice_query = """다음 설명 중 옳은 것은?
1) 행정심판법 제18조
2) 2004헌마275
3) 84누180
4) 05-0096"""
    long_choice_packet = _answer(answerer, long_choice_query, limit=3)
    long_choice_ok = (
        long_choice_packet.question_type == "exam_choice"
        and bool(long_choice_packet.option_reviews)
        and len(long_choice_packet.option_reviews) == 4
        and long_choice_packet.exam_assessment is not None
        and long_choice_packet.exam_assessment["status"] == "multiple_candidates"
    )
    print(
        f"{'OK' if long_choice_ok else 'FAIL'} query=long_choice "
        f"options_detected={len(long_choice_packet.option_reviews or [])} "
        f"assessment={long_choice_packet.exam_assessment['status'] if long_choice_packet.exam_assessment else None}"
    )
    if not long_choice_ok:
        failures += 1

    count_query = """다음 중 옳은 것의 개수는?
ㄱ. 행정심판법 제18조
ㄴ. 2004헌마275
ㄷ. 행정심판법 제18조는 대리인의 선임에 관한 조항이 아니다.
ㄹ. 84누180
1) 1개
2) 2개
3) 3개
4) 4개"""
    count_packet = _answer(answerer, count_query, limit=3)
    count_ok = (
        count_packet.question_type == "exam_choice"
        and count_packet.exam_assessment is not None
        and count_packet.exam_assessment["status"] == "candidate_answer"
        and count_packet.exam_assessment.get("count_answer") == 3
        and count_packet.exam_assessment["recommended_options"] == ["3)"]
    )
    print(
        f"{'OK' if count_ok else 'FAIL'} query=count "
        f"count_answer={count_packet.exam_assessment.get('count_answer') if count_packet.exam_assessment else None} "
        f"recommended={count_packet.exam_assessment['recommended_options'] if count_packet.exam_assessment else None}"
    )
    if not count_ok:
        failures += 1

    wrong_count_query = """다음 중 틀린 것의 개수는?
ㄱ. 행정심판법 제18조
ㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.
ㄷ. 2004헌마275
ㄹ. 84누180
1) 1개
2) 2개
3) 3개
4) 4개"""
    wrong_count_packet = _answer(answerer, wrong_count_query, limit=3)
    wrong_count_ok = (
        wrong_count_packet.question_type == "exam_choice"
        and wrong_count_packet.exam_assessment is not None
        and wrong_count_packet.exam_assessment["status"] == "candidate_answer"
        and wrong_count_packet.exam_assessment.get("count_answer") == 1
        and wrong_count_packet.exam_assessment["recommended_options"] == ["1)"]
    )
    print(
        f"{'OK' if wrong_count_ok else 'FAIL'} query=wrong_count "
        f"count_answer={wrong_count_packet.exam_assessment.get('count_answer') if wrong_count_packet.exam_assessment else None} "
        f"recommended={wrong_count_packet.exam_assessment['recommended_options'] if wrong_count_packet.exam_assessment else None}"
    )
    if not wrong_count_ok:
        failures += 1

    combo_query = """다음 중 옳은 것은?
1) ㄱ. 행정심판법 제18조
   ㄴ. 2004헌마275
2) ㄱ. 기소유예처분취소
   ㄴ. 84누180"""
    combo_packet = _answer(answerer, combo_query, limit=3)
    combo_ok = (
        combo_packet.question_type == "exam_choice"
        and bool(combo_packet.option_reviews)
        and combo_packet.option_reviews[0].get("sub_assessment") is not None
        and combo_packet.option_reviews[0]["sub_assessment"]["supported_labels"] == ["ㄱ.", "ㄴ."]
        and combo_packet.option_reviews[0].get("effective_support_level") == "supported"
        and combo_packet.exam_assessment is not None
        and combo_packet.exam_assessment["status"] == "candidate_answer"
        and combo_packet.exam_assessment["recommended_options"] == ["1)"]
    )
    print(
        f"{'OK' if combo_ok else 'FAIL'} query=combo "
        f"sub_supported={combo_packet.option_reviews[0]['sub_assessment']['supported_labels'] if combo_packet.option_reviews and combo_packet.option_reviews[0].get('sub_assessment') else None} "
        f"recommended={combo_packet.exam_assessment['recommended_options'] if combo_packet.exam_assessment else None}"
    )
    if not combo_ok:
        failures += 1

    combo_reference_query = """다음 중 옳은 것은?
ㄱ. 행정심판법 제18조
ㄴ. 2004헌마275
ㄷ. 기소유예처분취소
1) ㄱ, ㄴ
2) ㄱ, ㄷ
3) ㄴ, ㄷ
4) ㄱ, ㄴ, ㄷ"""
    combo_reference_packet = _answer(answerer, combo_reference_query, limit=3)
    combo_reference_ok = (
        combo_reference_packet.question_type == "exam_choice"
        and bool(combo_reference_packet.option_reviews)
        and combo_reference_packet.exam_assessment is not None
        and combo_reference_packet.exam_assessment["status"] == "candidate_answer"
        and combo_reference_packet.exam_assessment["recommended_options"] == ["1)"]
        and combo_reference_packet.option_reviews[0].get("referenced_labels") == ["ㄱ.", "ㄴ."]
    )
    print(
        f"{'OK' if combo_reference_ok else 'FAIL'} query=combo_reference "
        f"recommended={combo_reference_packet.exam_assessment['recommended_options'] if combo_reference_packet.exam_assessment else None}"
    )
    if not combo_reference_ok:
        failures += 1

    combo_variant_query = """다음 중 옳은 것은?
ㄱ. 행정심판법 제18조
ㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.
ㄷ. 2004헌마275
1) ㄱ, ㄷ만
2) ㄴ만
3) 모두 고른 것
4) ㄱ, ㄴ만"""
    combo_variant_packet = _answer(answerer, combo_variant_query, limit=3)
    combo_variant_ok = (
        combo_variant_packet.question_type == "exam_choice"
        and bool(combo_variant_packet.option_reviews)
        and combo_variant_packet.exam_assessment is not None
        and combo_variant_packet.exam_assessment["status"] == "candidate_answer"
        and combo_variant_packet.exam_assessment["recommended_options"] == ["1)"]
        and combo_variant_packet.option_reviews[0].get("referenced_labels") == ["ㄱ.", "ㄷ."]
        and combo_variant_packet.option_reviews[0].get("effective_support_level") == "supported"
    )
    print(
        f"{'OK' if combo_variant_ok else 'FAIL'} query=combo_variant "
        f"recommended={combo_variant_packet.exam_assessment['recommended_options'] if combo_variant_packet.exam_assessment else None}"
    )
    if not combo_variant_ok:
        failures += 1

    wrong_selection_query = """다음 중 틀린 것만 모두 고른 것은?
ㄱ. 행정심판법 제18조
ㄴ. 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.
ㄷ. 2004헌마275
1) ㄴ만
2) ㄱ, ㄷ만
3) 모두 고른 것
4) ㄱ, ㄴ만"""
    wrong_selection_packet = _answer(answerer, wrong_selection_query, limit=3)
    wrong_selection_ok = (
        wrong_selection_packet.question_type == "exam_choice"
        and bool(wrong_selection_packet.option_reviews)
        and wrong_selection_packet.exam_assessment is not None
        and wrong_selection_packet.exam_assessment["status"] == "candidate_answer"
        and wrong_selection_packet.exam_assessment["recommended_options"] == ["1)"]
        and wrong_selection_packet.option_reviews[0].get("referenced_labels") == ["ㄴ."]
        and wrong_selection_packet.option_reviews[0].get("effective_support_level") == "supported"
    )
    print(
        f"{'OK' if wrong_selection_ok else 'FAIL'} query=wrong_selection "
        f"recommended={wrong_selection_packet.exam_assessment['recommended_options'] if wrong_selection_packet.exam_assessment else None}"
    )
    if not wrong_selection_ok:
        failures += 1
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
