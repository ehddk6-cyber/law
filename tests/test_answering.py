from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qa.answering import (
    AnswerPacket,
    GroundedAnswerer,
    _build_admin_exam_explanation,
    _clean_snippet,
    _detect_question_type,
    _extract_choice_options,
    _extract_count_value,
    _extract_question_stem,
    _extract_referenced_substatement_labels,
    _extract_substatements,
    _is_all_correct_selection_query,
    _is_all_correct_selection_text,
    _is_all_incorrect_selection_query,
    _is_combo_reference_text,
    _is_count_only_text,
    _normalize_statement_descriptor,
    _overlap_ratio,
    _strip_anchor_text,
    _tokenize,
    _extract_anchor_query,
)
from qa.exact_verifier import ExactVerification, KoreanLawExactVerifier
from qa.unified import UnifiedResult
from qa.llm_prompting import build_llm_user_prompt, SYSTEM_PROMPT
from qa.response_schema import SCHEMA_VERSION, ANSWER_RESULT_SCHEMA


class TestDetectQuestionType(unittest.TestCase):
    def test_exam_choice(self):
        self.assertEqual(_detect_question_type("다음 중 옳은 것은"), "exam_choice")
        self.assertEqual(_detect_question_type("틀린 것을 고르시오"), "exam_choice")

    def test_exam_ox(self):
        self.assertEqual(_detect_question_type("다음은 OX로 답하시오"), "exam_ox")

    def test_statute_lookup(self):
        self.assertEqual(_detect_question_type("행정심판법 제18조의 내용은"), "statute_lookup")

    def test_authority_lookup(self):
        self.assertEqual(_detect_question_type("판례에서 판단한 바에 따르면"), "authority_lookup")
        self.assertEqual(_detect_question_type("결정례의 해석례에 의하면"), "authority_lookup")

    def test_case_lookup(self):
        self.assertEqual(_detect_question_type("84누180"), "case_lookup")

    def test_exam_general(self):
        self.assertEqual(_detect_question_type("행정소송이란 무엇인가"), "exam_general")


class TestExtractChoiceOptions(unittest.TestCase):
    def test_four_options(self):
        query = "다음 중 옳은 것은?\n① 첫 번째\n② 두 번째\n③ 세 번째\n④ 네 번째"
        options = _extract_choice_options(query)
        self.assertEqual(len(options), 4)
        self.assertEqual(options[0][0], "①")
        self.assertIn("첫 번째", options[0][1])

    def test_numbered_options(self):
        query = "고르시오\n1. AAA\n2. BBB\n3. CCC"
        options = _extract_choice_options(query)
        self.assertEqual(len(options), 3)

    def test_single_option(self):
        options = _extract_choice_options("① 하나만")
        self.assertEqual(options, [])

    def test_no_options(self):
        options = _extract_choice_options("질문만 있는 문장")
        self.assertEqual(options, [])


class TestExtractQuestionStem(unittest.TestCase):
    def test_with_options(self):
        query = "다음 중 옳은 것은?\n① 보기1\n② 보기2"
        stem = _extract_question_stem(query)
        self.assertEqual(stem, "다음 중 옳은 것은?")

    def test_no_options(self):
        query = "질문만 있습니다"
        stem = _extract_question_stem(query)
        self.assertEqual(stem, "질문만 있습니다")


class TestSubstatements(unittest.TestCase):
    def test_extract(self):
        text = "ㄱ. 첫째\nㄴ. 둘째\nㄷ. 셋째"
        items = _extract_substatements(text)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0][0], "ㄱ.")

    def test_no_substatements(self):
        items = _extract_substatements("일반 텍스트")
        self.assertEqual(items, [])

    def test_referenced_labels(self):
        labels = _extract_referenced_substatement_labels("ㄱ.과 ㄷ.만 고르시오")
        self.assertIn("ㄱ.", labels)
        self.assertIn("ㄷ.", labels)


class TestSelectionQueries(unittest.TestCase):
    def test_all_correct(self):
        self.assertTrue(_is_all_correct_selection_query("옳은 것을 모두 고르시오"))
        self.assertFalse(_is_all_correct_selection_query("하나를 고르시오"))

    def test_all_incorrect(self):
        self.assertTrue(_is_all_incorrect_selection_query("틀린 것을 모두 고르시오"))

    def test_combo_reference(self):
        self.assertTrue(_is_combo_reference_text("ㄱ. ㄴ. 모두"))
        self.assertFalse(_is_combo_reference_text("일반 텍스트"))

    def test_all_correct_selection_text(self):
        self.assertTrue(_is_all_correct_selection_text("전부 고른 것"))
        self.assertFalse(_is_all_correct_selection_text("하나만"))


class TestCountHelpers(unittest.TestCase):
    def test_is_count_only(self):
        self.assertTrue(_is_count_only_text("3"))
        self.assertTrue(_is_count_only_text("2개"))
        self.assertTrue(_is_count_only_text("한개"))
        self.assertFalse(_is_count_only_text("다섯"))

    def test_extract_count_value(self):
        self.assertEqual(_extract_count_value("3"), 3)
        self.assertEqual(_extract_count_value("두개"), 2)
        self.assertIsNone(_extract_count_value("여섯개"))


class TestTokenizeOverlap(unittest.TestCase):
    def test_tokenize(self):
        tokens = _tokenize("행정심판법 제18조 대리인")
        self.assertTrue(len(tokens) > 0)

    def test_overlap_ratio_identical(self):
        ratio = _overlap_ratio("abc def", "abc def")
        self.assertEqual(ratio, 1.0)

    def test_no_overlap(self):
        ratio = _overlap_ratio("abc def", "xyz uvw")
        self.assertEqual(ratio, 0.0)

    def test_empty_statement(self):
        ratio = _overlap_ratio("abc", "")
        self.assertEqual(ratio, 0.0)


class TestNormalizeDescriptor(unittest.TestCase):
    def test_strip_particles(self):
        result = _normalize_statement_descriptor("은 대리인이다")
        self.assertNotIn("은", result.split()[0] if result else "")

    def test_strip_ending(self):
        result = _normalize_statement_descriptor("대리인 선임이다")
        self.assertNotIn("이다", result)


class TestAnchorQuery(unittest.TestCase):
    def test_law_article(self):
        query = _extract_anchor_query("행정심판법 제18조에 따라")
        self.assertEqual(query, "행정심판법 제18조")

    def test_case_no(self):
        query = _extract_anchor_query("84누180 판례에 의하면")
        self.assertEqual(query, "84누180")

    def test_no_anchor(self):
        query = _extract_anchor_query("일반 텍스트입니다")
        self.assertIsNone(query)


class TestCleanSnippet(unittest.TestCase):
    def test_short(self):
        self.assertEqual(_clean_snippet("짧은 텍스트"), "짧은 텍스트")

    def test_long(self):
        text = "가" * 400
        result = _clean_snippet(text, limit=100)
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 103)

    def test_none(self):
        self.assertEqual(_clean_snippet(""), "")


class TestAnswerPacket(unittest.TestCase):
    def test_to_dict(self):
        packet = AnswerPacket(
            query="test",
            scope="test",
            question_type="test",
            route="test",
            status="test",
            grounded=True,
            answer="test",
            evidence=[],
            citations=[],
            warnings=[],
            source_type=None,
            doc_id=None,
        )
        d = packet.to_dict()
        self.assertIn("schema_version", d)
        self.assertEqual(d["query"], "test")


class TestLlmPrompting(unittest.TestCase):
    def test_system_prompt_exists(self):
        self.assertIsInstance(SYSTEM_PROMPT, str)
        self.assertGreater(len(SYSTEM_PROMPT), 0)

    def test_build_user_prompt(self):
        packet = AnswerPacket(
            query="행정심판법 제18조",
            scope="test",
            question_type="statute_lookup",
            route="exact_law",
            status="grounded",
            grounded=True,
            answer="답변",
            evidence=["근거1"],
            citations=["[법령] 행정심판법 제18조"],
            warnings=[],
            source_type="law",
            doc_id="law:1",
        )
        prompt = build_llm_user_prompt(packet)
        self.assertIn("행정심판법 제18조", prompt)
        self.assertIn("근거1", prompt)


class TestResponseSchema(unittest.TestCase):
    def test_schema_version(self):
        self.assertIsInstance(SCHEMA_VERSION, str)
        self.assertGreater(len(SCHEMA_VERSION), 0)

    def test_answer_result_schema(self):
        self.assertIn("properties", ANSWER_RESULT_SCHEMA)
        self.assertIn("query", ANSWER_RESULT_SCHEMA["properties"])
        self.assertIn("answer", ANSWER_RESULT_SCHEMA["properties"])
        self.assertIn("explain_style", ANSWER_RESULT_SCHEMA["properties"])
        self.assertIn("teaching_explanation", ANSWER_RESULT_SCHEMA["properties"])


class _StubVerifier:
    def __init__(self, verification):
        self.verification = verification

    def verify(self, decision_strategy, result):
        return self.verification


class TestGroundedAnswererExactPolicy(unittest.TestCase):
    def test_exact_miss_returns_review_packet(self):
        answerer = GroundedAnswerer(ROOT, verifier=_StubVerifier(None))
        packet = answerer._answer_from_results("행정심판법 제999조", "exact_law_article", [])
        self.assertEqual(packet.status, "needs_review")
        self.assertFalse(packet.grounded)
        self.assertIn("broad fallback", packet.answer)

    def test_verified_exact_law_uses_official_text(self):
        answerer = GroundedAnswerer(
            ROOT,
            verifier=_StubVerifier(
                ExactVerification(
                    status="grounded",
                    grounded=True,
                    warnings=["주의: 공식 조회 검증 완료"],
                    official_text="제18조(대리인의 선임) 청구인은 대리인을 선임할 수 있다.",
                    official_ref="law.go.kr / lawId=001363 / 제18조 / verified=2026-04-04",
                )
            ),
        )
        decision, results = answerer.searcher.search("행정심판법 제18조", limit=1)
        packet = answerer._answer_from_results("행정심판법 제18조", decision.strategy, results)
        self.assertTrue(packet.grounded)
        self.assertIn("공식 조문 전문", packet.evidence[1])
        self.assertTrue(any("공식검증" in citation for citation in packet.citations))

    def test_verification_failure_downgrades_grounding(self):
        answerer = GroundedAnswerer(
            ROOT,
            verifier=_StubVerifier(
                ExactVerification(
                    status="verification_failed",
                    grounded=False,
                    warnings=["주의: 로컬과 공식 본문이 다릅니다."],
                    official_text="공식 본문",
                    official_ref="law.go.kr / lawId=001706 / 제750조 / verified=2026-04-04",
                    mismatch_reason="local_exact_mismatch",
                )
            ),
        )
        decision, results = answerer.searcher.search("행정심판법 제18조", limit=1)
        packet = answerer._answer_from_results("행정심판법 제18조", decision.strategy, results)
        self.assertEqual(packet.status, "verification_failed")
        self.assertFalse(packet.grounded)
        self.assertIn("결론 보류", packet.answer)


class _StubKoreanLawVerifier(KoreanLawExactVerifier):
    def __init__(self, responses):
        self.responses = responses
        self.api_key = "da"
        self.timeout = 20
        self.command = "korean-law"
        self.available = True

    def _run(self, args):
        key = tuple(args)
        return self.responses.get(key, (False, "missing_stub"))


class TestKoreanLawExactVerifier(unittest.TestCase):
    def test_decc_search_success_marks_grounded_even_if_get_fails(self):
        verifier = _StubKoreanLawVerifier(
            {
                (
                    "search_admin_appeals",
                    "--query",
                    "2000-04033",
                    "--display",
                    "10",
                ): (
                    True,
                    "행정심판례 검색 결과 (총 1건, 1페이지):\n\n[17109] 재확인신체검사등외판정처분취소청구\n  사건번호: N/A\n  의결일: 2000.08.07\n",
                ),
                ("get_admin_appeal_text", "--id", "17109"): (
                    False,
                    "[EXTERNAL_API_ERROR] 행정심판례를 찾을 수 없거나 응답 형식이 올바르지 않습니다.",
                ),
            }
        )
        result = UnifiedResult(
            source_type="decc",
            search_path="exact",
            score=1.0,
            payload={
                "serial_no": "17109",
                "case_no": "2000-04033",
                "case_name": "재확인신체검사등외판정처분취소청구",
                "decision_date": "20000807",
            },
        )
        verification = verifier.verify("exact_decc_case", result)
        self.assertEqual(verification.status, "grounded")
        self.assertTrue(verification.grounded)
        self.assertIn("search_admin_appeals", verification.official_ref)

    def test_decc_search_id_mismatch_fails(self):
        verifier = _StubKoreanLawVerifier(
            {
                (
                    "search_admin_appeals",
                    "--query",
                    "2000-04033",
                    "--display",
                    "10",
                ): (
                    True,
                    "행정심판례 검색 결과 (총 1건, 1페이지):\n\n[99999] 재확인신체검사등외판정처분취소청구\n  사건번호: N/A\n  의결일: 2000.08.07\n",
                ),
            }
        )
        result = UnifiedResult(
            source_type="decc",
            search_path="exact",
            score=1.0,
            payload={
                "serial_no": "17109",
                "case_no": "2000-04033",
                "case_name": "재확인신체검사등외판정처분취소청구",
                "decision_date": "20000807",
            },
        )
        verification = verifier.verify("exact_decc_case", result)
        self.assertEqual(verification.status, "verification_failed")
        self.assertFalse(verification.grounded)

    def test_expc_search_success_marks_grounded(self):
        verifier = _StubKoreanLawVerifier(
            {
                (
                    "search_interpretations",
                    "--query",
                    "05-0096",
                    "--display",
                    "10",
                ): (
                    True,
                    "해석례 검색 결과 (총 1건, 1페이지):\n\n[313107] 1959년 12월 31일 이전에 퇴직한 군인의 퇴직급여금 지급에 관한특별법 시행령 제4조제2항 및 3항\n  해석례번호: 05-0096\n  회신일자: 2005.12.23\n",
                ),
                ("get_interpretation_text", "--id", "313107"): (
                    True,
                    "기본 정보:\n  해석례번호: 313107\n  회신일자: 20051223\n",
                ),
            }
        )
        result = UnifiedResult(
            source_type="expc",
            search_path="exact",
            score=1.0,
            payload={
                "serial_no": "313107",
                "issue_no": "05-0096",
                "title": "1959년 12월 31일 이전에 퇴직한 군인의 퇴직급여금 지급에 관한특별법 시행령 제4조제2항 및 3항",
                "decision_date": "20051223",
            },
        )
        verification = verifier.verify("exact_expc_issue", result)
        self.assertEqual(verification.status, "grounded")
        self.assertTrue(verification.grounded)


class TestAdminExamExplanation(unittest.TestCase):
    def test_exam_ox_explanation_marks_x_and_conflict(self):
        answerer = GroundedAnswerer(ROOT, verifier=_StubVerifier(None))
        answerer.searcher.search = lambda query, limit=5: (type("D", (), {"strategy": "fts"})(), [])
        answerer._answer_from_results = lambda query, route, results: AnswerPacket(
            query=query,
            scope="all-laws-exam",
            question_type="exam_ox",
            route="fts",
            status="not_found",
            grounded=False,
            answer="초기 답변",
            evidence=[],
            citations=[],
            warnings=[],
            source_type=None,
            doc_id=None,
        )
        answerer._review_single_statement = lambda statement, limit=5: {
            "label": "OX",
            "text": statement,
            "anchor_query": "행정심판법 제18조",
            "status": "grounded",
            "grounded": True,
            "support_level": "unsupported",
            "support_reason": "진술이 조문과 충돌합니다.",
            "source_type": "law",
            "doc_id": "law:001363:18",
            "answer": "근거상 충돌합니다.",
            "citations": ["[법령] 행정심판법 제18조"],
        }
        packet = answerer.answer(
            "행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다. OX",
            response_mode="grounded",
            judge_mode="off",
            explain_style="admin_exam",
        )
        self.assertEqual(packet.question_type, "exam_ox")
        self.assertEqual(packet.exam_assessment["recommended_options"], ["X"])
        self.assertEqual(packet.teaching_explanation["answer_candidate"], ["X"])
        self.assertIn("충돌", packet.teaching_explanation["wrong_point"])

    def test_exam_choice_explanation_builds_breakdown(self):
        answerer = GroundedAnswerer(ROOT, verifier=_StubVerifier(None))
        answerer.searcher.search = lambda query, limit=5: (type("D", (), {"strategy": "fts"})(), [])
        answerer._answer_from_results = lambda query, route, results: AnswerPacket(
            query=query,
            scope="all-laws-exam",
            question_type="exam_choice",
            route="fts",
            status="not_found",
            grounded=False,
            answer="초기 답변",
            evidence=[],
            citations=[],
            warnings=[],
            source_type=None,
            doc_id=None,
        )
        answerer._review_options = lambda query, options, limit=5: [
            {
                "label": "①",
                "text": "행정심판법 제18조는 대리인의 선임에 관한 조항이다.",
                "status": "grounded",
                "grounded": True,
                "support_level": "supported",
                "support_reason": "조문과 직접 일치합니다.",
                "source_type": "law",
                "doc_id": "law:001363:18",
                "answer": "일치합니다.",
                "citations": ["[법령] 행정심판법 제18조"],
            },
            {
                "label": "②",
                "text": "행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.",
                "status": "grounded",
                "grounded": True,
                "support_level": "unsupported",
                "support_reason": "조문과 직접 충돌합니다.",
                "source_type": "law",
                "doc_id": "law:001363:18",
                "answer": "충돌합니다.",
                "citations": ["[법령] 행정심판법 제18조"],
            },
        ]
        packet = answerer.answer(
            "다음 중 옳은 것은?\n① 행정심판법 제18조는 대리인의 선임에 관한 조항이다.\n② 행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.",
            response_mode="grounded",
            judge_mode="off",
            explain_style="admin_exam",
        )
        self.assertEqual(packet.teaching_explanation["answer_candidate"], ["①"])
        self.assertEqual(len(packet.teaching_explanation["option_breakdown"]), 2)
        self.assertTrue(packet.teaching_explanation["ground_rule"])
        self.assertTrue(packet.teaching_explanation["wrong_point"])

    def test_admin_exam_explanation_uses_review_language_for_needs_review(self):
        packet = AnswerPacket(
            query="다음 중 옳은 것은?\n① 보기1\n② 보기2",
            scope="all-laws-exam",
            question_type="exam_choice",
            route="exact_law_article",
            status="needs_review",
            grounded=False,
            answer="문항 판단 보류",
            evidence=[],
            citations=["[법령] 행정심판법 제18조"],
            warnings=["주의: 일부 근거만 확인했습니다."],
            source_type="law",
            doc_id="law:001363:18",
            option_reviews=[
                {
                    "label": "①",
                    "text": "보기1",
                    "grounded": False,
                    "support_level": "indeterminate",
                    "support_reason": "근거 부족",
                    "citations": [],
                }
            ],
            exam_assessment={
                "status": "multiple_candidates",
                "recommended_options": ["①"],
                "reason": "하나로 확정되지 않습니다.",
            },
        )
        explanation = _build_admin_exam_explanation(packet)
        self.assertIn("후보", explanation["summary"])
        self.assertIn("보류", explanation["caution"])

    def test_admin_exam_explanation_prioritizes_verification_failed_caution(self):
        packet = AnswerPacket(
            query="행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다. OX",
            scope="all-laws-exam",
            question_type="exam_ox",
            route="exact_law_article",
            status="verification_failed",
            grounded=False,
            answer="결론 보류",
            evidence=[],
            citations=["[법령] 행정심판법 제18조"],
            warnings=["주의: 로컬과 공식 본문이 다릅니다."],
            source_type="law",
            doc_id="law:001363:18",
            option_reviews=[
                {
                    "label": "OX",
                    "text": "행정심판법 제18조에 따라 청구인은 대리인을 선임할 수 없다.",
                    "grounded": False,
                    "support_level": "indeterminate",
                    "support_reason": "로컬/공식 충돌",
                    "citations": ["[법령] 행정심판법 제18조"],
                }
            ],
            exam_assessment={
                "status": "insufficient_grounding",
                "recommended_options": [],
                "reason": "충돌로 자동 확정 불가",
            },
        )
        explanation = _build_admin_exam_explanation(packet)
        self.assertIn("충돌", explanation["caution"])
        self.assertNotIn("정답", explanation["summary"])


if __name__ == "__main__":
    unittest.main()
