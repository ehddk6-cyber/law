from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qa.legal_analysis import (
    ArticleRef,
    OxQuiz,
    QuasiProvisionChain,
    QuasiProvisionRef,
    _extract_article_refs_from_query,
    _extract_key_phrases,
    _extract_nearby_articles,
    _extract_quasi_provisions,
    _extract_verb_pattern,
    _parse_law_text_to_articles,
    build_quasi_provision_chain,
    detect_erroneous_omission,
    generate_ox_quiz,
    generate_ox_quiz_from_text,
    format_analysis_json,
    format_analysis_text,
)


class TestExtractQuasiProvisions(unittest.TestCase):
    def test_range_pattern(self):
        text = "제14조부터 제32조까지의 규정은 무효확인소송에 준용한다"
        ref = _extract_quasi_provisions(text)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.target_articles, [str(i) for i in range(14, 33)])
        self.assertEqual(ref.scope, "무효확인소송")

    def test_single_article_pattern(self):
        text = "제28조의 규정은 취소소송에 준용한다"
        ref = _extract_quasi_provisions(text)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.target_articles, ["28"])
        self.assertEqual(ref.scope, "취소소송")

    def test_no_quasi_provision(self):
        text = "이 법은 행정소송에 관하여 규정한다"
        ref = _extract_quasi_provisions(text)
        self.assertIsNone(ref)

    def test_range_with_sub_articles(self):
        text = "제14조부터 제32조까지의 규정은 무효확인의 소에 준용한다"
        ref = _extract_quasi_provisions(text)
        self.assertIsNotNone(ref)
        self.assertIn("14", ref.target_articles)
        self.assertIn("28", ref.target_articles)
        self.assertIn("32", ref.target_articles)


class TestParseLawTextToArticles(unittest.TestCase):
    def test_basic_parsing(self):
        law_text = (
            "제1조 목적 이 법은 행정소송에 관하여 규정한다. "
            "제2조 정의 이 법에서 사용하는 용어의 정의는 다음과 같다. "
            "제3조 소의 종류 행정소송은 다음 각 호의 것으로 한다."
        )
        articles = _parse_law_text_to_articles(law_text)
        self.assertEqual(len(articles), 3)
        self.assertEqual(articles[0][0], "1")
        self.assertIn("목적", articles[0][1])
        self.assertEqual(articles[1][0], "2")
        self.assertEqual(articles[2][0], "3")

    def test_sub_article_parsing(self):
        law_text = "제5조의2 특례 특례 규정은 다음과 같다."
        articles = _parse_law_text_to_articles(law_text)
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0][0], "5의2")

    def test_empty_text(self):
        articles = _parse_law_text_to_articles("")
        self.assertEqual(articles, [])


class TestBuildQuasiProvisionChain(unittest.TestCase):
    def test_omission_detected(self):
        law_text = (
            "제14조 취소소송의 제기 취소소송은 처분등이 있음을 안 날부터 90일 이내에 제기하여야 한다. "
            "제28조 사정판결 원고의 청구에 이유 없음에도 불구하고 처분등을 취소하지 아니하면 공공복리에 적합하지 아니하다고 인정한 때에는 재판소는 처분등의 취소를 기각하는 판결을 선고할 수 있다. "
            "제38조 준용 제14조부터 제27조까지의 규정은 무효확인소송에 준용한다."
        )
        reference_articles = [str(i) for i in range(14, 29)]
        chain = build_quasi_provision_chain("행정소송법", law_text, reference_articles)
        self.assertEqual(chain.law_name, "행정소송법")
        self.assertIn("28", chain.omitted_articles)
        self.assertIn("14", chain.included_articles)

    def test_all_included(self):
        law_text = (
            "제14조 취소소송의 제기 취소소송은 처분등이 있음을 안 날부터 90일 이내에 제기하여야 한다. "
            "제15조 관할 관할법원은 다음과 같다. "
            "제28조 준용 제14조부터 제15조까지의 규정은 항소소송에 준용한다."
        )
        reference_articles = ["14", "15"]
        chain = build_quasi_provision_chain("행정소송법", law_text, reference_articles)
        self.assertEqual(chain.omitted_articles, [])
        self.assertIn("14", chain.included_articles)
        self.assertIn("15", chain.included_articles)

    def test_no_quasi_provisions(self):
        law_text = "제1조 목적 이 법은 행정소송에 관하여 규정한다."
        chain = build_quasi_provision_chain("행정소송법", law_text)
        self.assertEqual(chain.quasi_provisions, [])
        self.assertEqual(chain.omitted_articles, [])


class TestDetectErroneousOmission(unittest.TestCase):
    def test_omission_found(self):
        chain = QuasiProvisionChain(
            law_name="행정소송법",
            quasi_provisions=[
                QuasiProvisionRef(
                    source_article_no="38",
                    target_articles=[str(i) for i in range(14, 28)],
                    scope="무효확인소송",
                )
            ],
            omitted_articles=["28"],
            included_articles=[str(i) for i in range(14, 28)],
        )
        findings = detect_erroneous_omission(
            chain, "취소소송", "무효확인소송", ["14", "28"]
        )
        self.assertEqual(len(findings), 1)
        self.assertIn("28", findings[0])
        self.assertIn("누락 감지", findings[0])

    def test_no_omission(self):
        chain = QuasiProvisionChain(
            law_name="행정소송법",
            quasi_provisions=[
                QuasiProvisionRef(
                    source_article_no="38",
                    target_articles=[str(i) for i in range(14, 29)],
                    scope="무효확인소송",
                )
            ],
            omitted_articles=[],
            included_articles=[str(i) for i in range(14, 29)],
        )
        findings = detect_erroneous_omission(
            chain, "취소소송", "무효확인소송", ["14", "28"]
        )
        self.assertEqual(len(findings), 0)


class TestGenerateOxQuiz(unittest.TestCase):
    def test_true_question(self):
        chain = QuasiProvisionChain(
            law_name="행정소송법",
            quasi_provisions=[
                QuasiProvisionRef(
                    source_article_no="38",
                    target_articles=[str(i) for i in range(14, 28)],
                    scope="무효확인소송",
                )
            ],
            omitted_articles=["28"],
            included_articles=[str(i) for i in range(14, 28)],
        )
        quizzes = generate_ox_quiz(chain)
        self.assertTrue(len(quizzes) >= 2)
        true_quiz = quizzes[0]
        self.assertTrue(true_quiz.answer)
        self.assertIn("준용되는 조문의 범위", true_quiz.question)

    def test_false_question_omitted_article(self):
        chain = QuasiProvisionChain(
            law_name="행정소송법",
            quasi_provisions=[
                QuasiProvisionRef(
                    source_article_no="38",
                    target_articles=[str(i) for i in range(14, 28)],
                    scope="무효확인소송",
                )
            ],
            omitted_articles=["28"],
            included_articles=[str(i) for i in range(14, 28)],
        )
        quizzes = generate_ox_quiz(chain)
        false_quizzes = [q for q in quizzes if not q.answer]
        self.assertTrue(len(false_quizzes) >= 1)
        self.assertIn("28", false_quizzes[0].question)
        self.assertIn("적용되지 않습니다", false_quizzes[0].explanation)

    def test_generate_from_text(self):
        law_text = (
            "제14조 취소소송의 제기 취소소송은 처분등이 있음을 안 날부터 90일 이내에 제기하여야 한다. "
            "제27조 보조참가 이해관계가 있는 제3자는 재판소의 허가를 얻어 당사자의 보조참가를 할 수 있다. "
            "제28조 사정판결 원고의 청구에 이유 없음에도 불구하고 처분등을 취소하지 아니하면 공공복리에 적합하지 아니하다고 인정한 때에는 재판소는 처분등의 취소를 기각하는 판결을 선고할 수 있다. "
            "제38조 준용 제14조부터 제27조까지의 규정은 무효확인소송에 준용한다."
        )
        quizzes = generate_ox_quiz_from_text("행정소송법", "38", law_text)
        self.assertTrue(len(quizzes) >= 1)
        false_quizzes = [q for q in quizzes if not q.answer]
        self.assertTrue(len(false_quizzes) >= 1)


class TestExtractHelpers(unittest.TestCase):
    def test_extract_article_refs(self):
        refs = _extract_article_refs_from_query("행정소송법 제28조 사정판결")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].law_name, "행정소송법")
        self.assertEqual(refs[0].article_no, "28")

    def test_extract_article_refs_sub(self):
        refs = _extract_article_refs_from_query("행정심판법 제5조의2")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].article_no, "5의2")

    def test_extract_nearby_articles(self):
        neighbors = _extract_nearby_articles("행정소송법", "28")
        self.assertIn("26", neighbors)
        self.assertIn("27", neighbors)
        self.assertIn("29", neighbors)
        self.assertIn("30", neighbors)
        self.assertNotIn("28", neighbors)

    def test_extract_key_phrases(self):
        phrases = _extract_key_phrases("취소하는 것이 공공복리에 현저히 부적합하다")
        self.assertIn("취소하는", phrases)
        self.assertIn("공공복리", phrases)
        self.assertIn("현저히", phrases)

    def test_extract_verb_pattern(self):
        self.assertEqual(_extract_verb_pattern("취소하는 것이 적합하다"), "취소")
        self.assertEqual(_extract_verb_pattern("확인하는 것이 타당하다"), "확인")
        self.assertEqual(_extract_verb_pattern("기각한다"), "")


if __name__ == "__main__":
    unittest.main()
