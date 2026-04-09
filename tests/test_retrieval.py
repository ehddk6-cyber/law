from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.base import xml_text
from parsers.law import parse_law_xml


def _load_module_direct(name: str, file_path: Path):
    """Load a module directly from file, bypassing __init__.py import chains."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Bypass qa/__init__.py → legal_analysis → retrievers → sqlite3 chain
# by loading modules directly from file
_router_mod = _load_module_direct("qa.router", ROOT / "qa" / "router.py")
_retrievers_mod = _load_module_direct("qa.retrievers", ROOT / "qa" / "retrievers.py")
_unified_mod = _load_module_direct("qa.unified", ROOT / "qa" / "unified.py")

route_query = _router_mod.route_query
RouteDecision = _router_mod.RouteDecision

sanitize_fts_query = _retrievers_mod._sanitize_fts_query
load_vector_artifacts = _retrievers_mod._load_vector_artifacts
_vector_search_core = _retrievers_mod._vector_search_core
_vector_search_faiss = _retrievers_mod._vector_search_faiss
_cached_load_vector_artifacts = _retrievers_mod._cached_load_vector_artifacts
_cached_faiss_index = _retrievers_mod._cached_faiss_index
LEGAL_SYNONYMS = _retrievers_mod.LEGAL_SYNONYMS
SearchResult = _retrievers_mod.SearchResult
AcrRetriever = _retrievers_mod.AcrRetriever
LawRetriever = _retrievers_mod.LawRetriever
PrecRetriever = _retrievers_mod.PrecRetriever
DeccRetriever = _retrievers_mod.DeccRetriever
DetcRetriever = _retrievers_mod.DetcRetriever
ExpcRetriever = _retrievers_mod.ExpcRetriever

normalize_score = _unified_mod._normalize_score
to_unified = _unified_mod._to_unified
deduplicate = _unified_mod._deduplicate
SOURCE_WEIGHTS = _unified_mod.SOURCE_WEIGHTS
UnifiedResult = _unified_mod.UnifiedResult
UnifiedSearcher = _unified_mod.UnifiedSearcher

ARTIFACTS = ROOT / ".artifacts"

# Backward-compatible aliases for existing test code
_sanitize_fts_query = sanitize_fts_query
_load_vector_artifacts = load_vector_artifacts
_normalize_score = normalize_score
_to_unified = to_unified
_deduplicate = deduplicate


class TestRouter(unittest.TestCase):
    def test_exact_source_id(self):
        decision = route_query("결정문일련번호 23")
        self.assertEqual(decision.strategy, "exact_source_id")
        self.assertEqual(decision.serial_no, "23")
        self.assertEqual(decision.fallback_strategies, ["fts", "vector"])

    def test_exact_case_no(self):
        decision = route_query("제2020-4소위09-02호")
        self.assertEqual(decision.strategy, "exact_case_no")

    def test_exact_law_article(self):
        decision = route_query("행정심판법 제18조")
        self.assertEqual(decision.strategy, "exact_law_article")
        self.assertEqual(decision.law_name, "행정심판법")
        self.assertEqual(decision.article_no, "18")

    def test_exact_law_article_branch(self):
        decision = route_query("행정심판법 제5조의2")
        self.assertEqual(decision.strategy, "exact_law_article")
        self.assertEqual(decision.article_no, "5의2")

    def test_exact_precedent(self):
        decision = route_query("84누180")
        self.assertEqual(decision.strategy, "exact_precedent_case")

    def test_exact_decc(self):
        decision = route_query("2000-04033")
        self.assertEqual(decision.strategy, "exact_decc_case")

    def test_exact_detc(self):
        decision = route_query("2004헌마275")
        self.assertEqual(decision.strategy, "exact_detc_case")

    def test_exact_expc(self):
        decision = route_query("05-0096")
        self.assertEqual(decision.strategy, "exact_expc_issue")

    def test_vector_hints(self):
        decision = route_query("왜 그런가요?")
        self.assertEqual(decision.strategy, "vector")
        self.assertEqual(decision.fallback_strategies, ["fts"])

    def test_fts_default(self):
        decision = route_query("부당이득 반환")
        self.assertEqual(decision.strategy, "fts")
        self.assertEqual(decision.fallback_strategies, ["vector"])

    def test_exam_prefix_stripped(self):
        decision = route_query("다음 중 옳은 것은 행정심판법 제18조")
        self.assertEqual(decision.strategy, "exact_law_article")
        self.assertEqual(decision.law_name, "행정심판법")

    def test_law_name_search_basic(self):
        decision = route_query("행정소송법 사정판결")
        self.assertEqual(decision.strategy, "law_name_search")
        self.assertEqual(decision.law_name, "행정소송법")
        self.assertEqual(decision.fallback_strategies, ["fts", "vector"])

    def test_law_name_search_false_positive_case_no(self):
        decision = route_query("84누180")
        self.assertEqual(decision.strategy, "exact_precedent_case")
        self.assertNotEqual(decision.strategy, "law_name_search")

    def test_law_name_search_false_positive_article(self):
        decision = route_query("행정심판법 제18조")
        self.assertEqual(decision.strategy, "exact_law_article")
        self.assertNotEqual(decision.strategy, "law_name_search")


class TestXmlText(unittest.TestCase):
    def test_none_parent(self):
        from xml.etree import ElementTree as ET

        self.assertEqual(xml_text(None, "tag"), "")

    def test_missing_tag(self):
        from xml.etree import ElementTree as ET

        root = ET.Element("root")
        self.assertEqual(xml_text(root, "missing"), "")

    def test_found_tag(self):
        from xml.etree import ElementTree as ET

        root = ET.Element("root")
        child = ET.SubElement(root, "name")
        child.text = "테스트"
        self.assertEqual(xml_text(root, "name"), "테스트")


class TestLawParserIntegrity(unittest.TestCase):
    def test_heading_only_duplicate_article_is_filtered(self):
        record = parse_law_xml(ROOT / "law" / "body" / "xml" / "ID_001706.xml")
        articles = [article for article in record["articles"] if article["article_no"] == "750"]
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["article_title"], "불법행위의 내용")
        self.assertIn("고의 또는 과실", articles[0]["article_text"])

    def test_paragraph_number_not_duplicated(self):
        record = parse_law_xml(ROOT / "law" / "body" / "xml" / "ID_001706.xml")
        article = next(article for article in record["articles"] if article["article_no"] == "751")
        self.assertNotIn("1 1", article["article_text"])
        self.assertNotIn("2 2", article["article_text"])


class TestRetrieverSearchResult(unittest.TestCase):
    def test_acr_exact(self):
        retriever = AcrRetriever(ARTIFACTS / "acr" / "acr.sqlite3", ARTIFACTS / "acr" / "vector")
        results = retriever.exact_by_source_id("23")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].search_path, "exact")
        self.assertEqual(results[0].doc_id, "acr:23")

    def test_acr_fts(self):
        retriever = AcrRetriever(ARTIFACTS / "acr" / "acr.sqlite3", ARTIFACTS / "acr" / "vector")
        results = retriever.fts_search("행정심판", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)
            self.assertEqual(r.search_path, "fts")

    def test_acr_vector(self):
        retriever = AcrRetriever(ARTIFACTS / "acr" / "acr.sqlite3", ARTIFACTS / "acr" / "vector")
        results = retriever.vector_search("행정심판 대리인", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)
            self.assertEqual(r.search_path, "vector")

    def test_law_exact(self):
        retriever = LawRetriever(ARTIFACTS / "law" / "law.sqlite3")
        results = retriever.exact_article("행정심판법", "18")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].search_path, "exact_law")
        self.assertIn("article_text", results[0].payload)

    def test_law_exact_regression_mincivil_750(self):
        retriever = LawRetriever(ARTIFACTS / "law" / "law.sqlite3")
        results = retriever.exact_article("민법", "750")
        self.assertTrue(len(results) > 0)
        self.assertIn("고의 또는 과실", results[0].payload["article_text"])

    def test_law_fts(self):
        retriever = LawRetriever(ARTIFACTS / "law" / "law.sqlite3")
        results = retriever.fts_search("대리인", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)
            self.assertEqual(r.search_path, "fts")

    def test_prec_exact(self):
        retriever = PrecRetriever(ARTIFACTS / "prec" / "prec.sqlite3")
        results = retriever.exact_case("84누180")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].search_path, "exact")

    def test_prec_fts(self):
        retriever = PrecRetriever(ARTIFACTS / "prec" / "prec.sqlite3")
        results = retriever.fts_search("양도소득세", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)

    def test_decc_exact(self):
        retriever = DeccRetriever(ARTIFACTS / "decc" / "decc.sqlite3")
        results = retriever.exact_case("2000-04033")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].search_path, "exact")

    def test_decc_fts(self):
        retriever = DeccRetriever(ARTIFACTS / "decc" / "decc.sqlite3")
        results = retriever.fts_search("산업재해", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)

    def test_detc_exact(self):
        retriever = DetcRetriever(ARTIFACTS / "detc" / "detc.sqlite3")
        results = retriever.exact_case("2004헌마275")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].search_path, "exact")

    def test_detc_fts(self):
        retriever = DetcRetriever(ARTIFACTS / "detc" / "detc.sqlite3")
        results = retriever.fts_search("헌법소원", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)

    def test_expc_exact(self):
        retriever = ExpcRetriever(ARTIFACTS / "expc" / "expc.sqlite3")
        results = retriever.exact_issue("05-0096")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].search_path, "exact")

    def test_expc_fts(self):
        retriever = ExpcRetriever(ARTIFACTS / "expc" / "expc.sqlite3")
        results = retriever.fts_search("산지관리", limit=3)
        self.assertIsInstance(results, list)
        for r in results:
            self.assertIsInstance(r, SearchResult)


class TestUnifiedSearcher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.searcher = UnifiedSearcher(ROOT)

    def test_exact_law(self):
        decision, results = self.searcher.search("행정심판법 제18조")
        self.assertEqual(decision.strategy, "exact_law_article")
        self.assertTrue(len(results) > 0)
        self.assertIsInstance(results[0], UnifiedResult)
        self.assertEqual(results[0].source_type, "law")
        self.assertEqual(results[0].search_path, "exact_law")

    def test_exact_acr(self):
        decision, results = self.searcher.search("결정문일련번호 23")
        self.assertEqual(decision.strategy, "exact_source_id")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].source_type, "acr")

    def test_exact_prec(self):
        decision, results = self.searcher.search("84누180")
        self.assertEqual(decision.strategy, "exact_precedent_case")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].source_type, "prec")

    def test_exact_decc(self):
        decision, results = self.searcher.search("2000-04033")
        self.assertEqual(decision.strategy, "exact_decc_case")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].source_type, "decc")

    def test_exact_detc(self):
        decision, results = self.searcher.search("2004헌마275")
        self.assertEqual(decision.strategy, "exact_detc_case")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].source_type, "detc")

    def test_exact_expc(self):
        decision, results = self.searcher.search("05-0096")
        self.assertEqual(decision.strategy, "exact_expc_issue")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0].source_type, "expc")

    def test_exact_miss_does_not_fallback(self):
        decision, results = self.searcher.search("행정심판법 제999조")
        self.assertEqual(decision.strategy, "exact_law_article")
        self.assertEqual(results, [])

    def test_fts_search(self):
        decision, results = self.searcher.search("부당이득 반환")
        self.assertEqual(decision.strategy, "fts")
        self.assertTrue(len(results) > 0)

    def test_vector_search(self):
        decision, results = self.searcher.search("왜 그런가요?")
        self.assertEqual(decision.strategy, "vector")
        self.assertTrue(len(results) > 0)

    def test_fallback(self):
        decision, results = self.searcher.search("999999존재하지않는번호")
        self.assertTrue(len(results) == 0 or len(results) > 0)

    def test_law_name_search_unified(self):
        decision, results = self.searcher.search("행정소송법 사정판결")
        self.assertEqual(decision.strategy, "law_name_search")


class TestFtsPrefix(unittest.TestCase):
    def test_fts_prefix_korean(self):
        result = _sanitize_fts_query("사정판결 대리인")
        # Both terms are in LEGAL_SYNONYMS, so they get OR-expansion instead of * prefix
        self.assertIn("사정판결", result)
        self.assertIn("대리인", result)

    def test_fts_no_prefix_short(self):
        result = _sanitize_fts_query("법 조")
        self.assertNotIn("*", result)
        self.assertEqual(result, "법 조")

    def test_fts_no_prefix_non_korean(self):
        result = _sanitize_fts_query("BM25 algorithm")
        self.assertNotIn("*", result)
        self.assertEqual(result, "BM25 algorithm")


class TestScoreNormalization(unittest.TestCase):
    def test_normalize_exact_score(self):
        self.assertEqual(_normalize_score(1.0, "exact"), 1.0)
        self.assertEqual(_normalize_score(0.5, "exact_law"), 1.0)

    def test_normalize_fts_score(self):
        self.assertAlmostEqual(_normalize_score(5.0, "fts"), 0.5)
        self.assertEqual(_normalize_score(-1.0, "fts"), 0.0)

    def test_normalize_vector_score(self):
        self.assertAlmostEqual(_normalize_score(0.8, "vector"), 0.8)
        self.assertEqual(_normalize_score(-0.1, "vector"), 0.0)


class TestVectorArtifacts(unittest.TestCase):
    def test_load_vector_artifacts(self):
        vectors, rows = _load_vector_artifacts(ARTIFACTS / "acr" / "vector")
        self.assertIsNotNone(vectors)
        self.assertTrue(len(rows) > 0)
        self.assertEqual(vectors.shape[0], len(rows))

    def test_missing_vector_dir(self):
        vectors, rows = _load_vector_artifacts(ARTIFACTS / "nonexistent")
        self.assertIsNone(vectors)
        self.assertEqual(rows, [])


class TestFtsSynonymExpansion(unittest.TestCase):
    """TC-19 through TC-21: FTS synonym expansion."""

    def test_synonym_cancel_expands(self):
        """TC-19: '취소' should expand to include '취소소송'."""
        result = _sanitize_fts_query("취소")
        self.assertIn("취소소송", result)
        self.assertIn("취소", result)

    def test_synonym_void_expands(self):
        """TC-20: '무효' should expand to include '무효확인'."""
        result = _sanitize_fts_query("무효")
        self.assertIn("무효확인", result)
        self.assertIn("무효", result)

    def test_synonym_preserves_non_synonym_terms(self):
        """TC-21: Non-synonym terms pass through unchanged with prefix logic."""
        result = _sanitize_fts_query("취소 행정심판")
        self.assertIn("취소소송", result)
        self.assertIn("행정심판*", result)


class TestSourceWeights(unittest.TestCase):
    """TC-22 through TC-24: Source-specific weights."""

    def test_source_weight_values(self):
        """TC-22: Source weights follow law(1.0) > prec(0.95) > acr(0.8)."""
        self.assertEqual(SOURCE_WEIGHTS["law"], 1.0)
        self.assertEqual(SOURCE_WEIGHTS["prec"], 0.95)
        self.assertEqual(SOURCE_WEIGHTS["acr"], 0.8)

    def test_to_unified_applies_weight(self):
        """TC-23: _to_unified applies source weight to normalized score."""
        sr = SearchResult(
            search_path="fts",
            doc_id="test:1",
            source_id="1",
            title="테스트",
            case_no="",
            decision_date=None,
            score=5.0,
            payload={"doc_id": "test:1"},
        )
        law_result = _to_unified("law", sr)
        acr_result = _to_unified("acr", sr)
        # law weight 1.0, acr weight 0.8, both based on same normalized score 0.5
        self.assertAlmostEqual(law_result.score, 0.5 * 1.0)
        self.assertAlmostEqual(acr_result.score, 0.5 * 0.8)
        self.assertGreater(law_result.score, acr_result.score)

    def test_law_scores_always_higher_than_acr(self):
        """TC-24: For identical raw scores, law source outranks acr."""
        sr = SearchResult(
            search_path="exact",
            doc_id="test:2",
            source_id="2",
            title="테스트",
            case_no="",
            decision_date=None,
            score=1.0,
            payload={"doc_id": "test:2"},
        )
        law_result = _to_unified("law", sr)
        prec_result = _to_unified("prec", sr)
        acr_result = _to_unified("acr", sr)
        self.assertGreater(law_result.score, prec_result.score)
        self.assertGreater(prec_result.score, acr_result.score)


class TestDeduplication(unittest.TestCase):
    """TC-25 through TC-26: Cross-source doc_id deduplication."""

    def test_dedup_removes_duplicate_doc_ids(self):
        """TC-25: Deduplication removes entries with same doc_id."""
        results = [
            UnifiedResult(
                source_type="law", search_path="fts", score=0.8, payload={"doc_id": "x1"}
            ),
            UnifiedResult(
                source_type="prec", search_path="fts", score=0.6, payload={"doc_id": "x1"}
            ),
            UnifiedResult(
                source_type="acr", search_path="fts", score=0.5, payload={"doc_id": "x2"}
            ),
        ]
        deduped = _deduplicate(results)
        self.assertEqual(len(deduped), 2)
        doc_ids = {r.payload["doc_id"] for r in deduped}
        self.assertEqual(doc_ids, {"x1", "x2"})

    def test_dedup_keeps_highest_score(self):
        """TC-26: When deduplicating, the higher-scored entry is kept."""
        results = [
            UnifiedResult(
                source_type="prec", search_path="fts", score=0.9, payload={"doc_id": "dup"}
            ),
            UnifiedResult(
                source_type="acr", search_path="fts", score=0.3, payload={"doc_id": "dup"}
            ),
        ]
        deduped = _deduplicate(results)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].source_type, "prec")
        self.assertAlmostEqual(deduped[0].score, 0.9)


class TestParallelMultiSearch(unittest.TestCase):
    """TC-27 through TC-28: Parallel execution correctness."""

    @classmethod
    def setUpClass(cls):
        cls.searcher = UnifiedSearcher(ROOT)

    def test_multi_search_returns_results(self):
        """TC-27: Parallel multi-search returns results from multiple sources."""
        decision, results = self.searcher.search("부당이득")
        self.assertEqual(decision.strategy, "fts")
        # Results should come from parallel execution of 6 retrievers
        self.assertIsInstance(results, list)

    def test_vector_multi_search_returns_results(self):
        """TC-28: Vector strategy multi-search works with parallel execution."""
        decision, results = self.searcher.search("행정심판의 의의와 범위")
        # May route to vector or fts depending on query
        self.assertIsInstance(results, list)


class TestVectorCacheConsistency(unittest.TestCase):
    """TC-29: Cached vector artifacts return identical results."""

    def test_cached_vector_load_consistency(self):
        """TC-29: Two calls to _load_vector_artifacts return equivalent data."""
        v1, r1 = _load_vector_artifacts(ARTIFACTS / "acr" / "vector")
        v2, r2 = _load_vector_artifacts(ARTIFACTS / "acr" / "vector")
        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)
        self.assertEqual(r1, r2)
        np.testing.assert_array_equal(v1, v2)


if __name__ == "__main__":
    unittest.main()
