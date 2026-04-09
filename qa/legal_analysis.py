from __future__ import annotations

import re
from dataclasses import dataclass, field

from qa.retrievers import LawRetriever, SearchResult


@dataclass
class ArticleRef:
    law_name: str
    article_no: str
    article_title: str = ""
    doc_id: str = ""


@dataclass
class QuasiProvisionRef:
    source_article_no: str
    target_articles: list[str]
    scope: str = ""


@dataclass
class QuasiProvisionChain:
    law_name: str
    quasi_provisions: list[QuasiProvisionRef]
    omitted_articles: list[str] = field(default_factory=list)
    included_articles: list[str] = field(default_factory=list)


@dataclass
class OxQuiz:
    question: str
    answer: bool
    explanation: str
    related_articles: list[str] = field(default_factory=list)


@dataclass
class ArticleCompare:
    ref: ArticleRef
    full_text: str
    key_phrases: list[str] = field(default_factory=list)
    match_level: str = ""


@dataclass
class AnalysisResult:
    query: str
    matched_articles: list[ArticleCompare]
    related_articles: list[ArticleCompare]
    differences: list[str] = field(default_factory=list)
    verdict: str = ""
    evidence: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    quasi_chain: QuasiProvisionChain | None = None
    ox_quizzes: list[OxQuiz] = field(default_factory=list)


def _extract_key_phrases(text: str) -> list[str]:
    markers = [
        "현저히",
        "공공복리",
        "취소",
        "무효확인",
        "기각",
        "각하",
        "확인하는 것이",
        "취소하는 것이",
        "확인하는",
        "취소하는",
        "부적합",
        "적합",
    ]
    return [m for m in markers if m in text]


def _extract_verb_pattern(text: str) -> str:
    if "취소하는 것이" in text or "취소하는" in text:
        return "취소"
    if "확인하는 것이" in text or "확인하는" in text:
        return "확인"
    return ""


def _extract_article_refs_from_query(query: str) -> list[ArticleRef]:
    refs: list[ArticleRef] = []
    law_article_re = re.compile(r"(?:(.+?)\s+)?제\s*(\d+)(?:조의\s*(\d+)|의\s*(\d+)조|조)")
    for m in law_article_re.finditer(query):
        law_name = (m.group(1) or "").strip()
        article_main = m.group(2)
        article_sub = m.group(3) or m.group(4)
        article_no = article_main if not article_sub else f"{article_main}의{article_sub}"
        if law_name:
            refs.append(ArticleRef(law_name=law_name, article_no=article_no))
    return refs


def _extract_nearby_articles(law_name: str, article_no: str) -> list[str]:
    try:
        base = int(article_no)
    except ValueError:
        return []
    neighbors: list[str] = []
    for offset in (-2, -1, 1, 2):
        candidate = base + offset
        if candidate > 0:
            neighbors.append(str(candidate))
    return neighbors


def _fetch_articles(
    retriever: LawRetriever, law_name: str, article_nos: list[str], limit: int = 5
) -> list[tuple[ArticleRef, SearchResult]]:
    results: list[tuple[ArticleRef, SearchResult]] = []
    seen: set[str] = set()
    for article_no in article_nos:
        hits = retriever.exact_article(law_name, article_no, limit=limit)
        best: tuple[ArticleRef, SearchResult] | None = None
        for hit in hits:
            payload = hit.payload
            title = (payload.get("article_title") or "").strip()
            text = payload.get("article_text", "")
            if not title and len(text) < 20:
                continue
            key = f"{payload.get('law_name', '')}:{payload.get('article_no', '')}"
            if key in seen:
                continue
            ref = ArticleRef(
                law_name=payload.get("law_name", ""),
                article_no=payload.get("article_no", ""),
                article_title=title,
                doc_id=payload.get("doc_id", ""),
            )
            if best is None or len(title) > len(best[0].article_title):
                best = (ref, hit)
        if best:
            key = f"{best[0].law_name}:{best[0].article_no}"
            seen.add(key)
            results.append(best)
    return results


def _extract_quasi_provisions(text: str) -> QuasiProvisionRef | None:
    quasi_match = re.search(
        r"제\s*(\d+)조(?:의\s*(\d+))?\s*부터\s*제\s*(\d+)조(?:의\s*(\d+))?\s*까지의?\s*규정[은는]?\s*(.+?)에\s*준용한다",
        text,
    )
    if quasi_match:
        start = int(quasi_match.group(1))
        end = int(quasi_match.group(3))
        scope = quasi_match.group(5).strip()
        targets = [str(i) for i in range(start, end + 1)]
        return QuasiProvisionRef(
            source_article_no="",
            target_articles=targets,
            scope=scope,
        )

    single_match = re.search(
        r"제\s*(\d+)조(?:의\s*(\d+))?\s*의?\s*규정[은는]?\s*(.+?)에\s*준용한다",
        text,
    )
    if single_match:
        article_no = single_match.group(1)
        scope = single_match.group(3).strip()
        return QuasiProvisionRef(
            source_article_no="",
            target_articles=[article_no],
            scope=scope,
        )

    return None


def _parse_law_text_to_articles(law_text: str) -> list[tuple[str, str]]:
    article_pattern = re.compile(r"제\s*\d+조(?:의\s*\d+)?(?!\s*(?:부터|까지))")
    articles: list[tuple[str, str]] = []
    matches = list(article_pattern.finditer(law_text))
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(law_text)
        body = law_text[start:end].strip()
        label = m.group(0)
        m2 = re.search(r"제\s*(\d+)(?:조의\s*(\d+))?", label)
        if m2:
            article_no = m2.group(1)
            if m2.group(2):
                article_no = f"{article_no}의{m2.group(2)}"
            articles.append((article_no, body))
    return articles


def build_quasi_provision_chain(
    law_name: str,
    law_text: str,
    reference_articles: list[str] | None = None,
) -> QuasiProvisionChain:
    articles = _parse_law_text_to_articles(law_text)
    quasi_refs: list[QuasiProvisionRef] = []
    included: set[str] = set()

    for article_no, article_body in articles:
        ref = _extract_quasi_provisions(article_body)
        if ref:
            ref.source_article_no = article_no
            quasi_refs.append(ref)
            included.update(ref.target_articles)

    omitted: list[str] = []
    if reference_articles:
        omitted = [a for a in reference_articles if a not in included]

    return QuasiProvisionChain(
        law_name=law_name,
        quasi_provisions=quasi_refs,
        omitted_articles=omitted,
        included_articles=sorted(included),
    )


def detect_erroneous_omission(
    chain: QuasiProvisionChain,
    source_section: str,
    target_section: str,
    expected_included: list[str],
) -> list[str]:
    findings: list[str] = []
    for article_no in expected_included:
        if article_no in chain.omitted_articles:
            findings.append(
                f"[준용규정 누락 감지] {chain.law_name} "
                f"제{chain.quasi_provisions[0].source_article_no if chain.quasi_provisions else '?'}조 "
                f"({source_section}→{target_section} 준용)에 "
                f"제{article_no}조가 포함되어 있지 않습니다. "
                f"이는 {source_section}의 제{article_no}조 규정이 "
                f"{target_section}에 적용되지 않음을 의미합니다."
            )
    return findings


def generate_ox_quiz(chain: QuasiProvisionChain) -> list[OxQuiz]:
    quizzes: list[OxQuiz] = []

    for qref in chain.quasi_provisions:
        quizzes.append(
            OxQuiz(
                question=(
                    f"{chain.law_name} 제{qref.source_article_no}조의 준용규정에 따라 "
                    f"{qref.scope}에 준용되는 조문의 범위는 "
                    f"제{qref.target_articles[0]}조부터 "
                    f"제{qref.target_articles[-1]}조까지이다."
                ),
                answer=True,
                explanation=(
                    f"{chain.law_name} 제{qref.source_article_no}조에서 "
                    f"제{qref.target_articles[0]}조부터 제{qref.target_articles[-1]}조까지의 "
                    f"규정을 {qref.scope}에 준용하도록 정하고 있습니다."
                ),
                related_articles=[qref.source_article_no] + qref.target_articles,
            )
        )

    for omitted_no in chain.omitted_articles:
        quizzes.append(
            OxQuiz(
                question=(
                    f"{chain.law_name}에서 {chain.quasi_provisions[0].scope if chain.quasi_provisions else '관련'} 소송에 "
                    f"제{omitted_no}조의 규정이 준용된다."
                ),
                answer=False,
                explanation=(
                    f"{chain.law_name}의 준용규정(제{chain.quasi_provisions[0].source_article_no if chain.quasi_provisions else '?'}조)에서 "
                    f"열거한 조문 범위에 제{omitted_no}조가 포함되어 있지 않으므로, "
                    f"제{omitted_no}조의 규정은 해당 소송 유형에 적용되지 않습니다."
                ),
                related_articles=[omitted_no],
            )
        )

    return quizzes


def generate_ox_quiz_from_text(law_name: str, quasi_article_no: str, law_text: str) -> list[OxQuiz]:
    articles = _parse_law_text_to_articles(law_text)
    article_nos = [a[0] for a in articles]

    quasi_ref: QuasiProvisionRef | None = None
    for article_no, article_body in articles:
        if article_no == quasi_article_no:
            quasi_ref = _extract_quasi_provisions(article_body)
            if quasi_ref:
                quasi_ref.source_article_no = article_no
            break

    included = set()
    if quasi_ref:
        included.update(quasi_ref.target_articles)

    omitted = [a for a in article_nos if a not in included and a != quasi_article_no]

    chain = QuasiProvisionChain(
        law_name=law_name,
        quasi_provisions=[quasi_ref] if quasi_ref else [],
        omitted_articles=omitted,
        included_articles=sorted(included),
    )
    quizzes = generate_ox_quiz(chain)
    return quizzes


def analyze_law_query(query: str, law_retriever: LawRetriever) -> AnalysisResult:
    query_refs = _extract_article_refs_from_query(query)
    if not query_refs:
        return AnalysisResult(
            query=query,
            matched_articles=[],
            related_articles=[],
            verdict="쿼리에서 조문 참조를 찾지 못했습니다.",
        )

    primary_ref = query_refs[0]
    all_article_nos = [primary_ref.article_no]
    neighbors = _extract_nearby_articles(primary_ref.law_name, primary_ref.article_no)
    all_article_nos.extend(neighbors)

    fetched = _fetch_articles(law_retriever, primary_ref.law_name, all_article_nos)

    matched: list[ArticleCompare] = []
    related: list[ArticleCompare] = []

    query_phrases = _extract_key_phrases(query)

    for ref, hit in fetched:
        payload = hit.payload
        text = payload.get("article_text", "")
        title = payload.get("article_title", "").strip()
        if not title and len(text) < 20:
            continue
        phrases = _extract_key_phrases(text)
        ref.article_title = title
        ref.doc_id = payload.get("doc_id", "")

        comp = ArticleCompare(
            ref=ref,
            full_text=text,
            key_phrases=phrases,
        )

        if ref.article_no == primary_ref.article_no:
            comp.match_level = "primary"
            matched.append(comp)
        else:
            comp.match_level = "neighbor"
            related.append(comp)

    differences: list[str] = []
    for rc in related:
        diff_phrases = set(query_phrases) - set(rc.key_phrases)
        extra_phrases = set(rc.key_phrases) - set(query_phrases)
        if diff_phrases or extra_phrases:
            parts = []
            if diff_phrases:
                parts.append(f"쿼리에만 있는 표현: {', '.join(diff_phrases)}")
            if extra_phrases:
                parts.append(f"제{rc.ref.article_no}조에만 있는 표현: {', '.join(extra_phrases)}")
            differences.append(
                f"[제{rc.ref.article_no}조 {rc.ref.article_title}] " + " / ".join(parts)
            )

    evidence: list[str] = []
    citations: list[str] = []
    for mc in matched:
        evidence.append(f"제{mc.ref.article_no}조({mc.ref.article_title}): {mc.full_text[:300]}")
        citations.append(f"[법령] {mc.ref.law_name} 제{mc.ref.article_no}조")
    for rc in related:
        evidence.append(f"제{rc.ref.article_no}조({rc.ref.article_title}): {rc.full_text[:300]}")
        citations.append(f"[법령] {rc.ref.law_name} 제{rc.ref.article_no}조")

    verdict = ""
    if matched:
        query_verb = _extract_verb_pattern(query)
        primary_text = matched[0].full_text
        primary_verb = _extract_verb_pattern(primary_text)

        if query_verb and primary_verb and query_verb != primary_verb:
            verdict += (
                f"핵심 동사 패턴 불일치: 쿼리는 '{query_verb}'를 사용하지만, "
                f"제{matched[0].ref.article_no}조 원문은 '{primary_verb}'를 사용합니다. "
                f"이는 완전히 다른 소송 유형입니다.\n"
            )

        primary_phrases = matched[0].key_phrases
        query_set = set(query_phrases)
        primary_set = set(primary_phrases)

        missing_from_query = primary_set - query_set
        extra_in_query = query_set - primary_set

        if missing_from_query:
            verdict += (
                f"쿼리에 빠진 중요 표현: {', '.join(missing_from_query)}. "
                f"이 표현이 제{matched[0].ref.article_no}조 원문에는 있습니다.\n"
            )
        if extra_in_query:
            verdict += f"쿼리에만 있고 원문에 없는 표현: {', '.join(extra_in_query)}.\n"

    for rc in related:
        if matched:
            primary_phrases = matched[0].key_phrases
            if primary_phrases != rc.key_phrases:
                verdict += (
                    f"제{matched[0].ref.article_no}조와 제{rc.ref.article_no}조는 "
                    f"핵심 표현이 다릅니다. "
                    f"전자는 [{', '.join(primary_phrases)}], "
                    f"후자는 [{', '.join(rc.key_phrases)}]를 포함합니다.\n"
                )

    return AnalysisResult(
        query=query,
        matched_articles=matched,
        related_articles=related,
        differences=differences,
        verdict=verdict.strip()
        if verdict
        else "주변 조문과의 핵심 표현 차이가 발견되지 않았습니다.",
        evidence=evidence,
        citations=citations,
    )


def format_analysis_text(result: AnalysisResult) -> str:
    lines: list[str] = []
    lines.append(f"쿼리: {result.query}")
    lines.append("")

    if result.matched_articles:
        lines.append("=== 매칭된 조문 ===")
        for mc in result.matched_articles:
            lines.append(f"  {mc.ref.law_name} 제{mc.ref.article_no}조 {mc.ref.article_title}")
            lines.append(
                f"  핵심 표현: {', '.join(mc.key_phrases) if mc.key_phrases else '(없음)'}"
            )
            lines.append(f"  원문: {mc.full_text[:500]}")
            lines.append("")

    if result.related_articles:
        lines.append("=== 인접 조문 (비교 대상) ===")
        for rc in result.related_articles:
            lines.append(f"  {rc.ref.law_name} 제{rc.ref.article_no}조 {rc.ref.article_title}")
            lines.append(
                f"  핵심 표현: {', '.join(rc.key_phrases) if rc.key_phrases else '(없음)'}"
            )
            lines.append(f"  원문: {rc.full_text[:500]}")
            lines.append("")

    if result.differences:
        lines.append("=== 차이점 ===")
        for d in result.differences:
            lines.append(f"  {d}")
        lines.append("")

    lines.append(f"=== 분석 ===")
    lines.append(result.verdict)

    if result.quasi_chain and result.quasi_chain.quasi_provisions:
        lines.append("")
        lines.append("=== 준용규정 추적 ===")
        for qref in result.quasi_chain.quasi_provisions:
            lines.append(
                f"  제{qref.source_article_no}조: "
                f"제{qref.target_articles[0]}조~제{qref.target_articles[-1]}조 "
                f"→ {qref.scope}에 준용"
            )
        if result.quasi_chain.omitted_articles:
            lines.append(
                f"  준용 범위에서 누락된 조문: {', '.join(result.quasi_chain.omitted_articles)}"
            )
        if result.quasi_chain.included_articles:
            lines.append(
                f"  준용 범위에 포함된 조문: {', '.join(result.quasi_chain.included_articles)}"
            )

    if result.ox_quizzes:
        lines.append("")
        lines.append("=== OX 문제 ===")
        for i, quiz in enumerate(result.ox_quizzes, 1):
            lines.append(f"  [{i}] Q: {quiz.question}")
            lines.append(f"      A: {'O (참)' if quiz.answer else 'X (거짓)'}")
            lines.append(f"      해설: {quiz.explanation}")
            if quiz.related_articles:
                lines.append(
                    f"      관련 조문: {', '.join(f'제{a}조' for a in quiz.related_articles)}"
                )

    if result.citations:
        lines.append("")
        lines.append("=== 인용 ===")
        for c in result.citations:
            lines.append(f"  {c}")

    return "\n".join(lines)


def format_analysis_json(result: AnalysisResult) -> dict:
    payload: dict = {
        "query": result.query,
        "matched_articles": [
            {
                "law_name": mc.ref.law_name,
                "article_no": mc.ref.article_no,
                "article_title": mc.ref.article_title,
                "doc_id": mc.ref.doc_id,
                "key_phrases": mc.key_phrases,
                "full_text": mc.full_text,
            }
            for mc in result.matched_articles
        ],
        "related_articles": [
            {
                "law_name": rc.ref.law_name,
                "article_no": rc.ref.article_no,
                "article_title": rc.ref.article_title,
                "doc_id": rc.ref.doc_id,
                "key_phrases": rc.key_phrases,
                "full_text": rc.full_text,
            }
            for rc in result.related_articles
        ],
        "differences": result.differences,
        "verdict": result.verdict,
        "evidence": result.evidence,
        "citations": result.citations,
    }

    if result.quasi_chain:
        payload["quasi_provisions"] = {
            "law_name": result.quasi_chain.law_name,
            "entries": [
                {
                    "source_article_no": q.source_article_no,
                    "target_articles": q.target_articles,
                    "scope": q.scope,
                }
                for q in result.quasi_chain.quasi_provisions
            ],
            "omitted_articles": result.quasi_chain.omitted_articles,
            "included_articles": result.quasi_chain.included_articles,
        }

    if result.ox_quizzes:
        payload["ox_quizzes"] = [
            {
                "question": q.question,
                "answer": q.answer,
                "explanation": q.explanation,
                "related_articles": q.related_articles,
            }
            for q in result.ox_quizzes
        ]

    return payload
