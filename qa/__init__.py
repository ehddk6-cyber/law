from __future__ import annotations

from qa.legal_analysis import (
    AnalysisResult,
    ArticleRef,
    OxQuiz,
    QuasiProvisionChain,
    QuasiProvisionRef,
    analyze_law_query,
    build_quasi_provision_chain,
    detect_erroneous_omission,
    format_analysis_json,
    format_analysis_text,
    generate_ox_quiz,
    generate_ox_quiz_from_text,
)
from qa.retrievers import (
    AcrRetriever,
    DeccRetriever,
    DetcRetriever,
    ExpcRetriever,
    LawRetriever,
    PrecRetriever,
    SearchResult,
)
from qa.router import RouteDecision, route_query
from qa.unified import UnifiedResult, UnifiedSearcher

__all__ = [
    "ArticleRef",
    "QuasiProvisionRef",
    "QuasiProvisionChain",
    "OxQuiz",
    "AnalysisResult",
    "SearchResult",
    "RouteDecision",
    "UnifiedResult",
    "AcrRetriever",
    "LawRetriever",
    "PrecRetriever",
    "DeccRetriever",
    "DetcRetriever",
    "ExpcRetriever",
    "UnifiedSearcher",
    "route_query",
    "analyze_law_query",
    "build_quasi_provision_chain",
    "detect_erroneous_omission",
    "generate_ox_quiz",
    "generate_ox_quiz_from_text",
    "format_analysis_json",
    "format_analysis_text",
]
