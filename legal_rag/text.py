from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List


SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def normalize_text(value: str) -> str:
    value = value or ""
    value = value.replace("\u3000", " ")
    value = SPACE_RE.sub(" ", value)
    return value.strip()


def canonical_text(value: str) -> str:
    return NON_WORD_RE.sub("", normalize_text(value)).lower()


def tokenize(value: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(value).lower())


def char_ngrams(value: str, n: int = 3) -> List[str]:
    compact = canonical_text(value)
    if not compact:
        return []
    if len(compact) <= n:
        return [compact]
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]


def semantic_terms(value: str) -> List[str]:
    tokens = tokenize(value)
    grams = char_ngrams(value, n=3)
    return tokens + grams


def hash_embedding(
    value: str,
    *,
    dims: int = 768,
    idf: Dict[str, float] | None = None,
) -> Dict[int, float]:
    weights: Dict[int, float] = {}
    counts = Counter(semantic_terms(value))
    for term, count in counts.items():
        weight = float(count) * (idf.get(term, 1.0) if idf else 1.0)
        slot = hash(term) % dims
        weights[slot] = weights.get(slot, 0.0) + weight
    norm = math.sqrt(sum(score * score for score in weights.values()))
    if norm == 0.0:
        return {}
    return {slot: score / norm for slot, score in weights.items()}


def cosine_similarity(left: Dict[int, float], right: Dict[int, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(slot, 0.0) for slot, weight in left.items())


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered

