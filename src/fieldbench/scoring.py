"""Canonical field-level comparison for the FieldBench benchmark.

This is the single source of truth for how an extracted value is compared to
ground truth. It ports the semantics of Koji's ``cli/test_runner.py`` scorer
(binary pass/fail + four-way null semantics, array F1, enum/mapping folding,
punctuation-tolerant scalars) and makes the four-way outcome an explicit,
first-class tag on every field so the aggregate can report a hallucination /
miss / correct-absence breakdown, not just accuracy.

Fully generic: no field names, document types, or domain knowledge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

Bucket = Literal["correct_absence", "hallucination", "miss", "match", "wrong_value"]

SCORER_VERSION = "0.1.0"


# ── Normalization helpers ─────────────────────────────────────────────


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def string_similarity(a: str, b: str) -> float:
    """Levenshtein ratio in [0,1], case-insensitive after trimming."""
    a = a.strip().lower()
    b = b.strip().lower()
    if a == b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max_len


def normalize_date(value: Any) -> str | None:
    """Normalize to YYYY-MM-DD, or None if not a recognizable date."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    m = re.match(r"^(\d{1,2})[.\-](\d{1,2})[.\-](\d{4})$", s)
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
    return None


def to_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def is_empty(value: Any) -> bool:
    """None, empty string, empty list, empty dict → 'no value'."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _normalize_value(value: Any) -> Any:
    if is_empty(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        d = normalize_date(value)
        if d is not None:
            return d
        n = to_number(value)
        if n is not None:
            return n
        return value.strip().lower()
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            if isinstance(k, str) and k.startswith("__"):  # provenance keys
                continue
            nv = _normalize_value(v)
            if nv is not None:
                out[k] = nv
        return out or None
    if isinstance(value, list):
        norm = [_normalize_value(v) for v in value]
        norm = [v for v in norm if v is not None]
        if not norm:
            return None
        if all(isinstance(v, dict) for v in norm):
            norm.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return norm
    return value


def _normalize_for_set(items: list) -> list[str]:
    keys = []
    for item in items:
        n = _normalize_value(item)
        keys.append(json.dumps(n, sort_keys=True) if isinstance(n, dict) else json.dumps(n))
    return keys


def _dict_key_overlap(a: dict, b: dict) -> float:
    na = _normalize_value(a)
    nb = _normalize_value(b)
    if not isinstance(na, dict) or not isinstance(nb, dict):
        return 1.0 if json.dumps(na, sort_keys=True, default=str) == json.dumps(nb, sort_keys=True, default=str) else 0.0
    all_keys = set(na) | set(nb)
    if not all_keys:
        return 1.0
    matches = sum(
        1
        for k in all_keys
        if json.dumps(na.get(k), sort_keys=True, default=str) == json.dumps(nb.get(k), sort_keys=True, default=str)
    )
    return matches / len(all_keys)


def _array_of_dicts_similarity(expected: list, actual: list) -> float:
    if not expected:
        return 1.0 if not actual else 0.0
    remaining = list(range(len(actual)))
    total = 0.0
    for exp_item in expected:
        best, best_idx = 0.0, -1
        for j in remaining:
            sim = _dict_key_overlap(exp_item, actual[j])
            if sim > best:
                best, best_idx = sim, j
        total += best
        if best_idx >= 0:
            remaining.remove(best_idx)
    return total / len(expected)


def array_f1(expected: list, actual: list) -> float:
    """Element-wise F1 in [0,1] (quality-weighted precision/recall)."""
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    if all(isinstance(v, dict) for v in expected) and all(isinstance(v, dict) for v in actual):
        recall = _array_of_dicts_similarity(expected, actual)
        precision = _array_of_dicts_similarity(actual, expected)
    else:
        exp_c = Counter(_normalize_for_set(expected))
        act_c = Counter(_normalize_for_set(actual))
        matched = sum((exp_c & act_c).values())
        recall = matched / len(expected)
        precision = matched / len(actual)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def resolve_mapping(value: str, mappings: dict) -> str:
    """Fold a value to its canonical form via schema-declared enum aliases."""

    def fold(s: object) -> str:
        return re.sub(r"\s+", " ", str(s).strip().lower())

    v = fold(value)
    for canonical, aliases in mappings.items():
        if v == fold(canonical):
            return canonical
        if isinstance(aliases, list) and any(v == fold(a) for a in aliases):
            return canonical
    return value


# ── Result ────────────────────────────────────────────────────────────


@dataclass
class FieldResult:
    field_name: str
    passed: bool
    bucket: Bucket
    expected: Any = None
    actual: Any = None
    detail: str = ""
    score: float | None = None  # explicit partial credit; None → derive from `passed`

    @property
    def weighted_score(self) -> float:
        if self.score is not None:
            return self.score
        return 1.0 if self.passed else 0.0


# ── Core comparison ───────────────────────────────────────────────────


def compare_field(
    field_name: str,
    expected: Any,
    actual: Any,
    fuzzy_threshold: float = 0.0,
    mappings: dict | None = None,
) -> FieldResult:
    """Compare one expected value against one extracted value.

    Four-way null semantics run first, in order:
      1. both empty            → correct_absence (PASS)
      2. expected empty only   → hallucination   (FAIL)
      3. actual empty only     → miss            (FAIL)
      4. both present          → type-aware compare → match / wrong_value
    """
    exp_empty, act_empty = is_empty(expected), is_empty(actual)

    if exp_empty and act_empty:
        return FieldResult(field_name, True, "correct_absence", expected, actual, "correctly absent")
    if exp_empty and not act_empty:
        return FieldResult(field_name, False, "hallucination", expected, actual, f"hallucinated: expected null, got {actual!r}")
    if act_empty:
        return FieldResult(field_name, False, "miss", expected, actual, "missing from actual")

    def matched(detail: str = "", score: float | None = None) -> FieldResult:
        return FieldResult(field_name, True, "match", expected, actual, detail, score)

    def wrong(detail: str, score: float | None = None) -> FieldResult:
        return FieldResult(field_name, False, "wrong_value", expected, actual, detail, score)

    # Date
    ed, ad = normalize_date(expected), normalize_date(actual)
    if ed is not None and ad is not None:
        return matched() if ed == ad else wrong(f"expected {ed}, got {ad}")

    # Number (absolute tolerance 0.01 — cent-exact by design)
    en, an = to_number(expected), to_number(actual)
    if en is not None and an is not None:
        return matched() if round(abs(en - an), 10) <= 0.01 else wrong(f"expected {expected}, got {actual}")

    # Array (element-wise F1; strict pass = exact-set, else fuzzy_threshold)
    if isinstance(expected, list) and isinstance(actual, list):
        f1 = array_f1(expected, actual)
        exp_keys = sorted(_normalize_for_set(expected))
        act_keys = sorted(_normalize_for_set(actual))
        if exp_keys == act_keys:
            return matched(score=1.0)
        if fuzzy_threshold > 0 and expected:
            matched_n = sum(1 for k in exp_keys if k in act_keys)
            ratio = matched_n / len(exp_keys)
            if ratio >= fuzzy_threshold:
                return matched(f"fuzzy array match ({matched_n}/{len(exp_keys)})", score=f1)
            if (
                len(expected) == len(actual)
                and all(isinstance(v, dict) for v in expected)
                and all(isinstance(v, dict) for v in actual)
                and _array_of_dicts_similarity(expected, actual) >= fuzzy_threshold
            ):
                return matched("fuzzy structural match", score=f1)
        return wrong(f"array items differ ({f1:.0%} element F1)", score=f1)

    # Scalar / string
    if isinstance(expected, str) and isinstance(actual, str):
        e, a = expected, actual
        if mappings:
            e, a = resolve_mapping(e, mappings), resolve_mapping(a, mappings)
        if e.strip().lower() == a.strip().lower():
            return matched()
        ep = re.sub(r"[^a-z0-9]+", " ", e.lower()).strip()
        ea = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
        if ep and ep == ea:
            return matched()  # punctuation-only difference
        if fuzzy_threshold > 0 and string_similarity(e, a) >= fuzzy_threshold:
            return matched(f"fuzzy match ({string_similarity(e, a):.0%})")
        return wrong(f"expected {expected!r}, got {actual!r}")

    # Type mismatch or other → strict equality fallback
    if _normalize_value(expected) == _normalize_value(actual):
        return matched()
    return wrong(f"expected {expected!r}, got {actual!r}")
