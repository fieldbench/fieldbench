"""Aggregate per-field results into a FieldBench report.

Reports micro/macro field accuracy, always stratified real vs synthetic, with a
four-way confusion breakdown (correct-absence / hallucination / miss /
wrong-value) and the all-null degenerate floor — so every score is read against
what an empty extractor scores for free.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .scoring import SCORER_VERSION, FieldResult, is_empty


@dataclass
class DocResult:
    stem: str
    category: str
    source: str  # "real" | "synthetic" | "unknown"
    fields: list[FieldResult]


@dataclass
class Stratum:
    docs: int = 0
    fields: int = 0
    passed: int = 0
    weighted: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.passed / self.fields if self.fields else 0.0

    @property
    def partial(self) -> float:
        return self.weighted / self.fields if self.fields else 0.0

    def to_dict(self) -> dict:
        return {
            "docs": self.docs,
            "fields": self.fields,
            "passed": self.passed,
            "accuracy": round(self.accuracy, 4),
            "partial_credit": round(self.partial, 4),
        }


@dataclass
class Report:
    scorer_version: str = SCORER_VERSION
    overall: Stratum = field(default_factory=Stratum)
    by_source: dict[str, Stratum] = field(default_factory=dict)
    by_category: dict[str, Stratum] = field(default_factory=dict)
    four_way: Counter = field(default_factory=Counter)
    all_null_floor: float = 0.0

    def to_dict(self) -> dict:
        total = sum(self.four_way.values()) or 1
        return {
            "scorer_version": self.scorer_version,
            "overall": self.overall.to_dict(),
            "by_source": {k: v.to_dict() for k, v in sorted(self.by_source.items())},
            "by_category": {k: v.to_dict() for k, v in sorted(self.by_category.items())},
            "four_way": {
                bucket: {"count": n, "rate": round(n / total, 4)}
                for bucket, n in sorted(self.four_way.items())
            },
            "all_null_floor": round(self.all_null_floor, 4),
        }


def _add(strata: dict[str, Stratum], key: str, r: FieldResult) -> None:
    s = strata.setdefault(key, Stratum())
    s.fields += 1
    s.passed += int(r.passed)
    s.weighted += r.weighted_score


def build_report(docs: list[DocResult]) -> Report:
    rep = Report()
    empty_expected = 0
    total = 0
    doc_counts_source: dict[str, int] = defaultdict(int)
    doc_counts_cat: dict[str, int] = defaultdict(int)

    for doc in docs:
        doc_counts_source[doc.source] += 1
        doc_counts_cat[doc.category] += 1
        for r in doc.fields:
            total += 1
            rep.overall.fields += 1
            rep.overall.passed += int(r.passed)
            rep.overall.weighted += r.weighted_score
            rep.four_way[r.bucket] += 1
            _add(rep.by_source, doc.source, r)
            _add(rep.by_category, doc.category, r)
            if is_empty(r.expected):
                empty_expected += 1

    rep.overall.docs = len(docs)
    for src, n in doc_counts_source.items():
        rep.by_source.setdefault(src, Stratum()).docs = n
    for cat, n in doc_counts_cat.items():
        rep.by_category.setdefault(cat, Stratum()).docs = n

    # All-null floor: an empty extractor scores exactly the empty-GT fields
    # (they land in `correct_absence`); everything else becomes a miss.
    rep.all_null_floor = empty_expected / total if total else 0.0
    return rep
