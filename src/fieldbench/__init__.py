"""FieldBench — a cross-domain, field-level benchmark for schema-driven document extraction."""

from .aggregate import DocResult, Report, build_report
from .corpus import score_corpus
from .scoring import SCORER_VERSION, FieldResult, compare_field

__version__ = "0.1.0"

__all__ = [
    "compare_field",
    "FieldResult",
    "SCORER_VERSION",
    "score_corpus",
    "build_report",
    "DocResult",
    "Report",
    "__version__",
]
