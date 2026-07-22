"""FieldBench — a cross-domain, field-level benchmark for schema-driven document extraction."""

from .aggregate import DocResult, Report, build_report
from .corpus import list_documents, score_corpus
from .run import LLMRunner, Runner, run_corpus
from .scoring import SCORER_VERSION, FieldResult, compare_field

__version__ = "0.1.0"

__all__ = [
    "compare_field",
    "FieldResult",
    "SCORER_VERSION",
    "score_corpus",
    "list_documents",
    "build_report",
    "DocResult",
    "Report",
    "run_corpus",
    "Runner",
    "LLMRunner",
    "__version__",
]
