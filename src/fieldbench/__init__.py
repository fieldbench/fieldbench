"""FieldBench — a cross-domain, field-level benchmark for schema-driven document extraction."""

from .aggregate import DocResult, Report, build_report
from .corpus import list_documents, score_corpus
from .run import LLMRunner, Runner, run_corpus
from .scoring import SCORER_VERSION, FieldResult, compare_field

__version__ = "0.2.1"

__all__ = [
    "SCORER_VERSION",
    "DocResult",
    "FieldResult",
    "LLMRunner",
    "Report",
    "Runner",
    "__version__",
    "build_report",
    "compare_field",
    "list_documents",
    "run_corpus",
    "score_corpus",
]
