"""Loader tests: flat and source-first layouts must load + score identically."""

import json
from pathlib import Path

from fieldbench.aggregate import build_report
from fieldbench.corpus import list_documents, score_corpus


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _flat_corpus(root: Path) -> None:
    cat = root / "invoices"
    _write(cat / "documents" / "inv_1.md", "# invoice\nTotal 100")
    _write(cat / "expected" / "inv_1.expected.json", {"total": 100, "vendor": None})
    _write(cat / "manifests" / "inv_1.json", {"original_format": "Markdown (synthetic)", "schema": "x.yaml"})


def _sourcefirst_corpus(root: Path) -> None:
    cat = root / "invoices"
    _write(cat / "documents" / "inv_1" / "repr" / "markdown.md", "# invoice\nTotal 100")
    _write(cat / "expected" / "inv_1.expected.json", {"total": 100, "vendor": None})
    _write(
        cat / "documents" / "inv_1" / "meta.json",
        {
            "source_format": "pdf",
            "source": "real",
            "source_artifact": "source.pdf",
            "representations": {"markdown": {"path": "repr/markdown.md", "parser": "docling"}},
        },
    )
    _write(cat / "documents" / "inv_1" / "source.pdf", "%PDF-fake")


def test_flat_layout_loads_and_scores(tmp_path):
    _flat_corpus(tmp_path)
    results = tmp_path / "preds"
    _write(results / "inv_1.json", {"total": "100", "vendor": None})
    docs, missing = score_corpus(tmp_path, results)
    assert missing == 0
    rep = build_report(docs)
    assert rep.overall.fields == 2 and rep.overall.passed == 2  # total match + correct-absence
    assert rep.by_source["synthetic"].docs == 1


def test_sourcefirst_layout_loads_and_scores(tmp_path):
    _sourcefirst_corpus(tmp_path)
    results = tmp_path / "preds"
    _write(results / "inv_1.json", {"total": 100, "vendor": None})
    docs, missing = score_corpus(tmp_path, results)
    assert missing == 0
    rep = build_report(docs, mode="source")
    assert rep.overall.passed == 2
    assert rep.by_source["real"].docs == 1  # explicit source label honored
    assert rep.mode == "source"


def test_missing_prediction_counted_not_skipped(tmp_path):
    _flat_corpus(tmp_path)
    docs, missing = score_corpus(tmp_path, tmp_path / "empty")
    assert missing == 1
    rep = build_report(docs)
    assert rep.overall.fields == 2  # still scored (as all-null), not dropped


def test_schema_enum_mappings_fold(tmp_path):
    cat = tmp_path / "sec_filings"
    _write(cat / "documents" / "f1.md", "10K/A amendment")
    _write(cat / "expected" / "f1.expected.json", {"form_type": "10-K/A"})
    _write(cat / "manifests" / "f1.json", {"original_format": "html", "schema": "sec_filings/schemas/s.yaml"})
    _write(
        cat / "schemas" / "s.yaml",
        "fields:\n  form_type:\n    type: enum\n    mappings:\n      10-K/A:\n        - 10K/A\n        - 10-KA\n",
    )
    results = tmp_path / "preds"
    _write(results / "f1.json", {"form_type": "10K/A"})  # alias of the canonical GT
    docs, missing = score_corpus(tmp_path, results)
    assert missing == 0
    rep = build_report(docs)
    assert rep.overall.passed == 1  # folds via schema mappings -> match, not wrong_value


def test_representation_resolution(tmp_path):
    _sourcefirst_corpus(tmp_path)
    md = dict((stem, p) for stem, _c, p in list_documents(tmp_path, mode="markdown"))["inv_1"]
    assert md is not None and md.name == "markdown.md"
    src = dict((stem, p) for stem, _c, p in list_documents(tmp_path, mode="source"))["inv_1"]
    assert src is not None and src.name == "source.pdf"
