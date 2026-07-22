"""Load the FieldBench corpus and score prediction files against it.

Corpus layout (per category directory):
    <category>/documents/<stem>.md
    <category>/expected/<stem>.expected.json     # flat {field: value}
    <category>/manifests/<stem>.json             # metadata (original_format, schema)

Predictions: a directory of `<stem>.json` files, each a flat {field: value}
map (no wrapper) — the same zero-dependency format any system can emit.
Missing prediction files are scored as all-null (never silently skipped).
"""

from __future__ import annotations

import json
from pathlib import Path

from .aggregate import DocResult
from .scoring import compare_field

_REQUIRED = ("documents", "expected", "manifests")


def _source_of(manifest: dict) -> str:
    fmt = str(manifest.get("original_format", "")).lower()
    if not fmt:
        return "unknown"
    return "synthetic" if "synth" in fmt else "real"


def discover_categories(root: Path) -> list[str]:
    cats = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if all((entry / sub).is_dir() for sub in _REQUIRED):
            cats.append(entry.name)
    return cats


def _iter_docs(root: Path, category: str):
    cat = root / category
    for expected_path in sorted((cat / "expected").glob("*.expected.json")):
        stem = expected_path.name[: -len(".expected.json")]
        manifest_path = cat / "manifests" / f"{stem}.json"
        if not manifest_path.exists():
            continue
        try:
            expected = json.loads(expected_path.read_text())
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        yield stem, expected, manifest


def _load_prediction(results_dir: Path, stem: str) -> dict:
    p = results_dir / f"{stem}.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}
    # Accept either a flat map or a {"fields": {...}} wrapper.
    if isinstance(data, dict) and "fields" in data and isinstance(data["fields"], dict):
        return data["fields"]
    return data if isinstance(data, dict) else {}


def score_corpus(
    corpus_root: Path,
    results_dir: Path,
    category: str | None = None,
    fuzzy_threshold: float = 0.0,
) -> tuple[list[DocResult], int]:
    """Score every document. Returns (doc_results, missing_prediction_count).

    `missing_prediction_count` MUST be asserted == 0 before publishing any
    number — a missing file is scored as all-null, not skipped, so it can't
    silently inflate a system's accuracy.
    """
    categories = [category] if category else discover_categories(corpus_root)
    docs: list[DocResult] = []
    missing = 0

    for cat in categories:
        for stem, expected, manifest in _iter_docs(corpus_root, cat):
            pred_path = results_dir / f"{stem}.json"
            if not pred_path.exists():
                missing += 1
            prediction = _load_prediction(results_dir, stem)
            fields = [
                compare_field(name, exp, prediction.get(name), fuzzy_threshold=fuzzy_threshold)
                for name, exp in expected.items()
            ]
            docs.append(DocResult(stem, cat, _source_of(manifest), fields))

    return docs, missing
