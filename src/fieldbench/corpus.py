"""Load the FieldBench corpus and score prediction files against it.

Supports two on-disk layouts so scoring keeps working across the migration to
the source-first structure (see corpus-structure.md):

  Flat (current):
    <category>/documents/<stem>.md
    <category>/expected/<stem>.expected.json
    <category>/manifests/<stem>.json

  Source-first (target):
    <category>/documents/<stem>/meta.json
    <category>/documents/<stem>/repr/markdown.md   (+ other representations)
    <category>/documents/<stem>/source.<ext>
    <category>/expected/<stem>.expected.json        (GT unchanged, either layout)

Predictions: a directory of `<stem>.json` files, each a flat {field: value} map
(the zero-dependency format any system can emit). A missing prediction file is
scored as all-null, never silently skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .aggregate import DocResult
from .scoring import compare_field


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_manifest(cat_dir: Path, stem: str) -> dict | None:
    """Dual-path manifest lookup: per-doc meta.json (new) → manifests/ (flat)."""
    meta = cat_dir / "documents" / stem / "meta.json"
    if meta.exists():
        return _read_json(meta)
    return _read_json(cat_dir / "manifests" / f"{stem}.json")


def _source_of(manifest: dict) -> str:
    """real | synthetic | unknown. Explicit `source` wins; else derive from format."""
    src = str(manifest.get("source", "")).strip().lower()
    if src in ("real", "synthetic"):
        return src
    fmt = str(manifest.get("source_format") or manifest.get("original_format", "")).lower()
    if not fmt:
        return "unknown"
    return "synthetic" if "synth" in fmt else "real"


def discover_categories(root: Path) -> list[str]:
    cats = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith((".", "_")):
            continue
        if (entry / "documents").is_dir() and (entry / "expected").is_dir():
            cats.append(entry.name)
    return cats


def _schema_mappings(root: Path, schema_ref: str | None, cache: dict) -> dict:
    """{field_name: mappings} from a schema YAML (`fields.<name>.mappings`).

    Enum-alias folding is what lets `10K/A` match GT `10-K/A`. Best-effort:
    a missing/unreadable schema yields no mappings, never an error.
    """
    if not schema_ref:
        return {}
    if schema_ref in cache:
        return cache[schema_ref]
    out: dict = {}
    path = root / schema_ref
    try:
        schema = yaml.safe_load(path.read_text())
        for name, spec in (schema.get("fields") or {}).items():
            if isinstance(spec, dict) and isinstance(spec.get("mappings"), dict):
                out[name] = spec["mappings"]
    except (OSError, yaml.YAMLError, AttributeError):
        out = {}
    cache[schema_ref] = out
    return out


def _iter_docs(root: Path, category: str):
    cat = root / category
    for expected_path in sorted((cat / "expected").glob("*.expected.json")):
        stem = expected_path.name[: -len(".expected.json")]
        expected = _read_json(expected_path)
        manifest = _load_manifest(cat, stem)
        if expected is None or manifest is None:
            continue
        yield stem, expected, manifest


def representation_path(cat_dir: Path, stem: str, manifest: dict, mode: str) -> Path | None:
    """Resolve the file a prediction generator should read for `mode`.

    New layout: manifest.representations[mode].path (or source.* for mode='source').
    Flat layout: documents/<stem>.md for mode in {markdown, text}.
    """
    doc_dir = cat_dir / "documents" / stem
    if mode == "source":
        artifact = manifest.get("source_artifact")
        return (doc_dir / artifact) if artifact else None
    reps = manifest.get("representations")
    if isinstance(reps, dict) and mode in reps and isinstance(reps[mode], dict):
        p = doc_dir / reps[mode]["path"]
        return p if p.exists() else None
    flat = cat_dir / "documents" / f"{stem}.md"  # flat layout has only markdown
    return flat if (mode in ("markdown", "text") and flat.exists()) else None


def list_documents(corpus_root: Path, mode: str = "markdown", category: str | None = None):
    """Yield (stem, category, representation_path) for baseline/prediction runners."""
    categories = [category] if category else discover_categories(corpus_root)
    for cat in categories:
        for stem, _expected, manifest in _iter_docs(corpus_root, cat):
            yield stem, cat, representation_path(corpus_root / cat, stem, manifest, mode)


def _load_prediction(results_dir: Path, stem: str) -> dict:
    data = _read_json(results_dir / f"{stem}.json")
    if isinstance(data, dict) and isinstance(data.get("fields"), dict):
        return data["fields"]  # accept a {"fields": {...}} wrapper
    return data if isinstance(data, dict) else {}


def score_corpus(
    corpus_root: Path,
    results_dir: Path,
    category: str | None = None,
    fuzzy_threshold: float = 0.0,
) -> tuple[list[DocResult], int]:
    """Score every document. Returns (doc_results, missing_prediction_count).

    Assert `missing_prediction_count == 0` before publishing any number — a
    missing file is scored as all-null, not skipped, so it can't silently
    inflate a system's accuracy.
    """
    categories = [category] if category else discover_categories(corpus_root)
    docs: list[DocResult] = []
    missing = 0
    schema_cache: dict = {}

    for cat in categories:
        for stem, expected, manifest in _iter_docs(corpus_root, cat):
            if not (results_dir / f"{stem}.json").exists():
                missing += 1
            prediction = _load_prediction(results_dir, stem)
            mappings = _schema_mappings(corpus_root, manifest.get("schema"), schema_cache)
            fields = [
                compare_field(
                    name, exp, prediction.get(name), fuzzy_threshold=fuzzy_threshold, mappings=mappings.get(name)
                )
                for name, exp in expected.items()
            ]
            docs.append(DocResult(stem, cat, _source_of(manifest), fields))

    return docs, missing
