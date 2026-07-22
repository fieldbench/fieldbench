"""Drive an extractor over the corpus and write per-doc predictions.

The other half of the benchmark loop: `fieldbench run` produces the `<stem>.json`
prediction files that `fieldbench score` reads. Extractor-agnostic — supply any
Runner (turns document text + schema into a {field: value} dict). A schema-driven
LLM runner is provided; model-SDK wiring lives in examples/ so the core stays
dependency-light. Runs are resumable (existing prediction files are skipped) and
never invent results — a runner error writes nothing, so the doc is scored as
all-null (a real miss), not silently dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import yaml

from .corpus import _iter_docs, discover_categories, representation_path
from .scoring import is_empty


class Runner(Protocol):
    """Anything that turns a document + its schema into a {field: value} dict."""

    def extract(self, doc_text: str, schema: dict, stem: str) -> dict: ...


@dataclass
class RunStats:
    written: int = 0
    skipped: int = 0  # already present (resume)
    errors: int = 0
    no_representation: int = 0
    error_stems: list = field(default_factory=list)


def _schema_field_names(schema: dict) -> list[str]:
    return list((schema.get("fields") or {}).keys())


def build_extraction_prompt(doc_text: str, schema: dict) -> str:
    """Generic schema-driven extraction prompt. No domain assumptions."""
    lines = []
    for name, spec in (schema.get("fields") or {}).items():
        spec = spec if isinstance(spec, dict) else {}
        typ = spec.get("type", "string")
        desc = spec.get("description", "")
        opts = spec.get("options")
        suffix = f" (one of: {', '.join(map(str, opts))})" if isinstance(opts, list) and opts else ""
        lines.append(f"- {name} ({typ}): {desc}{suffix}")
    fields_block = "\n".join(lines)
    return (
        "Extract structured data from the document below.\n"
        "Return ONLY a JSON object with EXACTLY these fields. "
        "Use null when a field is not present in the document — do not guess.\n\n"
        f"Fields:\n{fields_block}\n\n"
        f"Document:\n{doc_text}\n\nJSON:"
    )


def parse_json_response(text: str) -> dict:
    """Robustly pull a JSON object out of an LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, depth, end = text.find("{"), 0, -1
        if start < 0:
            return {}
        for i in range(start, len(text)):
            depth += text[i] == "{"
            depth -= text[i] == "}"
            if depth == 0:
                end = i + 1
                break
        if end < 0:
            return {}
        try:
            obj = json.loads(text[start:end])
        except json.JSONDecodeError:
            return {}
    return obj if isinstance(obj, dict) else {}


def window_text(text: str, size: int, overlap: int = 2000) -> list[str]:
    """Split text into overlapping windows of `size` chars (overlap avoids
    splitting a value across a boundary)."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def merge_field_values(values: list) -> object:
    """Merge one field's values across windows: concat+dedup lists, first
    non-empty dict, else first non-empty scalar."""
    non_empty = [v for v in values if not is_empty(v)]
    if not non_empty:
        return None
    if any(isinstance(v, list) for v in non_empty):
        out, seen = [], set()
        for v in non_empty:
            for item in v if isinstance(v, list) else [v]:
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    seen.add(key)
                    out.append(item)
        return out
    for v in non_empty:
        if isinstance(v, dict):
            return v
    return non_empty[0]


class LLMRunner:
    """Schema-driven runner over any `complete(prompt) -> str` callable.

    Keeps model SDKs out of the core: pass a completion function (see
    examples/). Output is filtered to the schema's declared fields.

    `max_doc_chars` (None = off) makes it windowed: a document longer than the
    budget is split into overlapping windows, each extracted, then merged
    field-by-field — a fair long-document baseline that survives context limits.
    """

    def __init__(self, complete: Callable[[str], str], max_doc_chars: int | None = None, overlap: int = 2000):
        self._complete = complete
        self._max_doc_chars = max_doc_chars
        self._overlap = overlap

    def _extract_once(self, doc_text: str, schema: dict, allowed: set) -> dict:
        parsed = parse_json_response(self._complete(build_extraction_prompt(doc_text, schema)))
        return {k: v for k, v in parsed.items() if k in allowed} if allowed else parsed

    def extract(self, doc_text: str, schema: dict, stem: str) -> dict:
        allowed = set(_schema_field_names(schema))
        if self._max_doc_chars and len(doc_text) > self._max_doc_chars:
            windows = window_text(doc_text, self._max_doc_chars, self._overlap)
            preds = [self._extract_once(w, schema, allowed) for w in windows]
            names = allowed or {k for p in preds for k in p}
            return {name: merge_field_values([p.get(name) for p in preds]) for name in names}
        return self._extract_once(doc_text, schema, allowed)


def _load_schema(root: Path, schema_ref: str | None, cache: dict) -> dict:
    if not schema_ref:
        return {}
    if schema_ref in cache:
        return cache[schema_ref]
    try:
        schema = yaml.safe_load((root / schema_ref).read_text()) or {}
    except (OSError, yaml.YAMLError):
        schema = {}
    cache[schema_ref] = schema
    return schema


def run_corpus(
    corpus_root: Path,
    runner: Runner,
    out_dir: Path,
    mode: str = "markdown",
    category: str | None = None,
    resume: bool = True,
    limit: int | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> RunStats:
    """Run `runner` over the corpus, writing out_dir/<stem>.json per doc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = RunStats()
    schema_cache: dict = {}
    categories = [category] if category else discover_categories(corpus_root)
    done = 0

    for cat in categories:
        cat_dir = corpus_root / cat
        for stem, _expected, manifest in _iter_docs(corpus_root, cat):  # _expected is GT — NOT used at run time
            if limit is not None and done >= limit:
                return stats
            out_path = out_dir / f"{stem}.json"
            if resume and out_path.exists():
                stats.skipped += 1
                continue
            repr_path = representation_path(cat_dir, stem, manifest, mode)
            if repr_path is None or not repr_path.exists():
                stats.no_representation += 1
                if on_progress:
                    on_progress(stem, "no-representation")
                continue
            schema = _load_schema(corpus_root, manifest.get("schema"), schema_cache)
            try:
                prediction = runner.extract(repr_path.read_text(), schema, stem)
                out_path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2))
                stats.written += 1
                done += 1
                if on_progress:
                    on_progress(stem, "ok")
            except Exception as exc:  # noqa: BLE001 — a runner failure must not abort the whole run
                stats.errors += 1
                stats.error_stems.append(stem)
                if on_progress:
                    on_progress(stem, f"error: {exc}")

    return stats
