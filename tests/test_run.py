"""Run harness: predictions round-trip through run_corpus -> score_corpus."""

import json
from pathlib import Path

from fieldbench.aggregate import build_report
from fieldbench.corpus import score_corpus
from fieldbench.run import LLMRunner, build_extraction_prompt, parse_json_response, run_corpus


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)


def _corpus(root: Path) -> None:
    cat = root / "sec_filings"
    _write(cat / "documents" / "d1.md", "Acme Corp filed a 10-K on 2024-03-15.")
    _write(cat / "expected" / "d1.expected.json", {"filer_name": "Acme Corp", "form_type": "10-K", "unrelated": None})
    _write(cat / "manifests" / "d1.json", {"original_format": "html", "schema": "sec_filings/schemas/s.yaml"})
    _write(
        cat / "schemas" / "s.yaml",
        "fields:\n  filer_name:\n    type: string\n  form_type:\n    type: enum\n  unrelated:\n    type: string\n",
    )


class StubRunner:
    """Deterministic runner: returns the right answers (+ a hallucinated extra)."""

    def extract(self, doc_text: str, schema: dict, stem: str) -> dict:
        return {"filer_name": "Acme Corp", "form_type": "10-K", "unrelated": "invented", "not_in_schema": "x"}


def test_run_then_score_roundtrip(tmp_path):
    _corpus(tmp_path)
    out = tmp_path / "preds"
    stats = run_corpus(tmp_path, StubRunner(), out)
    assert stats.written == 1 and stats.errors == 0
    pred = json.loads((out / "d1.json").read_text())
    assert "not_in_schema" in pred  # run_corpus writes raw runner output; only LLMRunner filters

    docs, missing = score_corpus(tmp_path, out)
    assert missing == 0
    rep = build_report(docs)
    # filer_name match + form_type match + unrelated hallucination (expected null, got "invented")
    assert rep.four_way["match"] == 2
    assert rep.four_way["hallucination"] == 1


def test_run_resume_skips_existing(tmp_path):
    _corpus(tmp_path)
    out = tmp_path / "preds"
    run_corpus(tmp_path, StubRunner(), out)
    stats2 = run_corpus(tmp_path, StubRunner(), out)  # second pass
    assert stats2.written == 0 and stats2.skipped == 1


def test_runner_error_writes_nothing(tmp_path):
    _corpus(tmp_path)

    class Boom:
        def extract(self, *a):
            raise RuntimeError("model down")

    out = tmp_path / "preds"
    stats = run_corpus(tmp_path, Boom(), out)
    assert stats.errors == 1 and stats.written == 0
    assert not (out / "d1.json").exists()  # no invented result
    # scored as all-null (a real miss), not dropped
    docs, missing = score_corpus(tmp_path, out)
    assert missing == 1


def test_llm_runner_filters_and_parses():
    schema = {"fields": {"a": {"type": "string"}, "b": {"type": "string"}}}
    runner = LLMRunner(lambda prompt: '```json\n{"a": "1", "b": "2", "c": "junk"}\n```')
    out = runner.extract("doc", schema, "s")
    assert out == {"a": "1", "b": "2"}  # 'c' filtered out, fences stripped


def test_parse_json_embedded():
    assert parse_json_response('here you go: {"x": 1} thanks') == {"x": 1}
    assert parse_json_response("no json here") == {}


def test_prompt_lists_fields():
    schema = {"fields": {"total": {"type": "number", "description": "the total"}}}
    p = build_extraction_prompt("doc text", schema)
    assert "total (number): the total" in p and "doc text" in p


# ── Windowed (long-doc) runner ───────────────────────────────────────

def test_window_text_splits_with_overlap():
    from fieldbench.run import window_text
    assert window_text("abcdefghij", 100) == ["abcdefghij"]  # under size
    ws = window_text("x" * 250, 100, overlap=20)
    assert len(ws) >= 3 and all(len(w) <= 100 for w in ws)


def test_merge_field_values():
    from fieldbench.run import merge_field_values
    assert merge_field_values([None, "Acme", None]) == "Acme"      # first non-empty scalar
    assert merge_field_values([["a"], None, ["b", "a"]]) == ["a", "b"]  # concat + dedup
    assert merge_field_values([None, None]) is None


def test_windowed_runner_merges_across_windows():
    # window 1 sees field a, window 2 sees field b; merged has both.
    schema = {"fields": {"a": {}, "b": {}}}
    calls = {"n": 0}

    def complete(prompt: str) -> str:
        calls["n"] += 1
        return '{"a": "A", "b": null}' if calls["n"] == 1 else '{"a": null, "b": "B"}'

    from fieldbench.run import LLMRunner
    runner = LLMRunner(complete, max_doc_chars=100, overlap=0)
    out = runner.extract("y" * 250, schema, "s")  # 250 chars -> 3 windows
    assert out["a"] == "A" and out["b"] == "B"
    assert calls["n"] >= 3  # windowed, not a single call
