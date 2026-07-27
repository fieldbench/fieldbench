# FieldBench

A cross-domain, field-level benchmark for **schema-driven document extraction** — and the canonical scorer that goes with it.

Every document-extraction vendor claims 95%+ accuracy; almost none publish how they measure it. FieldBench makes extraction accuracy **falsifiable and comparable**: a shared corpus, a shared type-aware scorer, and a leaderboard anyone can submit to.

> **Status: v0.2.2, published.** `pip install fieldbench` installs the scorer and the run harness. The corpus (1,442 documents across 10 categories) lives at [`fieldbench/corpus`](https://github.com/fieldbench/corpus), mirrored on [HuggingFace](https://huggingface.co/datasets/fieldbench/corpus) and archived on [Zenodo](https://doi.org/10.5281/zenodo.21532677). A public leaderboard is forthcoming.

## Install

```bash
pip install fieldbench
```

## Run a baseline

`fieldbench run` drives any extractor over the corpus and writes the prediction files for you. It's extractor-agnostic — the whole integration surface is a `complete(prompt) -> str` callable (see `examples/openai_runner.py`):

```bash
pip install "fieldbench[openai]"
OPENAI_API_KEY=... FIELDBENCH_MODEL=gpt-4o-mini \
  fieldbench run --corpus ./corpus --out ./preds/gpt-4o-mini \
    --runner examples.openai_runner:make_runner
```

Runs are **resumable** (existing predictions are skipped) and **never invent results** — a runner error writes nothing, so that document is scored as a real miss, not dropped.

## Score your system

Score prediction files — a flat `{field: value}` JSON named `<doc_id>.json` per document (produced by `fieldbench run` or by your own pipeline):

```bash
fieldbench score --corpus /path/to/corpus --results /path/to/predictions/
```

You get overall accuracy, the **all-null floor** (what an empty extractor scores for free), a **real-vs-synthetic** split, a **five-outcome breakdown**, and a per-category table. `--json` emits the full report; `--category <name>` scopes to one category.

## What makes the scoring type-aware

`compare_field` is **null-aware** — it resolves the null/value boundary first, into five distinct outcomes plain accuracy hides:

| Outcome | Meaning |
|---|---|
| `correct_absence` | field absent in GT, system correctly returned null |
| `hallucination` | field absent in GT, system invented a value |
| `miss` | field present in GT, system returned null |
| `match` / `wrong_value` | both present → type-aware comparison |

For present-vs-present it is tolerant where it should be and strict where it must be: numeric with cent-exact tolerance, date normalization, order-independent array F1 (partial credit), punctuation-insensitive strings, and schema enum-alias folding — but never masking a genuine content difference.

## How to cite

```bibtex
@misc{fieldbench,
  title  = {FieldBench: A Cross-Domain Benchmark for Schema-Driven Document Extraction},
  author = {Thomas, Frank},
  year   = {2026},
  doi    = {10.5281/zenodo.21532677},
  url    = {https://github.com/fieldbench}
}
```
<!-- The paper citation will be added here once the accompanying paper is published. -->

## Development

```bash
pip install -e ".[dev]"
pytest        # golden scorer tests
ruff check .
```

The golden tests in `tests/` are the anti-drift contract: they pin the canonical scoring semantics that any conforming implementation must match.

## License

Code: MIT. The corpus is licensed per-source — see [`fieldbench/corpus`](https://github.com/fieldbench/corpus).
