"""FieldBench CLI: score prediction files against the corpus."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .aggregate import build_report
from .corpus import score_corpus
from .run import run_corpus


def _load_runner(ref: str):
    """Load a runner from `module:attr`. attr is a Runner instance or a
    zero-arg factory that returns one."""
    if ":" not in ref:
        raise ValueError(f"--runner must be 'module:attr', got {ref!r}")
    mod_name, attr = ref.split(":", 1)
    obj = getattr(importlib.import_module(mod_name), attr)
    if hasattr(obj, "extract"):
        return obj
    return obj()  # factory


def _print_table(report_dict: dict) -> None:
    o = report_dict["overall"]
    print(f"\nFieldBench  (scorer {report_dict['scorer_version']})")
    print("=" * 52)
    print(f"Overall:  {o['accuracy']:.1%}  ({o['passed']}/{o['fields']} fields, {o['docs']} docs)")
    print(f"All-null floor: {report_dict['all_null_floor']:.1%}  <- a `{{}}`-emitting baseline scores this free")

    print("\nBy source:")
    for src, s in report_dict["by_source"].items():
        print(f"  {src:<12} {s['accuracy']:.1%}  ({s['fields']} fields, {s['docs']} docs)")

    print("\nFour-way outcomes:")
    for bucket, v in report_dict["four_way"].items():
        print(f"  {bucket:<16} {v['count']:>6}  ({v['rate']:.1%})")

    print("\nBy category:")
    for cat, s in sorted(report_dict["by_category"].items(), key=lambda kv: kv[1]["accuracy"]):
        print(f"  {cat:<24} {s['accuracy']:.1%}  ({s['fields']} fields)")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fieldbench")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="Score prediction files against the corpus")
    score.add_argument("--corpus", required=True, type=Path, help="Path to the corpus root")
    score.add_argument("--results", required=True, type=Path, help="Directory of <stem>.json predictions")
    score.add_argument("--category", default=None, help="Score a single category only")
    score.add_argument(
        "--mode", default="unspecified", help="Label for which representation the predictions came from (markdown/source/...)"
    )
    score.add_argument("--fuzzy-threshold", type=float, default=0.0, help="Off (0.0) for the official metric")
    score.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    score.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail when prediction files are missing (they are scored as all-null)",
    )

    run = sub.add_parser("run", help="Run an extractor over the corpus, writing per-doc predictions")
    run.add_argument("--corpus", required=True, type=Path, help="Path to the corpus root")
    run.add_argument("--out", required=True, type=Path, help="Output dir for <stem>.json predictions")
    run.add_argument("--runner", required=True, help="Runner as 'module:attr' (instance or zero-arg factory)")
    run.add_argument("--mode", default="markdown", help="Which representation to feed (markdown/text/source)")
    run.add_argument("--category", default=None, help="Run a single category only")
    run.add_argument("--limit", type=int, default=None, help="Stop after N docs (smoke testing)")
    run.add_argument("--no-resume", action="store_true", help="Re-run docs even if a prediction file exists")

    args = parser.parse_args(argv)

    if args.command == "run":
        if not args.corpus.is_dir():
            print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
            return 2
        try:
            runner = _load_runner(args.runner)
        except (ValueError, ImportError, AttributeError) as exc:
            print(f"error: could not load runner {args.runner!r}: {exc}", file=sys.stderr)
            return 2
        stats = run_corpus(
            args.corpus,
            runner,
            args.out,
            mode=args.mode,
            category=args.category,
            resume=not args.no_resume,
            limit=args.limit,
            on_progress=lambda stem, status: print(f"  {stem}: {status}", file=sys.stderr),
        )
        print(
            f"\nwrote {stats.written}, skipped {stats.skipped} (resume), "
            f"errors {stats.errors}, no-representation {stats.no_representation}"
        )
        if stats.error_stems:
            print(f"error docs: {', '.join(stats.error_stems[:10])}"
                  f"{'…' if len(stats.error_stems) > 10 else ''}", file=sys.stderr)
        return 1 if (stats.errors or stats.no_representation) else 0

    if args.command == "score":
        if not args.corpus.is_dir():
            print(f"error: corpus not found: {args.corpus}", file=sys.stderr)
            return 2
        docs, missing = score_corpus(
            args.corpus, args.results, category=args.category, fuzzy_threshold=args.fuzzy_threshold
        )
        if not docs:
            print("error: no scorable documents found", file=sys.stderr)
            return 2
        report = build_report(docs, mode=args.mode).to_dict()
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_table(report)
        if missing and not args.allow_missing:
            print(
                f"error: {missing} prediction file(s) missing — scored as all-null. "
                f"Re-run with complete results, or pass --allow-missing to override.",
                file=sys.stderr,
            )
            return 1
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
