#!/usr/bin/env python3
"""Parity check: fieldbench scorer vs Koji scorer B (cli/test_runner.py).

FieldBench is the canonical scorer. This harness runs a shared case set through
both and reports every divergence in (passed, weighted_score) — the actionable
list for making Koji conform (or for catching an unfaithful port in fieldbench).

Not a CI test (CI has no Koji checkout). Run manually:

    python scripts/parity_koji.py --koji /path/to/koji/cli/test_runner.py

Exit 0 = full parity, 1 = divergences found.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from fieldbench.scoring import compare_field as fb_compare

# (expected, actual, kwargs, label) — mirrors the golden set + edge cases.
CASES = [
    # four-way null
    (None, None, {}, "both-null"),
    ("", [], {}, "both-empty-mixed"),
    (None, "Acme", {}, "hallucination"),
    ("", ["x"], {}, "hallucination-array"),
    ("Acme", None, {}, "miss"),
    (["a", "b"], [], {}, "miss-array"),
    # scalar
    ("Acme Corp", "Acme Corp", {}, "exact"),
    ("ACME CORP", "acme corp", {}, "case-insensitive"),
    ("CHARLOTTE, NC", "CHARLOTTE NC", {}, "punct-insensitive"),
    ("704-376-9896", "704.376.9896", {}, "punct-phone"),
    ("Acme Corp", "Beta Corp", {}, "content-diff"),
    ("Ste 300", "Ste 400", {}, "content-diff-2"),
    ("TED HENG", "TEO HENG", {}, "fuzzy-off"),
    ("TED HENG", "TEO HENG", {"fuzzy_threshold": 0.8}, "fuzzy-on"),
    # numbers
    ("$1,234.50", "1234.50", {}, "currency-strip"),
    (200, 200.0, {}, "int-float"),
    ("1234.50", "1234.50", {}, "num-equal"),
    ("1234.50", "1234.61", {}, "num-off-11c"),
    ("1234.50", "1234.505", {}, "num-boundary-half-cent"),
    (True, True, {}, "bool-equal"),
    (True, "yes", {}, "bool-vs-string"),
    # dates
    ("2024-03-15", "03/15/2024", {}, "date-format"),
    ("2024-03-15", "2024-03-16", {}, "date-mismatch"),
    ("2024-03-15", "15.03.2024", {}, "date-euro"),
    # arrays
    (["a", "b"], ["b", "a"], {}, "array-order"),
    ([{"name": "A"}, {"name": "B"}], [{"name": "B"}, {"name": "A"}], {}, "array-dicts-order"),
    (["a", "b", "c", "d", "e"], ["a", "b", "c", "d"], {}, "array-partial"),
    (["a", "b", "c"], ["a", "x", "y"], {}, "array-mostly-wrong"),
    # enum mappings
    ("10-K/A", "10K/A", {"mappings": {"10-K/A": ["10K/A", "10-KA"]}}, "enum-fold"),
    ("10-K/A", "10-Q", {"mappings": {"10-K/A": ["10K/A"]}}, "enum-no-match"),
    # provenance keys
    ({"name": "A"}, {"name": "A", "__source_text": "p2"}, {}, "provenance-strip"),
]

EPS = 1e-6

# Divergences where fieldbench is canonical-correct and Koji must conform.
# Keyed by case label → why fieldbench is right. The harness passes when the
# ONLY divergences are these (tracked koji-conformance items) and fails on any
# new/unexpected divergence (a regression or an unfaithful port change).
KNOWN_KOJI_GAPS = {
    "provenance-strip": (
        "Koji-B object fallback is `str(expected)==str(actual)` — order-sensitive and "
        "does not strip `__` provenance keys. fieldbench structural normalization is "
        "canonical. Koji to conform (fix object fallback, or import fieldbench)."
    ),
}


def load_koji_compare(path: Path):
    spec = importlib.util.spec_from_file_location("koji_test_runner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass + `from __future__ annotations` needs this
    spec.loader.exec_module(mod)
    return mod.compare_field


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--koji",
        type=Path,
        default=Path.home() / "dev/koji/playbook/koji/cli/test_runner.py",
        help="Path to Koji's cli/test_runner.py",
    )
    args = ap.parse_args(argv)
    if not args.koji.exists():
        print(f"error: koji scorer not found: {args.koji}", file=sys.stderr)
        return 2
    koji_compare = load_koji_compare(args.koji)

    diverged = []
    for expected, actual, kw, label in CASES:
        fb = fb_compare("f", expected, actual, **kw)
        kj = koji_compare("f", expected, actual, **kw)
        pass_ok = fb.passed == kj.passed
        score_ok = abs(fb.weighted_score - kj.weighted_score) < EPS
        if not (pass_ok and score_ok):
            diverged.append((label, fb, kj))

    total = len(CASES)
    if not diverged:
        print(f"PARITY OK — {total}/{total} cases agree (passed + weighted_score).")
        return 0

    unexpected = [d for d in diverged if d[0] not in KNOWN_KOJI_GAPS]
    print(f"{'case':<22} {'fieldbench':<30} {'koji-B':<24} status")
    print("-" * 88)
    for label, fb, kj in diverged:
        fbs = f"pass={fb.passed} score={fb.weighted_score:.3f} [{fb.bucket}]"
        kjs = f"pass={kj.passed} score={kj.weighted_score:.3f}"
        status = "known (koji-to-fix)" if label in KNOWN_KOJI_GAPS else "UNEXPECTED"
        print(f"{label:<22} {fbs:<30} {kjs:<24} {status}")

    print(f"\n{len(diverged)}/{total} diverge — {len(diverged) - len(unexpected)} known koji-gaps, {len(unexpected)} unexpected.")
    for label in (d[0] for d in diverged if d[0] in KNOWN_KOJI_GAPS):
        print(f"  · {label}: {KNOWN_KOJI_GAPS[label]}")
    if unexpected:
        print("\nUNEXPECTED divergences — investigate (fieldbench regression or unfaithful change).")
        return 1
    print("\nAll divergences are tracked koji-conformance items. Parity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
