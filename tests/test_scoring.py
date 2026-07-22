"""Golden cases for the canonical scorer — the anti-drift contract (BR-4 §10).

Every case here pins an (expected, actual) → (bucket, passed) outcome. Koji's
scorers must conform to these; a disagreement is a bug to fix in Koji, not here.
"""

from fieldbench.scoring import array_f1, compare_field


def bucket(exp, act, **kw):
    r = compare_field("f", exp, act, **kw)
    return r.bucket, r.passed


# ── Four-way null semantics ──────────────────────────────────────────

def test_correct_absence():
    assert bucket(None, None) == ("correct_absence", True)
    assert bucket("", []) == ("correct_absence", True)
    assert bucket([], {}) == ("correct_absence", True)


def test_hallucination():
    assert bucket(None, "Acme Corp") == ("hallucination", False)
    assert bucket("", ["x"]) == ("hallucination", False)


def test_miss():
    assert bucket("Acme Corp", None) == ("miss", False)
    assert bucket(["a", "b"], []) == ("miss", False)


# ── Scalar / string ──────────────────────────────────────────────────

def test_exact_and_case_insensitive():
    assert bucket("Acme Corp", "Acme Corp") == ("match", True)
    assert bucket("ACME CORP", "acme corp") == ("match", True)


def test_punctuation_insensitive():
    assert bucket("CHARLOTTE, NC", "CHARLOTTE NC") == ("match", True)
    assert bucket("704-376-9896", "704.376.9896") == ("match", True)


def test_content_difference_still_fails():
    assert bucket("Acme Corp", "Beta Corp") == ("wrong_value", False)
    assert bucket("Ste 300", "Ste 400") == ("wrong_value", False)


def test_fuzzy_only_when_enabled():
    assert bucket("TED HENG", "TEO HENG") == ("wrong_value", False)  # off by default
    assert bucket("TED HENG", "TEO HENG", fuzzy_threshold=0.8) == ("match", True)


# ── Numbers ──────────────────────────────────────────────────────────

def test_numeric_tolerance_and_formatting():
    assert bucket("$1,234.50", "1234.50") == ("match", True)
    assert bucket(200, 200.0) == ("match", True)
    assert bucket("1234.50", "1234.50") == ("match", True)
    assert bucket("1234.50", "1234.61") == ("wrong_value", False)  # 11c off > 0.01 tolerance


# ── Dates ────────────────────────────────────────────────────────────

def test_date_normalization():
    assert bucket("2024-03-15", "03/15/2024") == ("match", True)
    assert bucket("2024-03-15", "2024-03-16") == ("wrong_value", False)


# ── Arrays ───────────────────────────────────────────────────────────

def test_array_order_insensitive_exact():
    assert bucket(["a", "b"], ["b", "a"]) == ("match", True)


def test_array_of_dicts_order_insensitive():
    exp = [{"name": "A"}, {"name": "B"}]
    act = [{"name": "B"}, {"name": "A"}]
    assert bucket(exp, act) == ("match", True)


def test_array_partial_is_wrong_value_but_scores_f1():
    r = compare_field("f", ["a", "b", "c", "d", "e"], ["a", "b", "c", "d"])
    assert r.bucket == "wrong_value" and not r.passed
    assert 0.7 < r.weighted_score < 1.0  # ~0.89 F1, not 0


def test_array_f1_bounds():
    assert array_f1([], []) == 1.0
    assert array_f1(["a"], []) == 0.0
    assert array_f1(["a", "b"], ["a", "b"]) == 1.0


# ── Enum mappings ────────────────────────────────────────────────────

def test_enum_mapping_folding():
    m = {"10-K/A": ["10K/A", "10-KA"]}
    assert bucket("10-K/A", "10K/A", mappings=m) == ("match", True)
    assert bucket("10-K/A", "10-Q", mappings=m) == ("wrong_value", False)


# ── Provenance keys are ignored ──────────────────────────────────────

def test_provenance_keys_stripped():
    exp = {"name": "A"}
    act = {"name": "A", "__source_text": "found on page 2"}
    assert bucket(exp, act) == ("match", True)
