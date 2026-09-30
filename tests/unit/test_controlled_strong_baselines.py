"""Regression checks for the development-only strong-baseline audit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "25_audit_controlled_strong_baselines.py"
SPEC = importlib.util.spec_from_file_location("controlled_strong_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_rank_subset_is_exact_and_order_independent_for_ties():
    identifiers = np.array(["one", "two", "three", "four", "five"])
    scores = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    selected = AUDIT.deterministic_rank_subset(scores, identifiers, 0.4)
    assert selected.sum() == 2
    permutation = np.array([4, 2, 0, 3, 1])
    rearranged = AUDIT.deterministic_rank_subset(scores[permutation], identifiers[permutation], 0.4)
    assert set(identifiers[selected]) == set(identifiers[permutation][rearranged])


def test_rank_subset_rejects_invalid_scores_and_fraction():
    with pytest.raises(ValueError):
        AUDIT.deterministic_rank_subset(np.array([np.nan]), np.array(["a"]), 0.5)
    with pytest.raises(ValueError):
        AUDIT.deterministic_rank_subset(np.array([1.0]), np.array(["a"]), 0.0)


def test_paired_scenario_difference_preserves_observed_counts():
    valid = np.array([True, False, True, False, True, False])
    learned = np.array([True, False, True, False, True, False])
    comparator = np.array([False, True, False, True, False, True])
    scenarios = np.array(["a", "a", "b", "b", "c", "c"])
    result = AUDIT.paired_difference(valid, learned, comparator, scenarios)
    assert result["learned_minus_comparator_uar"] == -1.0
    assert result["scenario_count"] == 3
    assert result["confirmatory_significance_claim"] is False
