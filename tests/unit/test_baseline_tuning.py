from __future__ import annotations

import numpy as np

from bces.baselines.tuning import (
    risk_coverage_curve,
    select_risk_constrained_threshold,
)


def test_threshold_curve_is_deterministic_and_uses_accepted_denominator() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    valid = np.array([True, False, True, False])
    clusters = ("a", "a", "b", "b")
    first = risk_coverage_curve(
        scores, valid, clusters, (0.15, 0.25, 0.45),
        confidence=0.95, bootstrap_replicates=100, seed=11,
    )
    second = risk_coverage_curve(
        scores, valid, clusters, (0.15, 0.25, 0.45),
        confidence=0.95, bootstrap_replicates=100, seed=11,
    )
    assert first == second
    assert first[1]["unsafe_accept_rate"] == 0.5
    assert first[1]["coverage"] == 0.5


def test_threshold_selection_marks_failed_target_without_hiding_it() -> None:
    curve = [
        {"threshold": 0.1, "accepted": 10, "coverage": 0.1, "unsafe_accept_upper": 0.2},
        {"threshold": 0.2, "accepted": 20, "coverage": 0.2, "unsafe_accept_upper": 0.3},
    ]
    selected = select_risk_constrained_threshold(
        curve, unsafe_upper_target=0.05, minimum_accepted=10
    )
    assert selected["fallback"] is True
    assert selected["target_met"] is False
    assert selected["threshold"] == 0.1
