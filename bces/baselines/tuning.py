"""Validation-only risk-constrained threshold tuning for scalar gate scores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bces.models.calibration import clustered_upper_bound


def risk_coverage_curve(
    scores: np.ndarray,
    valid: np.ndarray,
    clusters: Sequence[str],
    thresholds: Sequence[float],
    *,
    confidence: float,
    bootstrap_replicates: int,
    seed: int,
) -> list[dict]:
    """Build a curve where lower score means safer and score <= threshold accepts."""
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if scores.ndim != 1 or scores.shape != valid.shape or len(clusters) != len(valid):
        raise ValueError("scores, labels, and clusters must be aligned one-dimensional arrays")
    curve = []
    for index, threshold in enumerate(thresholds):
        accepted = scores <= float(threshold)
        unsafe = accepted & ~valid
        accepted_count = int(accepted.sum())
        unsafe_count = int(unsafe.sum())
        curve.append(
            {
                "threshold": float(threshold),
                "total": len(valid),
                "accepted": accepted_count,
                "unsafe_accepted": unsafe_count,
                "coverage": accepted_count / len(valid),
                "unsafe_accept_rate": (
                    unsafe_count / accepted_count if accepted_count else None
                ),
                "unsafe_accept_upper": clustered_upper_bound(
                    accepted,
                    unsafe,
                    clusters,
                    confidence=confidence,
                    replicates=bootstrap_replicates,
                    seed=seed + index,
                ),
            }
        )
    return curve


def select_risk_constrained_threshold(
    curve: Sequence[dict], *, unsafe_upper_target: float, minimum_accepted: int
) -> dict:
    eligible = [
        row
        for row in curve
        if row["accepted"] >= minimum_accepted
        and row["unsafe_accept_upper"] <= unsafe_upper_target
    ]
    if eligible:
        selected = max(eligible, key=lambda row: (row["coverage"], row["threshold"]))
        return {**selected, "target_met": True, "fallback": False}
    nondegenerate = [row for row in curve if row["accepted"] >= minimum_accepted]
    selected = min(
        nondegenerate or list(curve),
        key=lambda row: (row["unsafe_accept_upper"], -row["coverage"], row["threshold"]),
    )
    return {**selected, "target_met": False, "fallback": True}
