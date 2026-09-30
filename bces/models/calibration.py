"""Scenario-clustered conservative shrinkage for quantized expiry surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bces.geometry.codebooks import NORMAL_CODEBOOK_V1

MAX_Q15_SCALE = 65535
Q15_DENOMINATOR = 32768.0


@dataclass(frozen=True)
class CalibrationTarget:
    unsafe_accept_upper: float = 0.05
    confidence: float = 0.95
    bootstrap_replicates: int = 2000
    refresh_guard_band: float = 0.05
    minimum_accepted: int = 100


def quantize_numpy(offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(offsets, dtype=np.float64)
    scales_q15 = np.clip(np.ceil(values.max(axis=1, keepdims=True) * Q15_DENOMINATOR), 1, MAX_Q15_SCALE)
    scales = scales_q15 / Q15_DENOMINATOR
    quantized = np.clip(np.floor(255.0 * values / scales + 0.5), 0, 255)
    return scales * quantized / 255.0


def decision_counts(
    offsets: np.ndarray, drift: np.ndarray, valid: np.ndarray,
    *, shrinkage: float, guard_band: float, normals: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    calibrated = quantize_numpy(np.maximum(0.0, offsets - float(shrinkage)))
    normal_array = (
        np.asarray(NORMAL_CODEBOOK_V1, dtype=np.float64)
        if normals is None
        else np.asarray(normals, dtype=np.float64)
    )
    minimum_slack = (
        calibrated - np.asarray(drift, dtype=np.float64) @ normal_array.T
    ).min(axis=1)
    accepted = minimum_slack >= guard_band
    invalid = minimum_slack < 0.0
    refresh = ~(accepted | invalid)
    unsafe = accepted & ~np.asarray(valid, dtype=bool)
    accepted_count = int(accepted.sum())
    return {
        "total": len(valid), "accepted": accepted_count,
        "unsafe_accepted": int(unsafe.sum()), "refresh": int(refresh.sum()),
        "invalid": int(invalid.sum()), "coverage": accepted_count / max(len(valid), 1),
        "unsafe_accept_rate": (int(unsafe.sum()) / accepted_count) if accepted_count else None,
    }


def clustered_upper_bound(
    accepted: np.ndarray, unsafe: np.ndarray, clusters: tuple[str, ...],
    *, confidence: float, replicates: int, seed: int,
) -> float:
    names = sorted(set(clusters))
    if not names:
        raise ValueError("at least one scenario cluster is required")
    accepted_by_cluster = np.asarray(
        [accepted[np.asarray([item == name for item in clusters])].sum() for name in names],
        dtype=np.int64,
    )
    unsafe_by_cluster = np.asarray(
        [unsafe[np.asarray([item == name for item in clusters])].sum() for name in names],
        dtype=np.int64,
    )
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(names), size=(replicates, len(names)))
    accepted_totals = accepted_by_cluster[sampled].sum(axis=1)
    unsafe_totals = unsafe_by_cluster[sampled].sum(axis=1)
    rates = np.where(accepted_totals > 0, unsafe_totals / np.maximum(accepted_totals, 1), 1.0)
    return float(np.quantile(rates, confidence, method="higher"))


def calibration_curve(
    offsets: np.ndarray, drift: np.ndarray, valid: np.ndarray,
    clusters: tuple[str, ...], grid: tuple[float, ...], target: CalibrationTarget,
    *, seed: int, normals: np.ndarray | None = None,
) -> list[dict]:
    normal_array = (
        np.asarray(NORMAL_CODEBOOK_V1, dtype=np.float64)
        if normals is None
        else np.asarray(normals, dtype=np.float64)
    )
    labels = np.asarray(valid, dtype=bool)
    rows = []
    for index, shrinkage in enumerate(grid):
        calibrated = quantize_numpy(np.maximum(0.0, offsets - shrinkage))
        minimum = (calibrated - drift @ normal_array.T).min(axis=1)
        accepted = minimum >= target.refresh_guard_band
        unsafe = accepted & ~labels
        counts = decision_counts(
            offsets, drift, valid, shrinkage=shrinkage,
            guard_band=target.refresh_guard_band, normals=normal_array,
        )
        counts.update(
            {
                "shrinkage": shrinkage,
                "unsafe_accept_upper": clustered_upper_bound(
                    accepted, unsafe, clusters, confidence=target.confidence,
                    replicates=target.bootstrap_replicates, seed=seed + index,
                ),
            }
        )
        rows.append(counts)
    return rows


def select_shrinkage(curve: list[dict], target: CalibrationTarget) -> dict:
    eligible = [
        row for row in curve
        if row["accepted"] >= target.minimum_accepted
        and row["unsafe_accept_upper"] <= target.unsafe_accept_upper
    ]
    if not eligible:
        fallback = min(
            curve,
            key=lambda row: (row["unsafe_accept_upper"], -row["coverage"], row["shrinkage"]),
        )
        return {**fallback, "target_met": False}
    selected = max(eligible, key=lambda row: (row["coverage"], -row["shrinkage"]))
    return {**selected, "target_met": True}
