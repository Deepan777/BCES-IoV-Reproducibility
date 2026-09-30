"""Training-partition-only normalization for receiver drift."""

from __future__ import annotations

import math
from collections.abc import Iterable

from bces.geometry.drift import (
    DRIFT_DIMENSION,
    DriftScales,
    KinematicState,
    compute_drift,
)

MINIMUM_SCALES = (1.0, 0.5, 0.5, math.radians(2), 0.5, 0.005, 0.5)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot fit a scale without training observations")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def fit_drift_scales(
    trajectories: Iterable[tuple[KinematicState, tuple[KinematicState, ...]]],
    *, quantile: float = 0.95,
) -> tuple[DriftScales, dict]:
    if not 0.5 <= quantile < 1.0:
        raise ValueError("scale quantile must be in [0.5, 1)")
    columns = [[] for _ in range(DRIFT_DIMENSION)]
    samples = 0
    unit = DriftScales(*(1.0,) * DRIFT_DIMENSION)
    for reference, currents in trajectories:
        for current in currents:
            if current.timestamp_ms <= reference.timestamp_ms:
                continue
            raw = compute_drift(reference, current, unit).raw
            for column, value in zip(columns, raw):
                column.append(abs(value))
            samples += 1
    fitted = tuple(max(floor, _quantile(column, quantile)) for floor, column in zip(MINIMUM_SCALES, columns))
    scales = DriftScales(*fitted)
    return scales, {
        "method": "absolute_training_drift_linear_quantile_robust_kinematics_v2",
        "quantile": quantile,
        "sample_count": samples,
        "minimum_scales": MINIMUM_SCALES,
        "observed_absolute_maxima": tuple(max(column) for column in columns),
    }
