"""Version-1 seven-dimensional receiver-state drift transform."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, fields

DRIFT_SCHEMA_ID = 1
DRIFT_DIMENSIONS = (
    "longitudinal_displacement",
    "lateral_displacement",
    "speed_change",
    "heading_change",
    "acceleration_change",
    "curvature_change",
    "elapsed_time",
)
DRIFT_DIMENSION = len(DRIFT_DIMENSIONS)


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to the half-open interval [-pi, pi)."""
    angle = _require_finite("angle_rad", angle_rad)
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return -math.pi if wrapped == math.pi else wrapped


@dataclass(frozen=True)
class KinematicState:
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float
    acceleration_mps2: float
    curvature_inv_m: float
    timestamp_ms: int

    def __post_init__(self) -> None:
        for field_name in (
            "x_m",
            "y_m",
            "speed_mps",
            "heading_rad",
            "acceleration_mps2",
            "curvature_inv_m",
        ):
            object.__setattr__(
                self, field_name, _require_finite(field_name, getattr(self, field_name))
            )
        if not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be a non-negative integer")


@dataclass(frozen=True)
class DriftScales:
    longitudinal_m: float
    lateral_m: float
    speed_mps: float
    heading_rad: float
    acceleration_mps2: float
    curvature_inv_m: float
    elapsed_s: float

    def __post_init__(self) -> None:
        for field_info in fields(self):
            value = _require_finite(field_info.name, getattr(self, field_info.name))
            if value <= 0.0:
                raise ValueError(f"{field_info.name} must be positive")
            object.__setattr__(self, field_info.name, value)

    def as_tuple(self) -> tuple[float, ...]:
        return tuple(getattr(self, field_info.name) for field_info in fields(self))


@dataclass(frozen=True)
class DriftVector:
    raw: tuple[float, ...]
    normalized: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.raw) != DRIFT_DIMENSION or len(self.normalized) != DRIFT_DIMENSION:
            raise ValueError(f"drift vectors must contain {DRIFT_DIMENSION} values")
        for index, value in enumerate((*self.raw, *self.normalized)):
            _require_finite(f"drift[{index}]", value)


def compute_drift(
    reference: KinematicState,
    current: KinematicState,
    scales: DriftScales,
) -> DriftVector:
    """Compute signed path-frame drift and normalize it using frozen scales."""
    if current.timestamp_ms < reference.timestamp_ms:
        raise ValueError("current timestamp precedes the query reference timestamp")
    dx = current.x_m - reference.x_m
    dy = current.y_m - reference.y_m
    cosine = math.cos(reference.heading_rad)
    sine = math.sin(reference.heading_rad)
    delta_s = cosine * dx + sine * dy
    delta_l = -sine * dx + cosine * dy
    raw = (
        delta_s,
        delta_l,
        current.speed_mps - reference.speed_mps,
        wrap_angle_rad(current.heading_rad - reference.heading_rad),
        current.acceleration_mps2 - reference.acceleration_mps2,
        current.curvature_inv_m - reference.curvature_inv_m,
        (current.timestamp_ms - reference.timestamp_ms) / 1000.0,
    )
    normalized = tuple(value / scale for value, scale in zip(raw, scales.as_tuple()))
    return DriftVector(raw=raw, normalized=normalized)


def validate_normalized_drift(values: Iterable[float]) -> tuple[float, ...]:
    drift = tuple(_require_finite("normalized drift", value) for value in values)
    if len(drift) != DRIFT_DIMENSION:
        raise ValueError(f"normalized drift must contain {DRIFT_DIMENSION} values")
    return drift
