"""Compact world and trajectory types shared by observational and SUMO oracles."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class EgoState:
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float
    acceleration_mps2: float = 0.0
    curvature_inv_m: float = 0.0
    length_m: float = 4.8
    width_m: float = 1.9

    def __post_init__(self) -> None:
        for name in (
            "x_m", "y_m", "speed_mps", "heading_rad", "acceleration_mps2",
            "curvature_inv_m", "length_m", "width_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.speed_mps < 0 or self.length_m <= 0 or self.width_m <= 0:
            raise ValueError("invalid ego dimensions or speed")


@dataclass(frozen=True)
class WorldObject:
    track_id: str
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    heading_rad: float
    length_m: float
    width_m: float
    class_id: int = 0
    confidence: float = 1.0
    source: str = "ground_truth"

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id is required")
        for name in (
            "x_m", "y_m", "vx_mps", "vy_mps", "heading_rad", "length_m",
            "width_m", "confidence",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.length_m <= 0 or self.width_m <= 0 or not 0 <= self.confidence <= 1:
            raise ValueError("invalid object dimensions or confidence")

    def propagated(self, seconds: float) -> WorldObject:
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("propagation time must be finite and non-negative")
        return WorldObject(
            track_id=self.track_id,
            x_m=self.x_m + self.vx_mps * seconds,
            y_m=self.y_m + self.vy_mps * seconds,
            vx_mps=self.vx_mps,
            vy_mps=self.vy_mps,
            heading_rad=self.heading_rad,
            length_m=self.length_m,
            width_m=self.width_m,
            class_id=self.class_id,
            confidence=self.confidence,
            source=self.source,
        )


@dataclass(frozen=True)
class GroundTruthWorld:
    timestamp_ms: int
    objects: tuple[WorldObject, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ms, int) or self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True)
class Route:
    origin_x_m: float
    origin_y_m: float
    heading_rad: float
    lane_width_m: float = 3.6
    speed_limit_mps: float = 20.0
    allow_left: bool = True
    allow_right: bool = True
    intended_behavior: str = "keep"
    curvature_inv_m: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "origin_x_m", "origin_y_m", "heading_rad", "lane_width_m",
            "speed_limit_mps", "curvature_inv_m",
        ):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.lane_width_m <= 0 or self.speed_limit_mps <= 0:
            raise ValueError("route widths and speed limit must be positive")
        if self.intended_behavior not in {"keep", "brake", "accelerate", "left", "right"}:
            raise ValueError("unsupported intended behavior")


@dataclass(frozen=True)
class TrajectoryPoint:
    time_s: float
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float
    acceleration_mps2: float
    lateral_offset_m: float


@dataclass(frozen=True)
class CostVector:
    collision: float
    ttc: float
    route: float
    lane: float
    comfort: float
    progress: float
    total: float
    minimum_ttc_s: float | None

    @property
    def risk(self) -> float:
        return max(self.collision, self.ttc)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedTrajectory:
    maneuver: str
    acceleration_mps2: float
    points: tuple[TrajectoryPoint, ...]
    perceived_cost: CostVector

    def to_dict(self) -> dict[str, Any]:
        return {
            "maneuver": self.maneuver,
            "acceleration_mps2": self.acceleration_mps2,
            "points": [asdict(point) for point in self.points],
            "perceived_cost": self.perceived_cost.to_dict(),
        }
