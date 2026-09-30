"""Validated version-1 receiver query and reference-trajectory schema."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import IntEnum

from bces.geometry.codebooks import NORMAL_CODEBOOK_ID_V1
from bces.geometry.drift import DRIFT_SCHEMA_ID, DriftScales, KinematicState

PROTOCOL_VERSION_V1 = 1
PATH_POINT_COUNT_V1 = 12


class Behavior(IntEnum):
    KEEP = 0
    BRAKE = 1
    ACCELERATE = 2
    LEFT = 3
    RIGHT = 4


def _uint(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must be uint{bits}")


@dataclass(frozen=True)
class PathPoint:
    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float
    curvature_inv_m: float
    time_offset_s: float

    def __post_init__(self) -> None:
        for name in (
            "x_m",
            "y_m",
            "speed_mps",
            "heading_rad",
            "curvature_inv_m",
            "time_offset_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.time_offset_s < 0.0:
            raise ValueError("time_offset_s must be non-negative")


@dataclass(frozen=True)
class ReceiverQuery:
    query_id: int
    expected_sender_id_hash: int
    reference_state: KinematicState
    reference_path: tuple[PathPoint, ...]
    behavior: Behavior
    risk_class: int
    policy_hash: int
    calibration_id: int
    drift_scales: DriftScales
    protocol_version: int = PROTOCOL_VERSION_V1
    drift_schema_id: int = DRIFT_SCHEMA_ID
    normal_codebook_id: int = NORMAL_CODEBOOK_ID_V1

    def __post_init__(self) -> None:
        _uint("query_id", self.query_id, 32)
        _uint("expected_sender_id_hash", self.expected_sender_id_hash, 32)
        _uint("risk_class", self.risk_class, 8)
        _uint("policy_hash", self.policy_hash, 32)
        _uint("calibration_id", self.calibration_id, 8)
        if self.protocol_version != PROTOCOL_VERSION_V1:
            raise ValueError("unsupported query protocol version")
        if self.drift_schema_id != DRIFT_SCHEMA_ID:
            raise ValueError("unsupported drift schema")
        if self.normal_codebook_id != NORMAL_CODEBOOK_ID_V1:
            raise ValueError("unsupported normal codebook")
        try:
            behavior = Behavior(self.behavior)
        except ValueError as exc:
            raise ValueError("unsupported behavior") from exc
        object.__setattr__(self, "behavior", behavior)
        path = tuple(self.reference_path)
        if len(path) != PATH_POINT_COUNT_V1:
            raise ValueError(
                f"version 1 requires exactly {PATH_POINT_COUNT_V1} path points"
            )
        if any(
            current.time_offset_s <= previous.time_offset_s
            for previous, current in itertools.pairwise(path)
        ):
            raise ValueError("reference-path time offsets must be strictly increasing")
        object.__setattr__(self, "reference_path", path)

    @property
    def wire_t_ref_ms(self) -> int:
        return self.reference_state.timestamp_ms & 0xFFFFFFFF
