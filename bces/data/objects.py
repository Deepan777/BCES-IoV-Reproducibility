"""Validated object-level cooperative-perception message representation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ObjectSource(str, Enum):
    VEHICLE = "vehicle"
    INFRASTRUCTURE = "infrastructure"
    COOPERATIVE = "cooperative"


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class PerceivedObject:
    track_id: str
    class_id: int
    x_m: float
    y_m: float
    vx_mps: float
    vy_mps: float
    length_m: float
    width_m: float
    confidence: float
    source: ObjectSource

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        if not isinstance(self.class_id, int) or not 0 <= self.class_id <= 255:
            raise ValueError("class_id must be uint8")
        for name in ("x_m", "y_m", "vx_mps", "vy_mps", "length_m", "width_m"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("object dimensions must be positive")
        confidence = _finite("confidence", self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        try:
            source = ObjectSource(self.source)
        except ValueError as exc:
            raise ValueError("unsupported object source") from exc
        object.__setattr__(self, "source", source)

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "vx_mps": self.vx_mps,
            "vy_mps": self.vy_mps,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "confidence": self.confidence,
            "source": self.source.value,
        }


@dataclass(frozen=True)
class ObjectMessage:
    message_id: int
    sender_id: str
    timestamp_ms: int
    objects: tuple[PerceivedObject, ...]
    frame: str = "map_local"

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, int) or not 0 <= self.message_id < 2**64:
            raise ValueError("message_id must be uint64")
        if not isinstance(self.sender_id, str) or not self.sender_id:
            raise ValueError("sender_id must be a non-empty string")
        if not isinstance(self.timestamp_ms, int) or not 0 <= self.timestamp_ms < 2**64:
            raise ValueError("timestamp_ms must be uint64")
        if self.frame != "map_local":
            raise ValueError("object messages must use the map_local frame")
        objects = tuple(self.objects)
        track_ids = [item.track_id for item in objects]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("object-message track IDs must be unique")
        object.__setattr__(self, "objects", objects)

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "timestamp_ms": self.timestamp_ms,
            "frame": self.frame,
            "objects": [item.to_wire_dict() for item in self.objects],
        }

    def encode(self) -> bytes:
        return json.dumps(
            self.to_wire_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
