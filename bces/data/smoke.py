"""Deterministic, generated 20-frame smoke fixture; never scientific evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bces.geometry.drift import KinematicState
from bces.protocol.query import Behavior

from .objects import ObjectMessage, ObjectSource, PerceivedObject

SMOKE_FIXTURE_VERSION = 1
SMOKE_FRAME_COUNT = 20
SMOKE_REFERENCE_TIMESTAMP_MS = 1_000_000
SMOKE_SENDER_ID = "rsu-smoke-1"
SMOKE_OCCLUDED_TRACK_ID = "ped-hidden"


@dataclass(frozen=True)
class SmokeFrame:
    index: int
    timestamp_ms: int
    intended_behavior: Behavior
    ego_state: KinematicState
    world_objects: tuple[PerceivedObject, ...]
    occluded_track_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            "intended_behavior": self.intended_behavior.name.lower(),
            "ego_state": {
                "x_m": self.ego_state.x_m,
                "y_m": self.ego_state.y_m,
                "speed_mps": self.ego_state.speed_mps,
                "heading_rad": self.ego_state.heading_rad,
                "acceleration_mps2": self.ego_state.acceleration_mps2,
                "curvature_inv_m": self.ego_state.curvature_inv_m,
                "timestamp_ms": self.ego_state.timestamp_ms,
            },
            "world_objects": [item.to_wire_dict() for item in self.world_objects],
            "occluded_track_ids": list(self.occluded_track_ids),
        }


@dataclass(frozen=True)
class SmokeFixture:
    cached_message: ObjectMessage
    frames: tuple[SmokeFrame, ...]
    known_surface_boundary_frame: int
    evidence_use: str = "software_verification_only"
    version: int = SMOKE_FIXTURE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "evidence_use": self.evidence_use,
            "known_surface_boundary_frame": self.known_surface_boundary_frame,
            "cached_message": self.cached_message.to_wire_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
        }


def _objects_at(elapsed_s: float) -> tuple[PerceivedObject, ...]:
    return (
        PerceivedObject(
            track_id="veh-lead",
            class_id=1,
            x_m=20.0 + 8.0 * elapsed_s,
            y_m=0.0,
            vx_mps=8.0,
            vy_mps=0.0,
            length_m=4.6,
            width_m=1.9,
            confidence=0.96,
            source=ObjectSource.INFRASTRUCTURE,
        ),
        PerceivedObject(
            track_id="veh-adjacent",
            class_id=1,
            x_m=12.0 + 9.0 * elapsed_s,
            y_m=3.5,
            vx_mps=9.0,
            vy_mps=0.0,
            length_m=4.4,
            width_m=1.8,
            confidence=0.93,
            source=ObjectSource.INFRASTRUCTURE,
        ),
        PerceivedObject(
            track_id=SMOKE_OCCLUDED_TRACK_ID,
            class_id=2,
            x_m=10.0,
            y_m=5.0 - 1.2 * elapsed_s,
            vx_mps=0.0,
            vy_mps=-1.2,
            length_m=0.6,
            width_m=0.6,
            confidence=0.88,
            source=ObjectSource.INFRASTRUCTURE,
        ),
    )


def generate_smoke_fixture() -> SmokeFixture:
    reference_objects = _objects_at(0.0)
    cached_message = ObjectMessage(
        message_id=9001,
        sender_id=SMOKE_SENDER_ID,
        timestamp_ms=SMOKE_REFERENCE_TIMESTAMP_MS,
        objects=reference_objects,
    )
    frames = []
    for index in range(SMOKE_FRAME_COUNT):
        timestamp_ms = SMOKE_REFERENCE_TIMESTAMP_MS + index * 100
        behavior = Behavior.KEEP if index < 10 else Behavior.BRAKE
        frames.append(
            SmokeFrame(
                index=index,
                timestamp_ms=timestamp_ms,
                intended_behavior=behavior,
                ego_state=KinematicState(
                    x_m=0.0,
                    y_m=0.0,
                    speed_mps=10.0,
                    heading_rad=0.0,
                    acceleration_mps2=0.0,
                    curvature_inv_m=0.0,
                    timestamp_ms=timestamp_ms,
                ),
                world_objects=_objects_at(index / 10.0),
                occluded_track_ids=(SMOKE_OCCLUDED_TRACK_ID,),
            )
        )
    return SmokeFixture(
        cached_message=cached_message,
        frames=tuple(frames),
        known_surface_boundary_frame=11,
    )
