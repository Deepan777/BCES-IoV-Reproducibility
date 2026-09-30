from __future__ import annotations

from bces.geometry.drift import DriftScales, KinematicState
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.codec import packet_from_offsets
from bces.protocol.packet import BCESPacket
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery

SENDER_ID = "rsu-phase1"
PAYLOAD = b"deterministic-object-set-payload"
POLICY_HASH = 0x10203040


def make_scales() -> DriftScales:
    return DriftScales(
        longitudinal_m=10.0,
        lateral_m=2.0,
        speed_mps=5.0,
        heading_rad=0.5,
        acceleration_mps2=2.0,
        curvature_inv_m=0.1,
        elapsed_s=1.0,
    )


def make_state(**overrides: float) -> KinematicState:
    values: dict[str, float | int] = {
        "x_m": 0.0,
        "y_m": 0.0,
        "speed_mps": 10.0,
        "heading_rad": 0.0,
        "acceleration_mps2": 0.0,
        "curvature_inv_m": 0.0,
        "timestamp_ms": 1_000_000,
    }
    values.update(overrides)
    return KinematicState(**values)  # type: ignore[arg-type]


def make_path() -> tuple[PathPoint, ...]:
    return tuple(
        PathPoint(
            x_m=float(index),
            y_m=0.0,
            speed_mps=10.0,
            heading_rad=0.0,
            curvature_inv_m=0.0,
            time_offset_s=index * 0.2,
        )
        for index in range(12)
    )


def make_query(**overrides: object) -> ReceiverQuery:
    values: dict[str, object] = {
        "query_id": 7,
        "expected_sender_id_hash": sender_id_hash(SENDER_ID),
        "reference_state": make_state(),
        "reference_path": make_path(),
        "behavior": Behavior.KEEP,
        "risk_class": 1,
        "policy_hash": POLICY_HASH,
        "calibration_id": 2,
        "drift_scales": make_scales(),
    }
    values.update(overrides)
    return ReceiverQuery(**values)  # type: ignore[arg-type]


def make_packet(query: ReceiverQuery | None = None, **overrides: object) -> BCESPacket:
    query = query or make_query()
    packet = packet_from_offsets(
        behavior_id=int(query.behavior),
        query_id=query.query_id,
        sender_id_hash=query.expected_sender_id_hash,
        payload_digest=digest32(PAYLOAD),
        policy_hash=query.policy_hash,
        t_ref_ms=query.wire_t_ref_ms,
        risk_class=query.risk_class,
        calibration_id=query.calibration_id,
        offsets=(1.0,) * 16,
    )
    values = dict(packet.__dict__)
    values.update(overrides)
    return BCESPacket(**values)  # type: ignore[arg-type]
