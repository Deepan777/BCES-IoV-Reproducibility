"""Object message -> oracle -> model batch -> codec -> receiver -> metrics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bces.data.smoke import (
    SMOKE_OCCLUDED_TRACK_ID,
    SmokeFixture,
    SmokeFrame,
    generate_smoke_fixture,
)
from bces.evaluation.validity import compute_validity_metrics
from bces.geometry.drift import DriftScales
from bces.models.batch import ModelBatchRow, build_model_batch_row
from bces.oracle.validity import (
    ValidityObservation,
    ValidityThresholds,
    evaluate_validity,
)
from bces.protocol.bindings import digest32, policy_hash32, sender_id_hash
from bces.protocol.codec import encode_packet, packet_from_offsets
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.protocol.receiver import ReceiverStateMachine
from bces.utils.hashing import canonical_json_hash

SMOKE_POLICY_DEFINITION = {
    "purpose": "phase2_verification_surrogate",
    "version": 1,
    "learned_model": False,
    "scientific_evidence": False,
}
SMOKE_POLICY_HASH = policy_hash32(SMOKE_POLICY_DEFINITION)
SMOKE_CALIBRATION_ID = 1
SMOKE_RISK_CLASS = 1
SMOKE_SURFACE_OFFSETS = (1.0,) * 16
SMOKE_REFRESH_GUARD_BAND = 0.05
EXPECTED_COVERAGE = 0.5
EXPECTED_UAR = 0.1

SMOKE_THRESHOLDS = ValidityThresholds(
    max_cost_regret=0.05,
    max_cached_risk=0.2,
    max_trajectory_deviation_m=0.5,
)


def _reference_path() -> tuple[PathPoint, ...]:
    return tuple(
        PathPoint(
            x_m=2.0 * index,
            y_m=0.0,
            speed_mps=10.0,
            heading_rad=0.0,
            curvature_inv_m=0.0,
            time_offset_s=0.2 * index,
        )
        for index in range(12)
    )


def _drift_scales() -> DriftScales:
    return DriftScales(
        longitudinal_m=10.0,
        lateral_m=2.0,
        speed_mps=5.0,
        heading_rad=0.5,
        acceleration_mps2=2.0,
        curvature_inv_m=0.1,
        elapsed_s=1.0,
    )


def build_smoke_query(fixture: SmokeFixture, behavior: Behavior) -> ReceiverQuery:
    return ReceiverQuery(
        query_id=100 + int(behavior),
        expected_sender_id_hash=sender_id_hash(fixture.cached_message.sender_id),
        reference_state=fixture.frames[0].ego_state,
        reference_path=_reference_path(),
        behavior=behavior,
        risk_class=SMOKE_RISK_CLASS,
        policy_hash=SMOKE_POLICY_HASH,
        calibration_id=SMOKE_CALIBRATION_ID,
        drift_scales=_drift_scales(),
    )


def _smoke_oracle(frame: SmokeFrame, fixture: SmokeFixture) -> ValidityObservation:
    cached_ids = {item.track_id for item in fixture.cached_message.objects}
    world_ids = {item.track_id for item in frame.world_objects}
    if (
        SMOKE_OCCLUDED_TRACK_ID not in cached_ids
        or SMOKE_OCCLUDED_TRACK_ID not in world_ids
    ):
        raise ValueError("smoke oracle requires the occluded pedestrian")
    excess_steps = max(0, frame.index - 8)
    return ValidityObservation(
        cached_cost=1.0 + 0.1 * excess_steps,
        fresh_cost=1.0,
        cached_risk=0.1 + 0.05 * max(0, frame.index - 9),
        trajectory_deviation_m=0.1 + 0.1 * max(0, frame.index - 9),
    )


def _batch_payload(batch: ModelBatchRow) -> dict[str, Any]:
    return asdict(batch)


def run_smoke_pipeline(fixture: SmokeFixture | None = None) -> dict[str, Any]:
    fixture = fixture or generate_smoke_fixture()
    payload = fixture.cached_message.encode()
    payload_digest = digest32(payload)
    receiver = ReceiverStateMachine(refresh_guard_band=SMOKE_REFRESH_GUARD_BAND)
    outcomes = []
    rows: list[dict[str, Any]] = []
    packet_hex_by_behavior: dict[str, str] = {}
    for frame in fixture.frames:
        query = build_smoke_query(fixture, frame.intended_behavior)
        batch = build_model_batch_row(
            fixture.cached_message,
            query,
            occlusion_proxy=1.0 / 3.0,
            estimated_delay_s=(frame.timestamp_ms - fixture.cached_message.timestamp_ms)
            / 1000.0,
            map_context_flags=1,
        )
        observation = _smoke_oracle(frame, fixture)
        oracle_valid = evaluate_validity(observation, SMOKE_THRESHOLDS)
        packet = packet_from_offsets(
            behavior_id=int(query.behavior),
            query_id=query.query_id,
            sender_id_hash=query.expected_sender_id_hash,
            payload_digest=payload_digest,
            policy_hash=query.policy_hash,
            t_ref_ms=query.wire_t_ref_ms,
            risk_class=query.risk_class,
            calibration_id=query.calibration_id,
            offsets=SMOKE_SURFACE_OFFSETS,
        )
        encoded = encode_packet(packet)
        packet_hex_by_behavior[query.behavior.name.lower()] = encoded.hex()
        decision = receiver.evaluate(
            encoded_packet=encoded,
            cooperative_payload=payload,
            query=query,
            current_state=frame.ego_state,
            current_behavior=frame.intended_behavior,
            current_policy_hash=SMOKE_POLICY_HASH,
            actual_sender_id=fixture.cached_message.sender_id,
        )
        outcomes.append((decision.state, oracle_valid))
        rows.append(
            {
                "frame": frame.index,
                "behavior": frame.intended_behavior.name.lower(),
                "oracle_valid": oracle_valid,
                "cost_regret": observation.cost_regret,
                "cached_risk": observation.cached_risk,
                "trajectory_deviation_m": observation.trajectory_deviation_m,
                "model_batch_sha256": canonical_json_hash(_batch_payload(batch)),
                "packet_bytes": len(encoded),
                "decision": decision.state.value,
                "minimum_slack": decision.minimum_slack,
            }
        )
    metrics = compute_validity_metrics(outcomes)
    first_surface_invalid = next(
        row["frame"]
        for row in rows
        if row["decision"] == "INVALID"
        and isinstance(row["minimum_slack"], float)
        and row["minimum_slack"] < 0.0
    )
    result = {
        "schema_version": 1,
        "fixture_version": fixture.version,
        "evidence_use": fixture.evidence_use,
        "frame_count": len(fixture.frames),
        "object_tracks": sorted(
            item.track_id for item in fixture.cached_message.objects
        ),
        "occluded_track_ids": sorted(
            {track for frame in fixture.frames for track in frame.occluded_track_ids}
        ),
        "behaviors": sorted(
            {frame.intended_behavior.name.lower() for frame in fixture.frames}
        ),
        "known_surface_boundary_frame": fixture.known_surface_boundary_frame,
        "observed_surface_boundary_frame": first_surface_invalid,
        "policy_hash": SMOKE_POLICY_HASH,
        "payload_digest": payload_digest,
        "payload_bytes": len(payload),
        "packet_hex_by_behavior": packet_hex_by_behavior,
        "metrics": metrics.to_dict(),
        "rows": rows,
    }
    result["deterministic_sha256"] = canonical_json_hash(result)
    return result
