"""Open-loop labels for the unregistered stateless empty-view planner.

The exact same decoded-view semantics are used for reference, cached, and
fresh decisions. Future world frames enter only the label evaluation.
"""

from __future__ import annotations

from dataclasses import asdict

from bces.geometry.drift import compute_drift
from bces.models.reference_input import freeze_reference_input
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.planner import trajectory_deviation
from bces.oracle.validity import ValidityObservation, evaluate_validity
from bces.protocol.bindings import digest32, sender_id_hash
from bces.protocol.query import Behavior, PathPoint, ReceiverQuery
from bces.simulation.controlled_decisions import (
    evaluate_recorded_future, kinematic, route_for,
)
from bces.simulation.development_empty_view_input import (
    DecodedPolicyInput, DevelopmentEmptyViewPremise, decode_policy_input,
)
from bces.simulation.development_negative_evidence_planner import VerifiedEmptyObservation
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner
from bces.utils.hashing import canonical_json_hash


def _premise_hash(premise: DevelopmentEmptyViewPremise | None) -> str | None:
    return None if premise is None else canonical_json_hash(asdict(premise))


def _choose(planner, local, decoded, ego, route):
    """Return the actual action and a sidecar that never misnames base margins."""
    before_fallback = planner.fallback_uses
    before_empty = planner.fresh_empty_uses
    chosen = planner.plan(local, decoded.objects, ego, route)
    fallback = planner.fallback_uses > before_fallback
    fresh_empty = planner.fresh_empty_uses > before_empty
    physical = tuple(o for o in decoded.objects
                     if not isinstance(o, VerifiedEmptyObservation))
    base = observable_planner_features(planner, local, physical, ego, route)
    if not fallback and chosen.acceleration_mps2 != base[3]:
        raise AssertionError("base sidecar does not match the selected action")
    sidecar = {
        "actual_maneuver": chosen.maneuver,
        "actual_acceleration_mps2": chosen.acceleration_mps2,
        "actual_perceived_cost": chosen.perceived_cost.to_dict(),
        "stop_hold_fallback": fallback,
        "fresh_empty_used": fresh_empty,
        "decoded_status": decoded.status,
        "base_lattice_features": base,
        "base_lattice_is_actual_policy_margin": not fallback,
        "candidate_margin_available": not fallback,
    }
    return chosen, sidecar


def empty_reference_contract(
    *, scenario, split, behavior, timestamp, ego, local, message,
    empty_premise, planner: DevelopmentStatelessEmptyPlanner, scales,
):
    """Create a reference whose path and policy hash use the new planner."""
    if not isinstance(planner, DevelopmentStatelessEmptyPlanner):
        raise TypeError("development stateless empty-view planner required")
    if message.timestamp_ms != timestamp:
        raise ValueError("reference message and receiver time must coincide")
    decoded = decode_policy_input(message, now_ms=timestamp,
                                  empty_premise=empty_premise)
    route = route_for(ego, behavior, planner)
    selected, sidecar = _choose(planner, local, decoded, ego, route)
    identity = f"development-empty:{scenario}:{behavior}:{timestamp}:{decoded.payload_sha256[:12]}"
    query = ReceiverQuery(
        query_id=digest32((identity + ":query").encode()),
        expected_sender_id_hash=sender_id_hash(message.sender_id),
        reference_state=kinematic(ego, timestamp),
        reference_path=tuple(PathPoint(p.x_m, p.y_m, p.speed_mps,
                                       p.heading_rad, route.curvature_inv_m,
                                       p.time_s) for p in selected.points[:12]),
        behavior=Behavior(BEHAVIOR_IDS[behavior]), risk_class=1,
        policy_hash=planner.policy_hash, calibration_id=0,
        drift_scales=scales,
    )
    frozen = freeze_reference_input(
        message, query, generated_at_ms=timestamp,
        query_available_at_ms=timestamp, context_available_at_ms=timestamp,
        query_provenance="frozen_reference_policy", occlusion_proxy=0.0,
        estimated_delay_s=0.0, map_context_flags=1,
    )
    reference = {
        "reference_id": identity, "scenario_id": str(scenario),
        "split": split, "behavior": behavior,
        "behavior_id": BEHAVIOR_IDS[behavior],
        "reference_timestamp_ms": timestamp,
        "policy_hash": planner.policy_hash,
        "frozen_input": frozen.to_dict(),
        "payload_sha256": decoded.payload_sha256,
        "decoded_status": decoded.status,
        "empty_view_premise_sha256": _premise_hash(empty_premise),
        "empty_view_id": decoded.view_id,
        "policy_selection_sidecar": sidecar,
        "margin_sidecar_wire_binding_integrated": False,
        "source_view_premise_is_synthetic": True,
        "label_source": "controlled_simulation_open_loop_development",
    }
    return reference, query, decoded


def empty_decision_pair(
    *, reference, query, cached_reference: DecodedPolicyInput,
    ego, local, fresh_message, fresh_empty_premise, future_world,
    planner: DevelopmentStatelessEmptyPlanner, thresholds,
):
    """Evaluate cached versus fresh with identical current state and future."""
    if not isinstance(planner, DevelopmentStatelessEmptyPlanner):
        raise TypeError("development stateless empty-view planner required")
    if reference["policy_hash"] != query.policy_hash or query.policy_hash != planner.policy_hash:
        raise ValueError("candidate policy binding changed")
    if reference["payload_sha256"] != cached_reference.payload_sha256:
        raise ValueError("cached reference is not the bound packet")
    timestamp = fresh_message.timestamp_ms
    age = (timestamp - query.reference_state.timestamp_ms) / 1000.0
    if age < 0:
        raise ValueError("current decision precedes its reference")
    cached = DecodedPolicyInput(
        tuple(item.propagated(age) for item in cached_reference.objects),
        cached_reference.status, cached_reference.payload_sha256,
        cached_reference.view_id, age,
    )
    fresh = decode_policy_input(fresh_message, now_ms=timestamp,
                                empty_premise=fresh_empty_premise)
    route = route_for(ego, reference["behavior"], planner)
    cached_plan, cached_sidecar = _choose(planner, local, cached, ego, route)
    fresh_plan, fresh_sidecar = _choose(planner, local, fresh, ego, route)
    cached_cost = evaluate_recorded_future(
        planner, cached_plan, future_world, ego, route, timestamp)
    fresh_cost = evaluate_recorded_future(
        planner, fresh_plan, future_world, ego, route, timestamp)
    deviation = trajectory_deviation(cached_plan, fresh_plan)
    observation = ValidityObservation(
        cached_cost.total, fresh_cost.total, cached_cost.risk, deviation)
    return {
        "point_id": f"{reference['reference_id']}:{timestamp}",
        "reference_id": reference["reference_id"],
        "scenario_id": reference["scenario_id"], "split": reference["split"],
        "behavior": reference["behavior"], "current_timestamp_ms": timestamp,
        "cache_age_s": age,
        "normalized_drift": compute_drift(
            query.reference_state, kinematic(ego, timestamp),
            query.drift_scales).normalized,
        "valid": evaluate_validity(observation, thresholds),
        "cost_regret": observation.cost_regret,
        "cached_risk": cached_cost.risk,
        "cached_cost": cached_cost.to_dict(),
        "fresh_cost": fresh_cost.to_dict(),
        "trajectory_deviation_m": deviation,
        "policy_hash": planner.policy_hash,
        "cached_payload_sha256": cached.payload_sha256,
        "fresh_payload_sha256": fresh.payload_sha256,
        "cached_status": cached.status, "fresh_status": fresh.status,
        "cached_empty_view_id": cached.view_id,
        "fresh_empty_view_id": fresh.view_id,
        "fresh_empty_view_premise_sha256": _premise_hash(fresh_empty_premise),
        "cached_policy_selection_sidecar": cached_sidecar,
        "fresh_policy_selection_sidecar": fresh_sidecar,
        "future_world_used_only_for_labels": True,
        "same_label_world": True,
        "source_view_premise_is_synthetic": True,
        "label_source": "controlled_simulation_open_loop_development",
    }
