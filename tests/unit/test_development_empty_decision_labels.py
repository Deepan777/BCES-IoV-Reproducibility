import hashlib
import math

import pytest

from bces.data.objects import ObjectMessage
from bces.geometry.drift import DriftScales
from bces.oracle.validity import ValidityThresholds
from bces.oracle.world import EgoState
from bces.simulation.development_empty_decision_labels import (
    empty_decision_pair, empty_reference_contract,
)
from bces.simulation.development_empty_view_input import DevelopmentEmptyViewPremise
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner


def message(timestamp):
    return ObjectMessage(timestamp, "development:source", timestamp, ())


def premise(packet):
    return DevelopmentEmptyViewPremise(
        hashlib.sha256(packet.encode()).hexdigest(), packet.sender_id,
        packet.timestamp_ms, "synthetic_ideal_conflict_view_v1", True)


def ego():
    return EgoState(x_m=104.8, y_m=60.0, speed_mps=14.0,
                    heading_rad=math.pi / 2)


def reference(planner):
    packet = message(1000)
    return empty_reference_contract(
        scenario="opened-dev-1", split="development", behavior="keep",
        timestamp=1000, ego=ego(), local=(), message=packet,
        empty_premise=premise(packet), planner=planner,
        scales=DriftScales(1, 1, 1, 1, 1, 1, 1))


def test_reference_is_bound_to_new_policy_and_true_empty_action():
    planner = DevelopmentStatelessEmptyPlanner()
    ref, query, cached = reference(planner)
    assert ref["policy_hash"] == query.policy_hash == planner.policy_hash
    assert ref["frozen_input"]["batch"]["identifiers"][2] == planner.policy_hash
    assert ref["decoded_status"] == "empty_complete_assumed"
    assert ref["policy_selection_sidecar"]["fresh_empty_used"]
    assert not ref["policy_selection_sidecar"]["stop_hold_fallback"]
    assert cached.payload_sha256 == ref["payload_sha256"]
    assert "future_world" not in str(ref)


def test_cached_expiry_and_fresh_empty_make_different_policy_matched_actions():
    planner = DevelopmentStatelessEmptyPlanner()
    ref, query, cached = reference(planner)
    fresh = message(1201)
    future = {1201 + 200 * index: () for index in range(17)}
    point = empty_decision_pair(
        reference=ref, query=query, cached_reference=cached, ego=ego(),
        local=(), fresh_message=fresh, fresh_empty_premise=premise(fresh),
        future_world=future, planner=planner,
        thresholds=ValidityThresholds(1.0, 1.0, 100.0),
    )
    assert point["policy_hash"] == planner.policy_hash
    assert point["cache_age_s"] == pytest.approx(0.201)
    assert point["cached_status"] == point["fresh_status"] == "empty_complete_assumed"
    assert point["cached_policy_selection_sidecar"]["stop_hold_fallback"]
    assert not point["cached_policy_selection_sidecar"]["candidate_margin_available"]
    assert point["fresh_policy_selection_sidecar"]["fresh_empty_used"]
    assert not point["fresh_policy_selection_sidecar"]["stop_hold_fallback"]
    assert point["trajectory_deviation_m"] > 0.0
    assert point["same_label_world"] and point["future_world_used_only_for_labels"]


def test_cached_reference_cannot_be_swapped_to_another_packet():
    planner = DevelopmentStatelessEmptyPlanner()
    ref, query, cached = reference(planner)
    altered = dict(ref, payload_sha256="0" * 64)
    with pytest.raises(ValueError, match="bound packet"):
        empty_decision_pair(
            reference=altered, query=query, cached_reference=cached,
            ego=ego(), local=(), fresh_message=message(1201),
            fresh_empty_premise=premise(message(1201)), future_world={},
            planner=planner, thresholds=ValidityThresholds(1, 1, 1),
        )
