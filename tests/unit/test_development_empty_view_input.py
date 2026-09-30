import hashlib
import math

import pytest

from bces.data.objects import ObjectMessage, ObjectSource, PerceivedObject
from bces.oracle.world import EgoState, Route
from bces.simulation.development_empty_view_input import (
    DevelopmentEmptyViewPremise, decode_policy_input,
)
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner


def empty_message():
    return ObjectMessage(1, "development:source", 1000, ())


def premise(message, *, complete=True):
    return DevelopmentEmptyViewPremise(
        hashlib.sha256(message.encode()).hexdigest(),
        message.sender_id, message.timestamp_ms,
        "synthetic_ideal_conflict_view_v1", complete,
    )


def crossing_state():
    ego = EgoState(x_m=104.8, y_m=60.0, speed_mps=14.0,
                   heading_rad=math.pi / 2)
    route = Route(origin_x_m=104.8, origin_y_m=60.0,
                  heading_rad=math.pi / 2, intended_behavior="keep")
    return ego, route


def test_no_packet_and_unverified_empty_fail_closed_but_bound_fresh_empty_releases():
    message = empty_message()
    no_packet = decode_policy_input(None, now_ms=1200)
    unverified = decode_policy_input(message, now_ms=1200)
    incomplete = decode_policy_input(message, now_ms=1200,
                                     empty_premise=premise(message, complete=False))
    fresh = decode_policy_input(message, now_ms=1200,
                                empty_premise=premise(message))
    assert [row.status for row in (no_packet, unverified, incomplete, fresh)] == [
        "no_message", "empty_unverified", "empty_incomplete",
        "empty_complete_assumed",
    ]
    assert not no_packet.objects and not unverified.objects and not incomplete.objects
    assert len(fresh.objects) == 1 and fresh.age_s == 0.2
    ego, route = crossing_state()
    planner = DevelopmentStatelessEmptyPlanner()
    hold = planner.plan((), no_packet.objects, ego, route)
    assert planner.plan((), unverified.objects, ego, route) == hold
    assert planner.plan((), incomplete.objects, ego, route) == hold
    assert planner.plan((), fresh.objects, ego, route).points[-1].y_m > hold.points[-1].y_m


def test_exact_age_expiry_is_not_a_new_observation():
    message = empty_message()
    stale = decode_policy_input(message, now_ms=1201,
                                empty_premise=premise(message))
    assert stale.age_s == pytest.approx(0.201)
    ego, route = crossing_state()
    planner = DevelopmentStatelessEmptyPlanner()
    assert planner.plan((), stale.objects, ego, route) == planner.plan((), (), ego, route)


def test_premise_must_bind_exact_empty_packet():
    message = empty_message()
    other = ObjectMessage(2, message.sender_id, message.timestamp_ms, ())
    with pytest.raises(ValueError, match="exact packet"):
        decode_policy_input(other, now_ms=1000, empty_premise=premise(message))
    with pytest.raises(ValueError, match="absent packet"):
        decode_policy_input(None, now_ms=1000, empty_premise=premise(message))


def test_nonempty_payload_never_becomes_empty_evidence():
    actor = PerceivedObject("vehicle:a", 0, 104.8, 95.2, 1.0, 0.0,
                            5.0, 1.8, 1.0, ObjectSource.INFRASTRUCTURE)
    message = ObjectMessage(3, "development:source", 1000, (actor,))
    decoded = decode_policy_input(message, now_ms=1100)
    assert decoded.status == "objects"
    assert len(decoded.objects) == 1 and decoded.objects[0].track_id == actor.track_id
    with pytest.raises(ValueError, match="nonempty packet"):
        decode_policy_input(message, now_ms=1100,
                            empty_premise=premise(empty_message()))
