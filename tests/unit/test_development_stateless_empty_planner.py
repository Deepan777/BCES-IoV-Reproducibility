import pytest

from bces.oracle.world import EgoState, Route, WorldObject
from bces.simulation.development_negative_evidence_planner import VerifiedEmptyObservation
from bces.simulation.development_stateless_empty_planner import DevelopmentStatelessEmptyPlanner


def state():
    ego = EgoState(x_m=104.8, y_m=60.0, speed_mps=14.0,
                   heading_rad=1.5707963267948966)
    route = Route(origin_x_m=104.8, origin_y_m=60.0,
                  heading_rad=1.5707963267948966, intended_behavior="keep")
    return ego, route


def test_fresh_empty_releases_stateless_fallback_and_short_age_is_binding():
    ego, route = state()
    planner = DevelopmentStatelessEmptyPlanner()
    no_view = planner.plan((), (), ego, route)
    stale_view = planner.plan((), (VerifiedEmptyObservation(0.201),), ego, route)
    fresh_view = planner.plan((), (VerifiedEmptyObservation(0.2),), ego, route)
    assert no_view.acceleration_mps2 == stale_view.acceleration_mps2 == -4.0
    assert fresh_view.points[-1].y_m > no_view.points[-1].y_m
    assert planner.fallback_uses == 2
    assert planner.fresh_empty_uses == 1


def test_previous_call_does_not_change_next_action():
    ego, route = state()
    reused = DevelopmentStatelessEmptyPlanner()
    reused.plan((), (), ego, route)
    after_fallback = reused.plan((), (VerifiedEmptyObservation(0.2),), ego, route)
    fresh_instance = DevelopmentStatelessEmptyPlanner().plan(
        (), (VerifiedEmptyObservation(0.2),), ego, route)
    assert after_fallback == fresh_instance


def test_contradictory_empty_marker_is_rejected():
    ego, route = state()
    actor = WorldObject("vehicle:cross", 104.8, 95.2, 0.0, 0.0, 0.0, 5.0, 1.8)
    with pytest.raises(ValueError, match="contradicts"):
        DevelopmentStatelessEmptyPlanner().plan(
            (), (VerifiedEmptyObservation(0.0), actor), ego, route)
