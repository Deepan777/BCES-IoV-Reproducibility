from bces.oracle.world import EgoState, Route
from bces.simulation.development_negative_evidence_planner import (
    DevelopmentNegativeEvidencePlanner, VerifiedEmptyObservation,
)


def _state():
    ego = EgoState(x_m=104.8, y_m=60.0, speed_mps=14.0,
                   heading_rad=1.5707963267948966)
    route = Route(origin_x_m=104.8, origin_y_m=60.0,
                  heading_rad=1.5707963267948966, intended_behavior="keep")
    return ego, route


def test_empty_message_distinguished_from_no_message():
    ego, route = _state()
    no_message = DevelopmentNegativeEvidencePlanner()
    no_message.plan((), (), ego, route)
    assert no_message.yield_latched
    verified = DevelopmentNegativeEvidencePlanner()
    verified.plan((), (VerifiedEmptyObservation(0.2),), ego, route)
    assert not verified.yield_latched
    assert verified.fresh_empty_uses == 1


def test_stale_empty_message_does_not_release_yield():
    ego, route = _state()
    planner = DevelopmentNegativeEvidencePlanner()
    planner.plan((), (VerifiedEmptyObservation(0.401),), ego, route)
    assert planner.yield_latched
    assert planner.fresh_empty_uses == 0


def test_negative_observation_age_propagates():
    marker = VerifiedEmptyObservation()
    assert marker.propagated(0.2).age_s == 0.2
