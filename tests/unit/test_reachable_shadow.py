"""Conditional geometry only; these tests do not certify empirical bounds."""

import math

import pytest

from bces.geometry.reachable_shadow import ReachabilityAssumptions, reachable_shadow
from bces.oracle.world import WorldObject


def object_at_reference():
    return WorldObject("vehicle:remote", 12.0, -3.0, 6.0, 2.0, 0.3,
                       4.8, 2.0, source="received_payload")


def test_radius_follows_integrated_acceleration_bound():
    bounds = ReachabilityAssumptions(0.4, 0.2, 3.0)
    assert bounds.center_error_radius_m(2.0) == pytest.approx(0.4 + 0.4 + 6.0)
    shadow = reachable_shadow(object_at_reference(), age_s=0.5, horizon_s=1.5,
                              assumptions=bounds)
    half_diagonal = 0.5 * math.hypot(4.8, 2.0)
    assert shadow.length_m == pytest.approx(2 * (6.8 + half_diagonal))
    assert shadow.width_m == shadow.length_m
    assert shadow.x_m == pytest.approx(15.0)
    assert shadow.y_m == pytest.approx(-2.0)
    assert shadow.track_id == "vehicle:remote"


@pytest.mark.parametrize("age_s,horizon_s", [(0.0, 0.0), (0.6, 1.0), (1.7, 3.0)])
def test_square_contains_rotating_actor_corners_under_stated_bounds(age_s, horizon_s):
    obj = object_at_reference()
    bounds = ReachabilityAssumptions(0.3, 0.5, 2.4)
    shadow = reachable_shadow(obj, age_s=age_s, horizon_s=horizon_s, assumptions=bounds)
    for s in (0.0, 0.25 * horizon_s, horizon_s):
        elapsed = age_s + s
        propagated = shadow.propagated(s)
        for angle in (0.0, 0.6, 1.8, 3.0):
            ux, uy = math.cos(angle), math.sin(angle)
            true_cx = (obj.x_m + obj.vx_mps * elapsed
                       + ux * (bounds.position_error_m
                               + bounds.velocity_error_mps * elapsed
                               + 0.5 * bounds.acceleration_bound_mps2 * elapsed**2))
            true_cy = (obj.y_m + obj.vy_mps * elapsed
                       + uy * (bounds.position_error_m
                               + bounds.velocity_error_mps * elapsed
                               + 0.5 * bounds.acceleration_bound_mps2 * elapsed**2))
            for heading in (0.0, 0.7, 1.9):
                fx, fy = math.cos(heading), math.sin(heading)
                sx, sy = -fy, fx
                for front in (-1, 1):
                    for side in (-1, 1):
                        x = true_cx + front * obj.length_m / 2 * fx + side * obj.width_m / 2 * sx
                        y = true_cy + front * obj.length_m / 2 * fy + side * obj.width_m / 2 * sy
                        assert abs(x - propagated.x_m) <= propagated.length_m / 2 + 1e-10
                        assert abs(y - propagated.y_m) <= propagated.width_m / 2 + 1e-10


@pytest.mark.parametrize("bad", [-0.1, math.inf, math.nan])
def test_rejects_invalid_assumptions_or_times(bad):
    with pytest.raises(ValueError):
        ReachabilityAssumptions(bad, 0.0, 0.0)
    valid = ReachabilityAssumptions(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        reachable_shadow(object_at_reference(), age_s=bad, horizon_s=1.0,
                         assumptions=valid)
    with pytest.raises(ValueError):
        reachable_shadow(object_at_reference(), age_s=0.0, horizon_s=bad,
                         assumptions=valid)
