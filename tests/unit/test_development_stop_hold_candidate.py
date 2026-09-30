import pytest

from bces.oracle.world import EgoState, Route
from bces.simulation.development_stop_hold_candidate import stop_hold_points


def _state(speed=8.0):
    ego = EgoState(x_m=0.0, y_m=0.0, speed_mps=speed, heading_rad=0.0)
    route = Route(origin_x_m=0.0, origin_y_m=0.0, heading_rad=0.0,
                  intended_behavior="keep")
    return ego, route


def test_brake_to_zero_and_hold_without_reverse_motion():
    ego, route = _state()
    points = stop_hold_points(ego, route, braking_mps2=-4.0,
                              horizon_s=3.0, step_s=0.2)
    assert len(points) == 15
    assert points[9].x_m == pytest.approx(8.0)
    assert points[-1].x_m == pytest.approx(8.0)
    assert points[-1].speed_mps == 0.0
    assert all(b.x_m >= a.x_m for a, b in zip(points, points[1:]))


def test_invalid_discretization_and_braking_rate_rejected():
    ego, route = _state()
    with pytest.raises(ValueError):
        stop_hold_points(ego, route, braking_mps2=0.0, horizon_s=3.0, step_s=0.2)
    with pytest.raises(ValueError):
        stop_hold_points(ego, route, braking_mps2=-4.0, horizon_s=3.0, step_s=0.17)
