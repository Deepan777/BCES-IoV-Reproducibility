from bces.oracle.world import EgoState, Route, WorldObject
from bces.simulation.development_early_yield_planner import DevelopmentEarlyYieldPlanner


def _ego(y, speed):
    return EgoState(x_m=104.8, y_m=y, speed_mps=speed, heading_rad=1.5707963267948966)


def _route():
    return Route(origin_x_m=104.8, origin_y_m=55.0,
                 heading_rad=1.5707963267948966, intended_behavior="keep")


def test_early_yield_latches_and_uses_stop_hold():
    planner = DevelopmentEarlyYieldPlanner()
    plan = planner.plan((), (), _ego(60.0, 14.0), _route())
    assert planner.yield_latched
    assert plan.acceleration_mps2 == -4.0
    assert plan.points[-1].speed_mps == 2.0
    continued = planner.plan((), (), _ego(84.0, 0.0), _route())
    assert planner.yield_latched
    assert continued.points[-1].speed_mps == 0.0


def test_observed_passed_actor_releases_yield():
    planner = DevelopmentEarlyYieldPlanner()
    planner.plan((), (), _ego(60.0, 14.0), _route())
    actor = WorldObject("vehicle:cross", 120.0, 95.2, 9.0, 0.0, 0.0, 5.0, 1.8)
    planner.plan((), (actor,), _ego(84.0, 0.0), _route())
    assert planner.actor_passed_observed
    assert not planner.yield_latched
