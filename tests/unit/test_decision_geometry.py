import itertools
import numpy as np
import pytest

from bces.evaluation.decision_geometry import conservative_box, fit_zero_error_surface, fit_zero_error_ttl, ray_radius, regret_upper_bound
from bces.oracle.causal_diagnostics import causal_kinematic_states, observable_planner_features
from bces.oracle.planner import KinematicPlanner
from bces.oracle.world import EgoState, Route
from bces.oracle.vectorized_planner import VectorizedDiagnosticPlanner
from bces.oracle.world import WorldObject, GroundTruthWorld


def test_regret_bound_covers_all_error_corners_even_nonminimum_selected():
    costs, errors = np.array([3., 1., 5.]), np.array([.3, .7, 2.])
    for selected in range(3):
        bound = regret_upper_bound(costs, errors, selected)
        for signs in itertools.product([-1, 1], repeat=3):
            actual = costs + errors*np.array(signs)
            assert actual[selected]-actual.min() <= bound+1e-12
    assert regret_upper_bound(costs, np.zeros(3), 0) == 2


def test_conservative_box_satisfies_uniform_envelope_at_corners():
    costs, base = np.array([0., 2., 3.]), np.array([.1, .2, .1])
    slopes = np.array([[1., 2.], [2., .5], [1., 3.]])
    result = conservative_box(costs, base, slopes, 0, .5, [2., 2.])
    assert not result['abstain']
    for signs in itertools.product([-1, 1], repeat=2):
        drift = np.array(result['widths'])*signs
        assert regret_upper_bound(costs, base+slopes@abs(drift), 0) <= .5+1e-9
    assert conservative_box([3, 1], [0, 0], [[1], [1]], 0, 1, [1])['abstain']
    with pytest.raises(ValueError):
        regret_upper_bound([0], [-1], 0)


def test_surface_separates_equal_age_points_that_ttl_cannot():
    normals = np.array([[1,0],[-1,0],[0,1],[0,-1]], float)
    drift = np.array([[.1,.2],[.8,.2],[.15,.3]])
    valid = np.array([True,False,True])
    surface = fit_zero_error_surface(drift, valid, normals)
    ttl = fit_zero_error_ttl(drift[:,1], valid)
    assert surface['optimal'] and sum(surface['accepted']) == 2
    assert sum(ttl['accepted']) == 0


def test_invalid_origin_precludes_positive_offset_surface():
    result = fit_zero_error_surface([[0],[.5]], [False,True], [[1],[-1]])
    assert result['abstain'] and not any(result['accepted'])


def test_ray_radius_is_not_assigned_plane_offset():
    normals = np.array([[1.,0.],[0.,1.],[1/np.sqrt(2),1/np.sqrt(2)]])
    assert ray_radius([1.,1.,.2], normals, [1.,0.]) == pytest.approx(.2*np.sqrt(2))


def test_causal_estimates_invariant_to_future_rows():
    rows = {1000:{'x':'0','y':'0','v_x':'4','v_y':'0'}, 1100:{'x':'.4','y':'0','v_x':'5','v_y':'0'}}
    prefix = causal_kinematic_states(rows)
    rows[1200] = {'x':'99','y':'9','v_x':'19','v_y':'7'}
    extended = causal_kinematic_states(rows)
    assert all(prefix[t] == extended[t] for t in prefix)


def test_margin_features_select_same_action_as_frozen_planner():
    planner = KinematicPlanner()
    ego, route = EgoState(0,0,10,0), Route(0,0,0)
    features = observable_planner_features(planner, (), (), ego, route)
    assert len(features) == 29
    assert features[3] == planner.plan((), (), ego, route).acceleration_mps2


def test_vectorized_cost_and_choice_agree_with_scalar_reference():
    rng = np.random.default_rng(717)
    scalar, vector = KinematicPlanner(), VectorizedDiagnosticPlanner()
    assert scalar.policy_hash != vector.policy_hash
    for index in range(30):
        ego = EgoState(1,2,10,float(rng.uniform(-3,3)))
        route = Route(1,2,ego.heading_rad,intended_behavior=['keep','left','right'][index%3],curvature_inv_m=float(rng.uniform(-.05,.05)))
        objects = tuple(WorldObject(str(i),float(rng.uniform(-20,50)),float(rng.uniform(-20,20)),
            float(rng.uniform(-10,10)),float(rng.uniform(-5,5)),float(rng.uniform(-3,3)),4.5,1.8) for i in range(index%12))
        first, second = scalar.plan((),objects,ego,route), vector.plan((),objects,ego,route)
        assert first.acceleration_mps2 == second.acceleration_mps2
        assert first.maneuver == second.maneuver
        for item in (first,second):
            a = scalar.evaluate(item,GroundTruthWorld(0,objects),ego,route)
            b = vector.evaluate(item,GroundTruthWorld(0,objects),ego,route)
            assert a.total == pytest.approx(b.total,abs=1e-10)
            assert a.risk == pytest.approx(b.risk,abs=1e-12)
