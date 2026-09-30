from __future__ import annotations

import pytest

from bces.oracle.boundary import (
    BoundaryConfig,
    OraclePoint,
    audit_boundary,
    search_boundary,
)

pytestmark = pytest.mark.phase4
AXIS = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def point(valid, feasible=True):
    return OraclePoint(valid, feasible, "analytic-world-v1")


def test_boundary_is_measured_by_bracketing_and_bisection():
    oracle = lambda z: point(z[0] <= 0.637)
    result = search_boundary(oracle, AXIS, BoundaryConfig())
    assert result.has_regression_target
    assert result.valid_radius <= 0.637 < result.invalid_radius
    assert result.invalid_radius - result.valid_radius <= 0.001
    assert result.probes[0].normalized_drift == (0.0,) * 7
    assert audit_boundary(result, oracle)["passed"]
    assert not audit_boundary(result, lambda z: point(True))["passed"]


def test_no_transition_is_censored_not_a_measured_boundary():
    result = search_boundary(lambda z: point(True), AXIS, BoundaryConfig(max_radius=0.45))
    assert result.status == "radius_limit" and result.censored
    assert result.valid_radius == 0.45
    assert result.invalid_radius is None and not result.has_regression_target


def test_invalid_origin_cannot_be_encoded_as_a_positive_surface():
    result = search_boundary(lambda z: point(False), AXIS, BoundaryConfig())
    assert result.status == "origin_invalid"
    assert result.valid_radius is None and not result.has_regression_target
    assert len(result.probes) == 1


def test_feasibility_is_distinct_from_invalidity_and_is_refined():
    result = search_boundary(lambda z: point(True) if z[0] <= 0.637 else point(None, False), AXIS, BoundaryConfig())
    assert result.status == "feasibility_limit" and result.censored
    assert 0.636 <= result.valid_radius <= 0.637
    assert result.invalid_radius is None


def test_unavailable_evidence_never_yields_a_target_or_censoring():
    result = search_boundary(lambda z: point(True) if z[0] < 0.1 else point(None), AXIS, BoundaryConfig())
    assert result.status == "missing_evidence"
    assert not result.has_regression_target and not result.censored


def test_first_sampled_invalid_region_is_not_skipped_for_later_validity():
    result = search_boundary(lambda z: point(not 0.3 <= z[0] <= 0.6), AXIS, BoundaryConfig())
    assert result.valid_radius < 0.3 <= result.invalid_radius


def test_negative_time_direction_is_feasibility_censored():
    direction = (0.0,) * 6 + (-1.0,)
    result = search_boundary(lambda z: point(True) if z[6] >= 0 else point(None, False), direction, BoundaryConfig())
    assert result.status == "feasibility_limit" and result.valid_radius == 0.0


def test_boundary_refuses_unbounded_work_and_nonunit_directions():
    with pytest.raises(RuntimeError, match="query budget"):
        search_boundary(lambda z: point(True), AXIS, BoundaryConfig(max_queries=2))
    with pytest.raises(ValueError, match="unit"):
        search_boundary(lambda z: point(True), (0.0,) * 7, BoundaryConfig())
