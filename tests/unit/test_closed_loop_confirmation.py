from copy import deepcopy

import pytest

from bces.evaluation.closed_loop_confirmation import (
    CONDITIONS, FAMILIES, METHODS, communication_radius, exact_upper, safety_power,
    scenario_record, summarize,
)


def _branch(seed, family, method, *, bytes_=100, progress=50., overlap=0):
    return {'scenario_id': str(seed), 'family': family, 'method': method,
            'initial_contract': {'snapshot': {'ego': 1}}, 'counts': {'geometric_overlap_ticks': overlap},
            'outcome': 'horizon_complete', 'actuation_contract_passed': True,
            'traffic': {'generated_bytes': bytes_, 'byte_conservation_ok': True,
                        'component_conservation_ok': True}, 'input_availability_audit': [],
            'duration_s': 6., 'route_distance_lower_bound_m': progress}


def _block(seed=1, family=FAMILIES[0]):
    return {(condition, method): _branch(seed, family, method, bytes_=40 if method == 'surface' else 100)
            for condition in CONDITIONS for method in METHODS}


def test_exact_safety_power_and_bounded_communication_radius():
    design = safety_power(1200, .01, .025, .002)
    assert design['maximum_passing_events'] == 5
    assert design['power_at_declared_alternative'] > .96
    assert exact_upper(5, 1200, .025) < .01 < exact_upper(6, 1200, .025)
    assert .11 < communication_radius(1200, .025) < .12


def test_complete_cluster_counts_one_scenario_not_eight_branches():
    block = _block()
    row = scenario_record(1, FAMILIES[0], block)
    assert row['engineering_ok'] and row['comparable'] and row['bounded_communication']
    assert row['mean_byte_reduction'] == .6
    assert not row['surface_adverse_episode']
    block[('impaired', 'surface')]['counts']['geometric_overlap_ticks'] = 1
    assert scenario_record(1, FAMILIES[0], block)['surface_adverse_episode']


def test_unfavorable_cases_are_not_removed_or_bounded_optimistically():
    block = _block()
    block[('nominal', 'surface')]['traffic']['generated_bytes'] = 400
    row = scenario_record(1, FAMILIES[0], block)
    assert not row['bounded_communication'] and row['mean_byte_reduction'] is None
    block = _block()
    block[('nominal', 'surface')]['route_distance_lower_bound_m'] = 44.9
    assert scenario_record(1, FAMILIES[0], block)['surface_adverse_episode']
    block = _block()
    block[('nominal', 'surface')]['actuation_contract_passed'] = False
    row = scenario_record(1, FAMILIES[0], block)
    assert not row['engineering_ok'] and row['surface_adverse_episode']
    block = _block()
    block[('nominal', 'surface')]['duration_s'] = 5.8
    assert not scenario_record(1, FAMILIES[0], block)['comparable']


def test_missing_branch_and_family_imbalance_fail_closed():
    block = _block()
    del block[('impaired', 'surface')]
    with pytest.raises(ValueError, match='incomplete'):
        scenario_record(1, FAMILIES[0], block)
    rows = [scenario_record(i, family, _block(i, family))
            for i, family in enumerate(FAMILIES, start=1)]
    result = summarize(rows, n_per_family=1)
    assert result['scenarios'] == 4 and not result['joint_success']
    with pytest.raises(ValueError, match='family balance'):
        summarize([*rows[:3], deepcopy(rows[0]) | {'seed': 5}], n_per_family=1)
