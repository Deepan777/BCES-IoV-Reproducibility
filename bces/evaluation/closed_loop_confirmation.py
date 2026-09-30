"""Prospective closed-loop endpoints on independent scenario clusters.

One scenario contains both network conditions and all four methods. These
functions never treat frames or the two conditions as independent samples.
"""
from __future__ import annotations

import math
from statistics import mean

from scipy.stats import beta, binom, norm

from bces.simulation.controlled_contract import compare_snapshots

METHODS = ('local_only', 'periodic_payload', 'surface', 'scalar_ttl')
CONDITIONS = ('nominal', 'impaired')
FAMILIES = ('straight_lead_braking', 'on_ramp_merge', 'unprotected_crossing', 'pedestrian_crossing')


def exact_upper(k: int, n: int, alpha: float) -> float:
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0):
        raise ValueError('invalid binomial count')
    if not 0 < alpha < 1:
        raise ValueError('invalid alpha')
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def safety_power(n: int, margin: float, alpha: float, alternative_rate: float) -> dict:
    maximum = max((k for k in range(n + 1) if exact_upper(k, n, alpha) < margin), default=-1)
    return {'maximum_passing_events': maximum,
            'power_at_declared_alternative': float(binom.cdf(maximum, n, alternative_rate))}


def communication_radius(n: int, alpha: float, lower_bound: float = -2., upper_bound: float = 1.) -> float:
    if n <= 0 or not 0 < alpha < 1 or lower_bound >= upper_bound:
        raise ValueError('invalid Hoeffding design')
    return (upper_bound - lower_bound) * math.sqrt(math.log(1 / alpha) / (2 * n))


def communication_power_normal_approx(n: int, threshold: float, alpha: float,
                                       alternative_mean: float, alternative_sd: float) -> float:
    """Planning calculation only; the actual gate uses Hoeffding, not normality."""
    if alternative_sd <= 0:
        raise ValueError('positive planning standard deviation required')
    required = threshold + communication_radius(n, alpha)
    return float(norm.sf((required - alternative_mean) * math.sqrt(n) / alternative_sd))


def _severe(branch: dict) -> bool:
    counts = branch['counts']
    return (not branch['actuation_contract_passed']
            or branch['outcome'] in ('planner_infeasible', 'receiver_disappeared')
            or counts.get('geometric_overlap_ticks', 0) > 0
            or counts.get('sumo_ego_collision_ticks', 0) > 0)


def scenario_record(seed: int, family: str, branches: dict, *, progress_margin_m: float = 5.) -> dict:
    """Summarize a complete eight-branch block without losing unfavorable cases."""
    required = {(condition, method) for condition in CONDITIONS for method in METHODS}
    if set(branches) != required or family not in FAMILIES or progress_margin_m <= 0:
        raise ValueError('incomplete or invalid registered scenario block')
    reference = branches[(CONDITIONS[0], METHODS[0])]['initial_contract']
    prefix_equal = True
    engineering_ok = True
    comparable = True
    severe_surface = False
    severe_periodic = False
    progress_deltas = []
    reductions = []
    for condition in CONDITIONS:
        for method in METHODS:
            branch = branches[(condition, method)]
            if int(branch['scenario_id']) != seed or branch['family'] != family or branch['method'] != method:
                raise ValueError('branch identity mismatch')
            if not compare_snapshots(reference, branch['initial_contract'], tolerance=1e-6)['equal_within_tolerance']:
                prefix_equal = False
            traffic = branch['traffic']
            if not traffic['byte_conservation_ok'] or not traffic['component_conservation_ok']:
                engineering_ok = False
            if any(row['used_timestamp_ms'] > row['available_ms'] for row in branch['input_availability_audit']):
                engineering_ok = False
            if not branch['actuation_contract_passed']:
                engineering_ok = False
        surface = branches[(condition, 'surface')]
        periodic = branches[(condition, 'periodic_payload')]
        if surface['duration_s'] != periodic['duration_s'] or surface['outcome'] != periodic['outcome']:
            comparable = False
        if periodic['traffic']['generated_bytes'] <= 0:
            comparable = False
            reductions.append(None)
        else:
            reductions.append(1 - surface['traffic']['generated_bytes'] / periodic['traffic']['generated_bytes'])
        progress_delta = surface['route_distance_lower_bound_m'] - periodic['route_distance_lower_bound_m']
        progress_deltas.append(progress_delta)
        severe_surface |= _severe(surface) or progress_delta < -progress_margin_m
        severe_periodic |= _severe(periodic)
    if not prefix_equal:
        engineering_ok = False
    bounded = all(value is not None and -2 <= value <= 1 for value in reductions)
    return {'seed': seed, 'family': family, 'engineering_ok': engineering_ok,
            'prefix_equal': prefix_equal, 'comparable': comparable,
            'bounded_communication': bounded, 'surface_adverse_episode': severe_surface,
            'periodic_adverse_episode': severe_periodic,
            'progress_deltas_m': progress_deltas, 'byte_reduction_by_condition': reductions,
            'mean_byte_reduction': mean(reductions) if bounded else None}


def summarize(records: list[dict], *, n_per_family: int, adverse_margin: float = .01,
              byte_threshold: float = .25, alpha_each: float = .025) -> dict:
    n = len(FAMILIES) * n_per_family
    if len(records) != n or len({row['seed'] for row in records}) != n:
        raise ValueError('fixed independent scenario count/identity mismatch')
    if any(sum(row['family'] == family for row in records) != n_per_family for family in FAMILIES):
        raise ValueError('family balance mismatch')
    k = sum(row['surface_adverse_episode'] for row in records)
    upper = exact_upper(k, n, alpha_each)
    engineering = all(row['engineering_ok'] for row in records)
    comparable = all(row['comparable'] for row in records)
    bounded = all(row['bounded_communication'] for row in records)
    average = mean(row['mean_byte_reduction'] for row in records) if bounded else None
    radius = communication_radius(n, alpha_each)
    lower = average - radius if average is not None else None
    risk_pass = upper < adverse_margin
    byte_pass = engineering and comparable and bounded and lower > byte_threshold
    return {'scenarios': n, 'n_per_family': n_per_family,
            'surface_adverse_episodes': k,
            'periodic_adverse_episodes': sum(row['periodic_adverse_episode'] for row in records),
            'adverse_rate': k / n, 'simultaneous_exact_adverse_upper': upper,
            'adverse_margin': adverse_margin, 'adverse_gate_passed': risk_pass,
            'mean_byte_reduction': average, 'hoeffding_radius': radius,
            'hoeffding_lower': lower, 'byte_threshold': byte_threshold,
            'byte_gate_passed': byte_pass, 'engineering_gate_passed': engineering,
            'all_outcomes_and_durations_comparable': comparable,
            'all_reductions_within_frozen_bounds': bounded,
            'joint_success': risk_pass and byte_pass,
            'public_road_safety_established': False}
