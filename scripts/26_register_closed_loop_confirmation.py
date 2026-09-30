#!/usr/bin/env python3
"""Freeze a balanced, independent corrected closed-loop study before outcomes."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import (
    CONDITIONS, FAMILIES, METHODS, communication_power_normal_approx,
    communication_radius, safety_power,
)
from bces.network.events import NetworkCondition
from bces.simulation.controlled_slots import development_generator
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.budget import Budget, require_headroom
from bces.utils.environment import _package_versions
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

OUTPUT = ROOT / 'outputs/study_b/closed_loop_confirmation_v1/preregistration.json'
SLOT = ROOT / 'outputs/study_b/slot_confirmation_v1/preregistration.json'
FINAL = ROOT / 'outputs/study_b/slot_confirmation_v1/confirmation/run_manifest.json'
PILOT = ROOT / 'outputs/study_b/controlled_closed_loop_compatible_multiseed_pilot_v1/audit.json'
N_PER_FAMILY = 300
SEED_START = 900000
SEED_SCAN_LIMIT = 20000


def select_seeds(sample, *, first=SEED_START, scan_limit=SEED_SCAN_LIMIT,
                 n_per_family=N_PER_FAMILY):
    """Stop when each family reaches its quota; observe no driving outcome."""
    counts = {family: 0 for family in FAMILIES}
    selected = []
    for seed in range(first, first + scan_limit):
        family = sample(seed)
        if family in counts and counts[family] < n_per_family:
            counts[family] += 1
            selected.append({'seed': seed, 'family': family})
        if all(value == n_per_family for value in counts.values()):
            return selected
    raise RuntimeError('fixed seed scan did not fill every family without replacement')


def main():
    state = git_state(ROOT)
    if not state['available'] or state['dirty']:
        raise RuntimeError('clean Git source required before preregistration')
    if OUTPUT.exists():
        raise FileExistsError('closed-loop preregistration is immutable')
    require_headroom(Budget(workspace_root=ROOT, data_root=ROOT/'data'), incoming_bytes=200_000_000)
    slot = json.loads(SLOT.read_text(encoding='utf-8'))
    final = json.loads(FINAL.read_text(encoding='utf-8'))
    if (final['status'] != 'COMPLETE' or final['registration_sha256'] != sha256_file(SLOT)
            or not final['results']['both_risk_gates_passed']
            or not final['results']['paired_adverse_superiority']
            or not final['results']['coverage_equivalence']):
        raise RuntimeError('registered independent open-loop gate not passed')
    for name, expected in final['artifact_sha256'].items():
        if sha256_file(FINAL.parent/name) != expected:
            raise RuntimeError('independent open-loop artifact changed: '+name)
    pilot = json.loads(PILOT.read_text(encoding='utf-8'))
    if (pilot['status'] != 'PASS' or len(pilot['cases']) != 160
            or not pilot['all_prefix_checks_passed']
            or not pilot['all_byte_and_input_availability_checks_passed']
            or not all(row['actuation_contract_passed'] for row in pilot['cases'])):
        raise RuntimeError('corrected compatible-family development pilot is not intact')
    for name, expected in pilot['artifact_sha256'].items():
        if sha256_file(PILOT.parent/name) != expected:
            raise RuntimeError('development pilot artifact changed: '+name)
    config = slot['config']
    generator = development_generator()
    selected = select_seeds(lambda seed: generator.sample_scenario(seed, config)[0].family)
    if len(selected) != len(FAMILIES) * N_PER_FAMILY:
        raise AssertionError('unbalanced seed registration')
    nominal = NetworkCondition(50., 0., 10.)
    impaired = NetworkCondition(100., .1, 1., jitter_ms=20.,
                                duplicate_probability=.05, reorder_probability=.05)
    versions, environment_hash = _package_versions()
    sources = [*sorted((ROOT/'bces').rglob('*.py')),
               ROOT/'scripts/16_generate_controlled_development.py',
               ROOT/'scripts/26_register_closed_loop_confirmation.py',
               ROOT/'scripts/27_run_closed_loop_confirmation.py']
    assets = [ROOT/config['network'], ROOT/config['planner'],
              ROOT/'configs/evaluation/study_b_controlled_v2.yaml', SLOT, FINAL, PILOT]
    assets.extend(ROOT/rule['path'] for rule in slot['operating_rules'].values())
    registration = {
        'schema_version': 1, 'status': 'FROZEN_BEFORE_FRESH_OUTCOMES',
        'created_utc': utc_now(), 'git': state,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources},
        'asset_sha256': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in assets},
        'runtime': {'package_versions': versions, 'python_environment_hash': environment_hash,
                    'sumo_version': sumo_version()},
        'slot_registration_sha256': sha256_file(SLOT),
        'slot_confirmation_manifest_sha256': sha256_file(FINAL),
        'development_pilot_sha256': sha256_file(PILOT),
        'scenario_sampling': {'candidate_start': SEED_START, 'scan_limit': SEED_SCAN_LIMIT,
                              'n_per_family': N_PER_FAMILY, 'families': list(FAMILIES),
                              'selected': selected, 'selected_sha256': canonical_json_hash(selected),
                              'selection': 'first outcome-blind generator family assignments to reach each quota'},
        'conditions': {'nominal': asdict(nominal), 'impaired': asdict(impaired)},
        'condition_order': list(CONDITIONS), 'methods': list(METHODS),
        'model_rules': {method: {key: rule[key] for key in ('path','sha256','shrinkage')}
                        for method, rule in slot['operating_rules'].items()},
        'policy_hash': slot['policy_hash'], 'config': config,
        'horizon_s': 6., 'step_s': config['step_s'],
        'periodic_payload_period_s': config['step_s'],
        'sender_computation_delay_ms': 6., 'request_timeout_ms': 400.,
        'maximum_retries': 1, 'modeled_security_envelope_bytes': 64,
        'tracking_review_thresholds': {'position_m': .5, 'speed_mps': .5, 'heading_rad': .1},
        'endpoints': {'independent_unit': 'scenario across two network conditions',
                      'adverse_progress_deficit_m': 5., 'adverse_rate_margin': .01,
                      'communication_minimum_mean_reduction': .25,
                      'communication_per_condition_bounds': [-2., 1.],
                      'alpha_each_one_sided': .025,
                      'adverse_gate': 'one-sided exact Clopper-Pearson upper strictly below margin',
                      'communication_gate': 'one-sided Hoeffding lower strictly above minimum',
                      'engineering_gate': 'all branches causal, byte-conserving, prefix-equal and actuation-valid',
                      'comparability_gate': 'surface and periodic have equal duration and outcome in every condition',
                      'failure_rule': 'any missing, corrupt, noncomparable or out-of-bounds branch prevents joint success'},
        'power': {'adverse_alternative_rate': .002,
                  **safety_power(len(selected), .01, .025, .002),
                  'communication_alternative_mean': .4,
                  'communication_alternative_sd': .7,
                  'communication_hoeffding_radius': communication_radius(len(selected), .025),
                  'communication_normal_approx_planning_power':
                      communication_power_normal_approx(len(selected), .25, .025, .4, .7),
                  'conditional_on_declared_alternatives': True},
        'stopping': 'fixed 1200 scenarios; no efficacy interim looks, replacement, extension or family removal',
        'original_test_accessed': False, 'manuscript_allowed': False,
        'scope': 'four-family designed SUMO mixture; modeled network overhead; no public-road safety claim',
    }
    write_json_atomic(OUTPUT, registration)
    print(json.dumps({'status': registration['status'], 'selected_scenarios': len(selected),
                      'seed_list_sha256': registration['scenario_sampling']['selected_sha256'],
                      'registration_sha256': sha256_file(OUTPUT), 'power': registration['power']}, indent=2))


if __name__ == '__main__':
    main()
