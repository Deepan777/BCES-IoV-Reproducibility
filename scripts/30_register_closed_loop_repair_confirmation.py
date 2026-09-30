#!/usr/bin/env python3
"""Freeze v2 repair confirmation before any fresh outcome is generated."""
from __future__ import annotations

import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.evaluation.closed_loop_confirmation import (
    CONDITIONS, FAMILIES, METHODS, communication_power_normal_approx,
    communication_radius, safety_power,
)
from bces.network.events import NetworkCondition
from bces.simulation.controlled_closed_loop import EGO_LANE_CHANGE_MODE
from bces.simulation.controlled_slots import development_generator
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.budget import Budget, require_headroom
from bces.utils.environment import _package_versions
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

OUTPUT = ROOT / 'outputs/study_b/closed_loop_confirmation_v2/preregistration.json'
SLOT = ROOT / 'outputs/study_b/slot_confirmation_v1/preregistration.json'
OPEN_LOOP = ROOT / 'outputs/study_b/slot_confirmation_v1/confirmation/run_manifest.json'
FAILED_V1 = ROOT / 'outputs/study_b/closed_loop_confirmation_v1/confirmation/run_manifest.json'
PLAN = ROOT / 'docs/STUDY_B_REPAIR_AND_ANALYSIS_PLAN_V2.md'
PILOT = ROOT / 'outputs/study_b/controlled_closed_loop_repaired_pilot_v2/audit.json'
DEV = ROOT / 'outputs/study_b/lane_actuation_development_v1/result.json'
DEBUG = ROOT / 'outputs/study_b/lane_mode_512_debug_v1/result.json'
PREMISES = ROOT / 'outputs/study_b/regret_premise_audit_v1/result.json'
N_PER_FAMILY = 300
SEED_START = 1_000_000
SEED_SCAN_LIMIT = 20_000


def select_seeds(sample):
    counts = {family: 0 for family in FAMILIES}
    selected = []
    for seed in range(SEED_START, SEED_START + SEED_SCAN_LIMIT):
        family = sample(seed)
        if family in counts and counts[family] < N_PER_FAMILY:
            counts[family] += 1
            selected.append({'seed': seed, 'family': family})
        if all(value == N_PER_FAMILY for value in counts.values()):
            return selected
    raise RuntimeError('outcome-blind fixed seed scan did not fill every family')


def verify_artifacts(manifest_path, folder, *, expected_status):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != expected_status:
        raise RuntimeError('prerequisite status not met: ' + str(manifest_path))
    for name, digest in manifest['artifact_sha256'].items():
        if sha256_file(folder / name) != digest:
            raise RuntimeError('prerequisite artifact changed: ' + name)
    return manifest


def main() -> None:
    state = git_state(ROOT)
    if not state['available'] or state['dirty']:
        raise RuntimeError('clean committed source required before v2 preregistration')
    if OUTPUT.exists():
        raise FileExistsError('v2 preregistration is immutable')
    if EGO_LANE_CHANGE_MODE != 512:
        raise RuntimeError('safety-preserving ego lane mode 512 required')
    require_headroom(Budget(workspace_root=ROOT, data_root=ROOT / 'data'), incoming_bytes=200_000_000)
    slot = json.loads(SLOT.read_text(encoding='utf-8'))
    open_loop = verify_artifacts(OPEN_LOOP, OPEN_LOOP.parent, expected_status='COMPLETE')
    if (open_loop['registration_sha256'] != sha256_file(SLOT)
            or not open_loop['results']['both_risk_gates_passed']
            or not open_loop['results']['paired_adverse_superiority']
            or not open_loop['results']['coverage_equivalence']):
        raise RuntimeError('independent open-loop prerequisite not passed')
    failed = json.loads(FAILED_V1.read_text(encoding='utf-8'))
    if failed['status'] != 'COMPLETE' or failed['results']['joint_success']:
        raise RuntimeError('v1 failed result must remain disclosed')
    dev = verify_artifacts(DEV, DEV.parent, expected_status='COMPLETE')
    if dev['pairs'] != 300 or dev['baseline_failed'] < 1 or dev['repaired_failed'] != 0:
        raise RuntimeError('development diagnosis not preserved')
    debug = verify_artifacts(DEBUG, DEBUG.parent, expected_status='COMPLETE')
    if debug['branches'] != 16 or debug['failed_actuation'] != 0:
        raise RuntimeError('mode-512 debug replay not preserved')
    premises = verify_artifacts(PREMISES, PREMISES.parent, expected_status='COMPLETE')
    if (premises['scenes'] != 80 or premises['uniform_envelope_proved']
            or premises['surface_containment_proved']):
        raise RuntimeError('conditional-bound premise audit not preserved')
    pilot = verify_artifacts(PILOT, PILOT.parent, expected_status='PASS')
    if (len(pilot['cases']) != 640 or not pilot['all_prefix_checks_passed']
            or not pilot['all_byte_and_input_availability_checks_passed']
            or not all(row['actuation_contract_passed'] for row in pilot['cases'])):
        raise RuntimeError('repaired all-family development pilot failed')
    for name in pilot['artifact_sha256']:
        with gzip.open(PILOT.parent / name, 'rt', encoding='utf-8') as handle:
            branch = json.load(handle)
        if branch['initial_contract']['snapshot']['vehicles']['ego']['lane_change_mode'] != 512:
            raise RuntimeError('pilot branch retained autonomous ego lane changes: ' + name)
    config = slot['config']
    selected = select_seeds(lambda seed: development_generator().sample_scenario(seed, config)[0].family)
    if any(row['seed'] < SEED_START for row in selected):
        raise AssertionError('old failure/development seed in fresh sampling frame')
    nominal = NetworkCondition(50., 0., 10.)
    impaired = NetworkCondition(100., .1, 1., jitter_ms=20.,
                                duplicate_probability=.05, reorder_probability=.05)
    versions, environment_hash = _package_versions()
    sources = [*sorted((ROOT / 'bces').rglob('*.py')),
               *(ROOT / 'scripts' / name for name in (
                   '16_generate_controlled_development.py', '24_audit_controlled_closed_loop.py',
                   '27_run_closed_loop_confirmation.py', '28_diagnose_lane_actuation.py',
                   '29_validate_lane_mode.py', '30_register_closed_loop_repair_confirmation.py',
                   '31_audit_regret_premises.py', '32_analyze_closed_loop_v2.py')),
               ROOT / 'tests/unit/test_controlled_closed_loop.py']
    assets = [ROOT / config['network'], ROOT / config['planner'],
              ROOT / 'configs/evaluation/study_b_controlled_v2.yaml',
              SLOT, OPEN_LOOP, FAILED_V1, PLAN, PILOT, DEV, DEBUG, PREMISES]
    assets.extend(ROOT / rule['path'] for rule in slot['operating_rules'].values())
    registration = {
        'schema_version': 2, 'status': 'FROZEN_BEFORE_FRESH_OUTCOMES',
        'created_utc': utc_now(), 'git': state,
        'source_sha256': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sources},
        'asset_sha256': {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in assets},
        'runtime': {'package_versions': versions, 'python_environment_hash': environment_hash,
                    'sumo_version': sumo_version()},
        'previous_failed_v1_manifest_sha256': sha256_file(FAILED_V1),
        'development_diagnosis_sha256': sha256_file(DEV),
        'mode_512_debug_sha256': sha256_file(DEBUG),
        'regret_premise_audit_sha256': sha256_file(PREMISES),
        'repaired_pilot_sha256': sha256_file(PILOT),
        'repair_analysis_plan_sha256': sha256_file(PLAN),
        'ego_lane_change_mode': EGO_LANE_CHANGE_MODE,
        'scenario_sampling': {'candidate_start': SEED_START, 'scan_limit': SEED_SCAN_LIMIT,
                              'n_per_family': N_PER_FAMILY, 'families': list(FAMILIES),
                              'selected': selected, 'selected_sha256': canonical_json_hash(selected),
                              'selection': 'first outcome-blind generator family assignments to reach each quota'},
        'conditions': {'nominal': asdict(nominal), 'impaired': asdict(impaired)},
        'condition_order': list(CONDITIONS), 'methods': list(METHODS),
        'model_rules': {method: {key: rule[key] for key in ('path', 'sha256', 'shrinkage')}
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
        'scope': 'repaired four-family designed SUMO mixture; no public-road safety claim',
    }
    write_json_atomic(OUTPUT, registration)
    print(json.dumps({'status': registration['status'], 'selected_scenarios': len(selected),
                      'seed_list_sha256': registration['scenario_sampling']['selected_sha256'],
                      'registration_sha256': sha256_file(OUTPUT)}, indent=2))


if __name__ == '__main__':
    main()
