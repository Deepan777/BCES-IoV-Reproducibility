#!/usr/bin/env python3
"""Freeze validation-only wire rules and a powered NEW independent slot audit."""
from __future__ import annotations

import gzip
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from bces.evaluation.confirmation_power import exact_risk_audit_power, exact_paired_superiority_power
from bces.evaluation.slot_confirmation import random_slot
from bces.geometry.codebooks import get_normal_codebook
from bces.geometry.drift import compute_drift
from bces.models.bound_policy import FrozenWirePolicy, bind_reference, reference_query
from bces.models.decision_surface import CAP, STEP, GUARD
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.world import WorldObject
from bces.protocol.bindings import sender_id_hash
from bces.protocol.bound_exchange import BoundReceiver
from bces.protocol.receiver import DecisionState
from bces.simulation.controlled_contract import make_message, ego_from_sample
from bces.simulation.controlled_decisions import kinematic
from bces.simulation.controlled_slots import decode_slot
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.environment import _package_versions
from bces.utils.hashing import sha256_file, canonical_json_hash
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

OUTPUT = ROOT/'outputs/study_b/slot_confirmation_v1/preregistration.json'


def main():
    state = git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean source required')
    if OUTPUT.exists(): raise FileExistsError('registration is immutable')
    base = ROOT/'outputs/study_b'
    prerequisites = [base/'controlled_wire_audit_v2.json', base/'controlled_slot_contract_v2.json']
    for path in prerequisites:
        if json.loads(path.read_text())['status'] != 'PASS': raise RuntimeError('engineering gate failed')
    config_path = ROOT/'configs/evaluation/study_b_controlled_v2.yaml'
    config = yaml.safe_load(config_path.read_text())
    data_root = base/'controlled_development_v2'
    model_root = base/'decision_surface_controlled_v2'
    data = json.loads((data_root/'run_manifest.json').read_text())
    training = json.loads((model_root/'run_manifest.json').read_text())
    if training['data_manifest_sha256'] != sha256_file(data_root/'run_manifest.json'):
        raise ValueError('training provenance mismatch')
    torch.set_num_threads(2)
    models, checkpoints = {}, {}
    for family in ('surface', 'scalar_ttl'):
        group = training['runs']['reference_margins:'+family]
        seed = group['selected_seed']
        path = model_root/f'reference_margins_{family}_{seed}.pt'
        digest = next(r['checkpoint_sha256'] for r in group['runs'] if r['seed'] == seed)
        models[family] = FrozenWirePolicy(path, expected_sha256=digest, policy_hash=data['policy_hash'])
        checkpoints[family] = {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest, 'seed': seed}

    # Selection uses ONLY observable drift and model outputs. Every fixed-domain
    # slot is in the coverage denominator, including unknown oracle labels.
    candidates = np.arange(256, dtype=float)*STEP
    records = []
    scenarios = 0
    normals = get_normal_codebook(1)
    for name, digest in sorted(data['artifact_sha256'].items()):
        if not name.startswith('validation_'): continue
        path = data_root/name
        if sha256_file(path) != digest: raise ValueError('validation artifact mismatch')
        with gzip.open(path, 'rt') as f: scene = json.load(f)
        scenarios += 1
        points = {(p['behavior'], p['receiver_drift_variant'], p['cache_age_s']): p for p in scene['points']}
        for index, reference in enumerate(scene['references']):
            t = reference['reference_timestamp_ms']
            frame = scene['traces']['0'][str(t)]
            ego = ego_from_sample(frame['receiver'])
            objects = tuple(WorldObject(**o) for o in frame['objects'])
            message = make_message(tuple(o for o in objects if math.hypot(o.x_m-ego.x_m,o.y_m-ego.y_m)<=config['cooperative_range_m']),
                                   t, message_id=int(scene['scenario_id']))
            payload = message.encode()
            query = reference_query(reference, query_id=index+1, sender_hash=sender_id_hash(message.sender_id))
            behavior = tuple(BEHAVIOR_IDS)[int(query.behavior)]
            slots = []
            for variant in range(len(config['receiver_drift_variants'])):
                for age in config['ages_s']:
                    timestamp = t+round(age*1000)
                    sample = scene['traces'][str(variant)][str(timestamp)]['receiver']
                    if sample is None: continue
                    current = kinematic(ego_from_sample(sample), timestamp)
                    drift = compute_drift(query.reference_state, current, query.drift_scales).normalized
                    # Match the receiver's Python summation, not BLAS ordering.
                    projection = [sum(n*v for n,v in zip(row,drift)) for row in normals]
                    point = points.get((behavior, variant, age))
                    slots.append((current, projection, drift[6], point['valid'] if point else None))
            for family, model in models.items():
                bound = bind_reference(reference, query, payload, model_sha256=model.sha256,
                                       view=model.view, family=family, shrinkage=0.)
                offsets, origin = model.predict(bound.encode(), payload)
                quantized = np.floor(np.maximum(0., offsets[None,:]-candidates[:,None])/STEP)*STEP
                projections = np.asarray([s[1] if family == 'surface' else [s[2]] for s in slots])
                mask = np.min(quantized[:,None,:]-projections[None,:,:], axis=2) >= GUARD
                mask &= origin
                records.append({'family': family, 'bound': bound, 'payload': payload,
                                'sender': message.sender_id, 'slots': slots, 'mask': mask})
    total = scenarios*len(BEHAVIOR_IDS)*len(config['receiver_drift_variants'])*len(config['ages_s'])
    if scenarios != config['partitions']['validation']: raise AssertionError('validation count mismatch')
    rules = {}
    for family, model in models.items():
        group = [r for r in records if r['family'] == family]
        counts = sum((r['mask'].sum(axis=1) for r in group), start=np.zeros(256, dtype=int))
        if np.any(np.diff(counts)>0): raise AssertionError('shrinkage must be monotone')
        index = int(np.flatnonzero(counts <= math.floor(.8*total))[0])
        accepted = unknown = invalid = checked = 0
        for record in group:
            wire = replace(record['bound'], shrinkage=float(candidates[index])).encode()
            receiver = BoundReceiver(); receiver.register(wire)
            receiver.install(model.issue(wire, record['payload']), record['payload'])
            q = record['bound'].query
            for j, (current, _, _, valid) in enumerate(record['slots']):
                actual = receiver.evaluate(current,q.behavior,q.policy_hash,record['sender']).state == DecisionState.ACCEPT_REUSE
                if actual != bool(record['mask'][index,j]): raise AssertionError('selected wire decision mismatch')
                checked += 1; accepted += actual
                unknown += actual and valid is None
                invalid += actual and valid is False
        if accepted != int(counts[index]): raise AssertionError('count mismatch')
        rules[family] = {**checkpoints[family], 'feature_view': model.view,
            'shrinkage_grid_index': index, 'shrinkage': float(candidates[index]),
            'validation_total_slots': total, 'validation_wire_checks': checked,
            'validation_accepted': accepted, 'validation_coverage': accepted/total,
            'validation_invalid_accepted': invalid, 'validation_unknown_accepted': unknown,
            'validation_adverse_reuse_rate': (invalid+unknown)/accepted if accepted else None,
            'selected_by': 'smallest grid shrinkage with coverage <= 0.8; labels not used'}
    plan = {'calibration': {'scenario_seed_start': 600000, 'scenarios': 8000, 'phase_id': 1},
            'confirmation': {'scenario_seed_start': 700000, 'scenarios': 8000, 'phase_id': 2}}
    randomization_seed = 150926
    for phase, spec in plan.items():
        assignments = [(seed, random_slot(seed,randomization_seed,spec['phase_id'])) for seed in
                       range(spec['scenario_seed_start'],spec['scenario_seed_start']+spec['scenarios'])]
        spec['assignment_sha256'] = canonical_json_hash(assignments)
        spec['risk_audit_power_at_declared_alternative'] = exact_risk_audit_power(spec['scenarios'],.75,.04)
        spec['paired_power_at_declared_alternative'] = exact_paired_superiority_power(spec['scenarios'],.008,.032)
    # Bind all project Python and selected simulator/config inputs; new independent
    # engineering modules can be added later without rewriting this registration.
    sources = list((ROOT/'bces').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'scripts/16_generate_controlled_development.py',
        ROOT/'scripts/23_run_slot_confirmation.py', ROOT/'uv.lock', ROOT/'pyproject.toml',
        config_path, ROOT/config['planner'],ROOT/config['validity'],ROOT/config['network']]
    versions, environment_hash = _package_versions()
    report = {'schema_version': 1, 'status': 'REGISTERED_NOT_RUN', 'registered_utc': utc_now(),
        'git': state, 'config': config, 'policy_hash': data['policy_hash'], 'operating_rules': rules,
        'runtime': {'package_versions': versions, 'python_environment_hash': environment_hash,
                    'sumo_version': sumo_version()},
        'plan': plan, 'slot_randomization_seed': randomization_seed, 'slot_domain': 315,
        'slot_layout': [decode_slot(i,config) for i in range(315)],
        'source_sha256': {p.relative_to(ROOT).as_posix():sha256_file(p) for p in sources},
        'prerequisite_sha256': {p.relative_to(ROOT).as_posix():sha256_file(p) for p in prerequisites},
        'data_manifest_sha256':sha256_file(data_root/'run_manifest.json'),
        'training_manifest_sha256':sha256_file(model_root/'run_manifest.json'),
        'risk': {'target': .05, 'family_alpha': .05, 'methods': 2, 'minimum_accepted': 100,
                 'alternative_coverage': .75, 'alternative_adverse_reuse_rate': .04},
        'paired': {'alpha': .025, 'endpoint':'accepted AND (invalid OR oracle unavailable), per scenario slot',
                   'alternative_surface_only_adverse': .008, 'alternative_ttl_only_adverse': .032,
                   'coverage_equivalence_tolerance': .03, 'coverage_interval_alpha': .05,
                   'coverage_equivalence_power': 'not independently established; mandatory additional claim gate'},
        'stopping': 'fixed N per phase; no interim endpoint reports; no significance stopping; unknown errors halt; resume exact identities only',
        'gates': 'confirmation inaccessible unless BOTH frozen rules pass calibration; no post-calibration threshold selection; failure retains abstention',
        'assumptions': ['independent scenarios and label-independent uniform slots from the designed mixture',
            'infeasible reference/absent current receiver abstains; unavailable oracle labels do not suppress receiver acceptance',
            'accepted unavailable labels count adverse in conservative risk and paired endpoints',
            'power is conditional on stated alternatives, not a promised result'],
        'limitations': ['controlled same-future open-loop validity only; not closed-loop physical safety',
            'application exchange bytes are not network savings; no radio or authenticated-security claim',
            'uncertainty envelopes and uniform regret bounds remain unverified',
            'closed-loop pilot and independent closed-loop registration remain required'],
        'original_test_accessed': False, 'fresh_outcomes_accessed': False,
        'scientific_success': False, 'manuscript_allowed': False,
        'statistical_reference': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html'}
    write_json_atomic(OUTPUT, report)
    print(json.dumps({'status': report['status'], 'rules': rules, 'plan': plan}, indent=2))


if __name__ == '__main__': main()
