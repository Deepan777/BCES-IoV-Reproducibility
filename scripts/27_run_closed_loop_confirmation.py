#!/usr/bin/env python3
"""Resumable fixed-N causal closed-loop collection; no interim endpoints."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import psutil
import torch

from bces.evaluation.closed_loop_confirmation import CONDITIONS, FAMILIES, METHODS, scenario_record, summarize
from bces.models.bound_policy import FrozenWirePolicy
from bces.network.events import NetworkCondition
from bces.simulation.controlled_closed_loop import run_branch
from bces.simulation.controlled_slots import development_generator
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.budget import Budget, require_headroom, build_budget_report
from bces.utils.environment import _package_versions
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

DEFAULT_REGISTRATION = ROOT/'outputs/study_b/closed_loop_confirmation_v1/preregistration.json'


def verify_registration(registration: dict, root: Path = ROOT):
    if registration['status'] != 'FROZEN_BEFORE_FRESH_OUTCOMES':
        raise RuntimeError('closed-loop registration not frozen')
    for name, expected in {**registration['source_sha256'], **registration['asset_sha256']}.items():
        if sha256_file(root/name) != expected:
            raise RuntimeError('registered source or asset changed: '+name)
    if registration['methods'] != list(METHODS) or registration['condition_order'] != list(CONDITIONS):
        raise RuntimeError('registered branch grid changed')
    if registration['scenario_sampling']['families'] != list(FAMILIES):
        raise RuntimeError('registered family grid changed')
    selected = registration['scenario_sampling']['selected']
    if canonical_json_hash(selected) != registration['scenario_sampling']['selected_sha256']:
        raise RuntimeError('seed-list digest changed')
    if len({row['seed'] for row in selected}) != len(selected):
        raise RuntimeError('duplicate independent scenario seed')
    generator = development_generator()
    config = registration['config']
    for row in selected:
        if generator.sample_scenario(row['seed'], config)[0].family != row['family']:
            raise RuntimeError('registered family assignment changed')
    versions, environment_hash = _package_versions()
    runtime = {'package_versions': versions, 'python_environment_hash': environment_hash,
               'sumo_version': sumo_version()}
    if runtime != registration['runtime']:
        raise RuntimeError('registered runtime changed')
    return runtime


def artifact_name(seed: int, condition: str, method: str) -> str:
    if condition not in CONDITIONS or method not in METHODS:
        raise ValueError('unregistered branch')
    return f'branch_{seed}_{condition}_{method}.json.gz'


def read_branch(path: Path, *, digest: str, seed: int, family: str, condition: str, method: str) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        wrapper = json.load(handle)
    if (wrapper['registration_sha256'] != digest or wrapper['seed'] != seed
            or wrapper['family'] != family or wrapper['condition'] != condition
            or wrapper['method'] != method):
        raise RuntimeError('preserved branch identity mismatch: '+path.name)
    branch = wrapper['branch']
    if canonical_json_hash(branch) != wrapper['branch_sha256']:
        raise RuntimeError('preserved branch digest mismatch: '+path.name)
    if (branch['scenario_id'] != str(seed) or branch['family'] != family
            or branch['method'] != method):
        raise RuntimeError('preserved branch body identity mismatch: '+path.name)
    return branch


def save_branch(path: Path, *, digest: str, seed: int, family: str,
                condition: str, method: str, branch: dict):
    temporary = path.with_suffix(path.suffix+'.tmp')
    if path.exists() or temporary.exists():
        raise FileExistsError('cannot replace a preserved closed-loop branch')
    wrapper = {'registration_sha256': digest, 'seed': seed, 'family': family,
               'condition': condition, 'method': method,
               'branch_sha256': canonical_json_hash(branch), 'branch': branch}
    with gzip.open(temporary, 'xt', encoding='utf-8') as handle:
        json.dump(wrapper, handle, sort_keys=True, separators=(',', ':'), allow_nan=False)
    temporary.rename(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, default=DEFAULT_REGISTRATION)
    args = parser.parse_args()
    state = git_state(ROOT)
    if not state['available'] or state['dirty']:
        raise RuntimeError('clean Git source required')
    registration_path = args.registration.resolve()
    registration_path.relative_to(ROOT)
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    digest = sha256_file(registration_path)
    runtime = verify_registration(registration)
    output = registration_path.parent/'confirmation'
    if (output/'run_manifest.json').exists():
        raise FileExistsError('completed powered confirmation is immutable')
    budget = Budget(workspace_root=ROOT, data_root=ROOT/'data')
    require_headroom(budget, incoming_bytes=200_000_000)
    if psutil.virtual_memory().available < 1_500_000_000:
        raise RuntimeError('defer powered closed-loop run below 1.5 GB free RAM')
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('*.tmp')):
        raise RuntimeError('partial branch requires integrity review; no silent replacement')
    context = {'registration_sha256': digest, 'runtime': runtime,
               'runner_sha256': sha256_file(Path(__file__))}
    context_path = output/'collection_context.json'
    if context_path.exists():
        if json.loads(context_path.read_text(encoding='utf-8')) != context:
            raise RuntimeError('resume context changed')
    else:
        write_json_atomic(context_path, context)
    selected = registration['scenario_sampling']['selected']
    expected = {artifact_name(row['seed'], condition, method)
                for row in selected for condition in CONDITIONS for method in METHODS}
    if any(path.name not in expected for path in output.glob('branch_*.json.gz')):
        raise RuntimeError('unregistered branch artifact present')
    torch.set_num_threads(2)
    models = {method: FrozenWirePolicy(ROOT/rule['path'], expected_sha256=rule['sha256'],
                                      policy_hash=registration['policy_hash'])
              for method, rule in registration['model_rules'].items()}
    conditions = {name: NetworkCondition(**registration['conditions'][name]) for name in CONDITIONS}
    rows = []
    started = utc_now()
    for index, assignment in enumerate(selected, start=1):
        seed, family = assignment['seed'], assignment['family']
        block = {}
        for condition in CONDITIONS:
            for method in METHODS:
                path = output/artifact_name(seed, condition, method)
                if path.exists():
                    branch = read_branch(path, digest=digest, seed=seed, family=family,
                                         condition=condition, method=method)
                else:
                    branch = run_branch(seed=seed, method=method, config=registration['config'],
                                        condition=conditions[condition], network_seed=seed+910000,
                                        model=models.get(method), rule=registration['model_rules'].get(method),
                                        horizon_s=registration['horizon_s'])
                    save_branch(path, digest=digest, seed=seed, family=family,
                                condition=condition, method=method, branch=branch)
                block[(condition, method)] = branch
        rows.append(scenario_record(seed, family, block,
                                    progress_margin_m=registration['endpoints']['adverse_progress_deficit_m']))
        if index % 10 == 0:
            require_headroom(budget, incoming_bytes=10_000_000)
            if psutil.virtual_memory().available < 1_500_000_000:
                raise RuntimeError('resource pause; resume same registration after RAM recovers')
            progress = {'status': 'RUNNING', 'completed_scenarios': index,
                        'fixed_total': len(selected), 'registration_sha256': digest,
                        'updated_utc': utc_now()}
            write_json_atomic(output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    endpoints = registration['endpoints']
    result = summarize(rows, n_per_family=registration['scenario_sampling']['n_per_family'],
                       adverse_margin=endpoints['adverse_rate_margin'],
                       byte_threshold=endpoints['communication_minimum_mean_reduction'],
                       alpha_each=endpoints['alpha_each_one_sided'])
    report = {'status': 'COMPLETE', 'started_utc': started, 'completed_utc': utc_now(),
              'registration_sha256': digest, 'git': state, 'context': context,
              'results': result, 'artifact_sha256': {name: sha256_file(output/name)
                                                   for name in sorted(expected)},
              'budget': build_budget_report(budget)['measurements'],
              'original_test_accessed': False, 'manuscript_allowed': False,
              'scope': 'powered designed-distribution closed loop; not public-road safety'}
    write_json_atomic(output/'run_manifest.json', report)
    write_json_atomic(output/'progress.json', {'status': 'COMPLETE',
        'completed_scenarios': len(rows), 'fixed_total': len(rows),
        'registration_sha256': digest, 'updated_utc': utc_now()})
    print(json.dumps({'status': 'COMPLETE', 'results': result}, indent=2), flush=True)


if __name__ == '__main__':
    main()
