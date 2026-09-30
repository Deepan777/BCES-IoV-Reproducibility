#!/usr/bin/env python3
"""Fixed-N resumable collection; final confirmation is gated by calibration.

Progress reveals identities/counts only, never interim scientific endpoints.
Raw outcomes are preserved; an interrupted partial file halts for integrity
review instead of silently replacing a scenario. No retries with new seeds.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bces.evaluation.slot_confirmation import random_slot, evaluate_record, summarize
from bces.models.bound_policy import FrozenWirePolicy
from bces.simulation.controlled_slots import generate_slot
from bces.simulation.sumo_adapter import sumo_version
from bces.utils.budget import Budget, require_headroom, build_budget_report
from bces.utils.environment import _package_versions
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


def collect(job):
    seed, slot, config, thresholds, phase = job
    return generate_slot(seed,slot,config,thresholds,phase)


def verify_registration(registration, phase, root=ROOT):
    if phase not in ('calibration','confirmation'): raise ValueError('unknown phase')
    for name,digest in {**registration['source_sha256'],**registration['prerequisite_sha256']}.items():
        if sha256_file(root/name) != digest: raise RuntimeError('registered input changed: '+name)
    spec = registration['plan'][phase]
    assignments = [(seed,random_slot(seed,registration['slot_randomization_seed'],spec['phase_id'])) for seed in
                   range(spec['scenario_seed_start'],spec['scenario_seed_start']+spec['scenarios'])]
    if canonical_json_hash(assignments) != spec['assignment_sha256']:
        raise RuntimeError('slot assignment mismatch')
    return assignments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=['calibration','confirmation'])
    parser.add_argument('--registration',default='outputs/study_b/slot_confirmation_v1/preregistration.json')
    parser.add_argument('--workers',type=int,default=2,choices=[1,2])
    args = parser.parse_args()
    state = git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean source required')
    registration_path = (ROOT/args.registration).resolve()
    registration_path.relative_to(ROOT)
    registration = json.loads(registration_path.read_text())
    digest = sha256_file(registration_path)
    assignments = verify_registration(registration,args.phase)
    versions, environment_hash = _package_versions()
    runtime = {'package_versions':versions,'python_environment_hash':environment_hash,'sumo_version':sumo_version()}
    if runtime != registration['runtime']: raise RuntimeError('registered runtime changed')
    output = registration_path.parent/args.phase
    if (output/'run_manifest.json').exists(): raise FileExistsError('completed phase is immutable')
    if args.phase == 'confirmation':
        previous = registration_path.parent/'calibration/run_manifest.json'
        calibration = json.loads(previous.read_text())
        if (calibration['registration_sha256'] != digest or calibration['status'] != 'COMPLETE'
                or not calibration['results']['both_risk_gates_passed']):
            raise RuntimeError('both frozen rules must pass independent calibration before final confirmation')
        for name,expected in calibration['artifact_sha256'].items():
            if sha256_file(previous.parent/name) != expected: raise RuntimeError('calibration artifact changed')
    budget = Budget(workspace_root=ROOT,data_root=ROOT/'data')
    require_headroom(budget,incoming_bytes=250_000_000)
    output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('*.tmp')): raise RuntimeError('partial file requires integrity review; no silent replacement')
    started = utc_now()
    context = {'registration_sha256':digest,'phase':args.phase,'sumo_version':sumo_version(),
               'runner_sha256':sha256_file(Path(__file__)),'runtime':runtime}
    launch_path = output/'collection_context.json'
    if launch_path.exists():
        if json.loads(launch_path.read_text()) != context: raise RuntimeError('resume runtime/context mismatch')
    else:
        write_json_atomic(launch_path,context)
    config = registration['config']
    validity = yaml.safe_load((ROOT/config['validity']).read_text())
    thresholds = {k:validity[k] for k in ('max_cost_regret','max_cached_risk','max_trajectory_deviation_m')}
    torch.set_num_threads(2)
    models = {family:FrozenWirePolicy(ROOT/rule['path'],expected_sha256=rule['sha256'],
                policy_hash=registration['policy_hash']) for family,rule in registration['operating_rules'].items()}
    completed = {}
    expected_names = {f'slot_{seed}.json.gz' for seed,_ in assignments}
    if any(p.name not in expected_names for p in output.glob('slot_*.json.gz')):
        raise RuntimeError('unregistered scenario artifact present')
    remaining = []
    for seed,slot in assignments:
        path = output/f'slot_{seed}.json.gz'
        if not path.exists():
            remaining.append((seed,slot,config,thresholds,args.phase))
            continue
        with gzip.open(path,'rt') as f: record = json.load(f)
        if (record['registration_sha256'] != digest or record['scenario_id'] != str(seed)
                or record['slot'] != slot or record['split'] != args.phase):
            raise RuntimeError('resume record identity mismatch')
        # Recompute exact receiver results, without printing aggregate endpoints.
        replay = evaluate_record(record,models,registration['operating_rules'])
        if canonical_json_hash(replay) != canonical_json_hash(record['evaluation']):
            raise RuntimeError('resumed wire evaluation mismatch')
        completed[seed] = record['evaluation']
    def save(record):
        seed = int(record['scenario_id'])
        row = evaluate_record(record,models,registration['operating_rules'])
        record = {**record,'registration_sha256':digest,'evaluation':row}
        path = output/f'slot_{seed}.json.gz'
        temporary = path.with_suffix(path.suffix+'.tmp')
        if path.exists() or temporary.exists(): raise FileExistsError('cannot replace raw slot')
        with gzip.open(temporary,'xt',encoding='utf-8') as f:
            json.dump(record,f,sort_keys=True,separators=(',',':'),allow_nan=False)
        temporary.rename(path)
        completed[seed] = row
        if len(completed)%100 == 0:
            require_headroom(budget,incoming_bytes=10_000_000)
            progress = {'status':'RUNNING','phase':args.phase,'completed':len(completed),
                'fixed_total':len(assignments),'updated_utc':utc_now(),'registration_sha256':digest}
            write_json_atomic(output/'progress.json',progress)
            print(json.dumps(progress),flush=True)
    # Keep at most two raw traces resident per batch, even on Windows spawn.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0,len(remaining),args.workers):
            for record in pool.map(collect,remaining[start:start+args.workers]): save(record)
    rows = [completed[seed] for seed,_ in assignments]
    risk,paired = registration['risk'],registration['paired']
    results = summarize(rows,expected_n=len(assignments),target=risk['target'],alpha=risk['family_alpha'],
        minimum_accepted=risk['minimum_accepted'],paired_alpha=paired['alpha'],
        coverage_tolerance=paired['coverage_equivalence_tolerance'])
    report = {'status':'COMPLETE','phase':args.phase,'registration_sha256':digest,
        'started_utc':started,'completed_utc':utc_now(),'git':state,'context':context,
        'results':results,'artifact_sha256':{name:sha256_file(output/name) for name in sorted(expected_names)},
        'budget':build_budget_report(budget)['measurements'],'original_test_accessed':False,
        'scientific_success':False,'manuscript_allowed':False,
        'scope':'independent controlled open-loop slot audit; not closed-loop safety or communication savings'}
    write_json_atomic(output/'run_manifest.json',report)
    write_json_atomic(output/'progress.json',{'status':'COMPLETE','phase':args.phase,'completed':len(rows),
        'fixed_total':len(rows),'registration_sha256':digest,'updated_utc':utc_now()})
    print(json.dumps({'status':'COMPLETE','phase':args.phase,'results':results},indent=2),flush=True)


if __name__ == '__main__': main()
