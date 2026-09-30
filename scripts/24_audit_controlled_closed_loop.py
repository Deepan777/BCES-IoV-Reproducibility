#!/usr/bin/env python3
"""Existing-development-seed closed-loop audit; never confirmation evidence."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def select_cases(available, families, max_cases, cases_per_family):
    """Select complete deterministic training-seed blocks, never cherry-pick outcomes."""
    if type(cases_per_family) is not int or cases_per_family < 1:
        raise ValueError('positive cases per family required')
    if families is not None:
        if len(families) != len(set(families)):
            raise ValueError('family list contains duplicates')
        chosen = families
    else:
        chosen = sorted(available)[:max_cases]
        if len(chosen) != max_cases:
            raise RuntimeError('missing development family')
    missing = [family for family in chosen if family not in available]
    if missing:
        raise ValueError('unknown development families: '+','.join(missing))
    short = [family for family in chosen if len(available[family]) < cases_per_family]
    if short:
        raise RuntimeError('insufficient training cases: '+','.join(short))
    return [case for family in chosen for case in available[family][:cases_per_family]]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-cases',type=int,choices=range(1,7),default=6)
    parser.add_argument('--families',nargs='+',default=None,
        help='exact development families to audit, in the requested order')
    parser.add_argument('--cases-per-family',type=int,default=1,
        help='first N hash-verified training seeds for each selected family')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/study_b/controlled_closed_loop_pilot_v1')
    args=parser.parse_args()
    import psutil
    if psutil.virtual_memory().available<1_500_000_000:
        raise RuntimeError('pilot deferred below 1.5 GB available RAM; do not disturb another experiment')
    import torch
    from bces.models.bound_policy import FrozenWirePolicy
    from bces.network.events import NetworkCondition
    from bces.simulation.controlled_closed_loop import run_branch
    from bces.simulation.controlled_contract import compare_snapshots
    from bces.utils.budget import Budget,require_headroom,build_budget_report
    from bces.utils.hashing import sha256_file
    from bces.utils.reproducibility import git_state,utc_now,write_json_atomic
    state=git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean source required')
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'outputs/study_b'):
        raise ValueError('output must stay within Study B')
    if output.exists(): raise FileExistsError('pilot output is immutable')
    registration_path=ROOT/'outputs/study_b/slot_confirmation_v1/preregistration.json'
    registration=json.loads(registration_path.read_text())
    for name,digest in registration['source_sha256'].items():
        if sha256_file(ROOT/name)!=digest: raise RuntimeError('frozen calibration source changed: '+name)
    budget=Budget(workspace_root=ROOT,data_root=ROOT/'data')
    require_headroom(budget,incoming_bytes=50_000_000)
    data_root=ROOT/'outputs/study_b/controlled_development_v2'
    manifest=json.loads((data_root/'run_manifest.json').read_text())
    available={}
    for name,digest in sorted(manifest['artifact_sha256'].items()):
        if not name.startswith('train_'): continue
        path=data_root/name
        if sha256_file(path)!=digest: raise ValueError('development input changed')
        with gzip.open(path,'rt') as handle: scene=json.load(handle)
        available.setdefault(scene['family'],[]).append({'seed':int(scene['scenario_id']),
            'family':scene['family'],'source_sha256':digest})
    cases=select_cases(available,args.families,args.max_cases,args.cases_per_family)
    conditions={'nominal':NetworkCondition(50.,0.,10.),
        'impaired':NetworkCondition(100.,.1,1.,jitter_ms=20.,duplicate_probability=.05,reorder_probability=.05)}
    torch.set_num_threads(2)
    models={family:FrozenWirePolicy(ROOT/rule['path'],expected_sha256=rule['sha256'],
        policy_hash=registration['policy_hash']) for family,rule in registration['operating_rules'].items()}
    protocol={'scope':'engineering pilot; existing training seeds only','created_utc':utc_now(),'git':state,
        'cases':cases,'conditions':{key:asdict(value) for key,value in conditions.items()},'horizon_s':6.,
        'case_selection':{'explicit_families':args.families,'maximum_cases':args.max_cases,
            'cases_per_family':args.cases_per_family,'selection':'first sorted hash-verified training artifacts'},
        'methods':['local_only','periodic_payload','surface','scalar_ttl'],
        'periodic_payload_period_s':registration['config']['step_s'],'sender_computation_delay_ms':6.,
        'request_timeout_ms':400.,'maximum_retries':1,'security_placeholder_bytes':64,
        'tracking_review_thresholds':{'position_m':.5,'speed_mps':.5,'heading_rad':.1},
        'registration_sha256':sha256_file(registration_path),
        'source_sha256':{path.relative_to(ROOT).as_posix():sha256_file(path) for path in
            (ROOT/'bces/network/bound_session.py',ROOT/'bces/simulation/controlled_closed_loop.py',Path(__file__).resolve())},
        'not_powered_for_benefit_or_safety':True,'scientific_success':False,'manuscript_allowed':False}
    write_json_atomic(output/'pilot_protocol.json',protocol)
    summaries=[];comparisons=[];artifacts={};initial_by_seed={}
    for case in cases:
        for condition_name,condition in conditions.items():
            for method in protocol['methods']:
                result=run_branch(seed=case['seed'],method=method,config=registration['config'],
                    condition=condition,network_seed=case['seed']+910000,model=models.get(method),
                    rule=registration['operating_rules'].get(method),horizon_s=protocol['horizon_s'])
                initial=result['initial_contract']
                if case['seed'] not in initial_by_seed: initial_by_seed[case['seed']]=initial
                comparison=compare_snapshots(initial_by_seed[case['seed']],initial,tolerance=1e-6)
                comparisons.append({'seed':case['seed'],'condition':condition_name,'method':method,**comparison})
                if not comparison['equal_within_tolerance']: raise AssertionError('complete-prefix branch mismatch')
                traffic=result['traffic']
                if not traffic['byte_conservation_ok'] or not traffic['component_conservation_ok']:
                    raise AssertionError('closed-loop traffic conservation failed')
                if any(row['used_timestamp_ms']>row['available_ms'] for row in result['input_availability_audit']):
                    raise AssertionError('future observation reached closed-loop inputs')
                filename=f"{case['seed']}_{condition_name}_{method}.json.gz"
                with gzip.open(output/filename,'xt',encoding='utf-8') as handle:
                    json.dump(result,handle,sort_keys=True,separators=(',',':'),allow_nan=False)
                artifacts[filename]=sha256_file(output/filename)
                summary={'seed':case['seed'],'family':case['family'],'condition':condition_name,'method':method,
                    'outcome':result['outcome'],'actuation_contract_passed':result['actuation_contract_passed'],
                    'counts':result['counts'],'generated_bytes':traffic['generated_bytes'],'duration_s':result['duration_s']}
                summaries.append(summary);print(json.dumps(summary),flush=True)
        require_headroom(budget)
    passed=all(row['actuation_contract_passed'] for row in summaries)
    communication=[]
    for case in cases:
        for condition_name in conditions:
            matched=[row for row in summaries if row['seed']==case['seed'] and row['condition']==condition_name]
            by_method={row['method']:row for row in matched}
            periodic=by_method['periodic_payload']
            for method in ('surface','scalar_ttl'):
                candidate=by_method[method]
                periodic_rate=periodic['generated_bytes']/periodic['duration_s'] if periodic['duration_s'] else None
                candidate_rate=candidate['generated_bytes']/candidate['duration_s'] if candidate['duration_s'] else None
                comparable=(periodic['duration_s']==candidate['duration_s'] and periodic['outcome']==candidate['outcome'])
                communication.append({'seed':case['seed'],'family':case['family'],'condition':condition_name,
                    'method':method,'periodic_payload_bytes':periodic['generated_bytes'],
                    'candidate_bytes':candidate['generated_bytes'],'periodic_bytes_per_second':periodic_rate,
                    'candidate_bytes_per_second':candidate_rate,'equal_duration_and_outcome':comparable,
                    'byte_reduction_fraction':(1-candidate_rate/periodic_rate)
                        if comparable and periodic_rate else None})
    report={'status':'PASS' if passed else 'ACTUATION_REVIEW_REQUIRED','completed_utc':utc_now(),'git':state,
        'protocol_sha256':sha256_file(output/'pilot_protocol.json'),'cases':summaries,
        'prefix_comparisons':comparisons,'artifact_sha256':artifacts,'all_prefix_checks_passed':True,
        'all_byte_and_input_availability_checks_passed':True,'communication_comparisons':communication,
        'communication_accounting_complete_at_application_and_declared_security_envelope':True,
        'budget':build_budget_report(budget)['measurements'],
        'scientific_success':False,'manuscript_allowed':False,'fresh_calibration_outcomes_accessed':False,
        'original_test_accessed':False,
        'next_gate':'resolve actuation contract before independent closed-loop power analysis and confirmation'}
    write_json_atomic(output/'audit.json',report)
    print(json.dumps({'status':report['status'],'branches':len(summaries),'scientific_success':False}),flush=True)
    return 0 if passed else 2


if __name__=='__main__': raise SystemExit(main())
