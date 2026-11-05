"""Empirical theory-premise audit and measured receiver computation costs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import torch
from scipy.optimize import nnls
from development_study import ORIGINAL,OUT,save_json,log,sha
from local_policies import LocalRule
from bces.geometry.drift import DriftScales,compute_drift
from bces.oracle.planner import PlannerConfig
from bces.oracle.world import WorldObject,PlannedTrajectory,CostVector
from bces.simulation.controlled_contract import ego_from_sample
from bces.simulation.controlled_decisions import ControlledDecisionPlanner,route_for,evaluate_recorded_future,kinematic

def candidate_costs(planner,ego,behavior,world,timestamp):
    route=route_for(ego,behavior,planner);result={}
    maneuver={'keep':'keep','brake':'keep','accelerate':'keep','left':'left','right':'right'}[behavior]
    for acceleration in planner.config.longitudinal_accelerations_mps2:
        points=planner._candidate(ego,route,acceleration,maneuver)
        if points is None:continue
        trajectory=PlannedTrajectory(maneuver,acceleration,points,CostVector(0,0,0,0,0,0,0,None))
        result[acceleration]=evaluate_recorded_future(planner,trajectory,world,ego,route,timestamp).total
    return result

def premises():
    torch.set_num_threads(2)
    base=ORIGINAL/'outputs/study_b/controlled_development_v2'
    manifest=json.loads((base/'run_manifest.json').read_text())
    planner=ControlledDecisionPlanner(PlannerConfig.from_yaml(ORIGINAL/'configs/planner/kinematic.yaml'))
    counts=Counter();cases=[];start=time.perf_counter()
    paths=sorted([p for p in base.glob('*.json.gz') if p.name.startswith(('train_','validation_'))])
    train_names=sorted(p.stem for p in paths if p.name.startswith('train_'))
    fitting=set(train_names[:160]);residual=set(train_names[160:])
    for number,path in enumerate(paths,1):
        assert sha(path)==manifest['artifact_sha256'][path.name]
        with gzip.open(path,'rt',encoding='utf-8') as f:scene=json.load(f)
        traces=scene['traces']
        worlds={variant:{int(t):tuple(WorldObject(**o) for o in frame['objects']) for t,frame in trace.items()} for variant,trace in traces.items()}
        references={r['reference_id']:r for r in scene['references']}
        for point in scene['points']:
            if point['cache_age_s'] not in (.4,2.,4.):continue
            variant=str(point['receiver_drift_variant']);trace=traces[variant];world=worlds[variant]
            reference=references[point['reference_id']];t0=reference['reference_timestamp_ms'];t=point['current_timestamp_ms']
            ego0=ego_from_sample(trace[str(t0)]['receiver']);ego=ego_from_sample(trace[str(t)]['receiver'])
            c0=candidate_costs(planner,ego0,point['behavior'],world,t0)
            c=candidate_costs(planner,ego,point['behavior'],world,t)
            if not c0 or not c:counts['no_feasible_set']+=1;continue
            retained=float(reference['observable_margin_features'][3]);chosen=float(point['cached_acceleration_mps2'])
            counts['sampled_states']+=1;counts['feasible_set_changed']+=set(c0)!=set(c)
            counts['retained_action_infeasible']+=retained not in c
            counts['cached_replanning_changed_action']+=retained!=chosen
            common=sorted(set(c0)&set(c));z=np.abs(np.asarray(point['normalized_drift']))
            phase='fit' if path.stem in fitting else 'residual' if path.stem in residual else 'validation'
            cases.append({'scenario':scene['scenario_id'],'phase':phase,'behavior':point['behavior'],'x':np.r_[1,z],
                'c0':c0,'c':c,'common':common,'retained':retained,'chosen':chosen})
        if number%40==0:log(stage='premise_costs',scenarios=number,total=len(paths),seconds=time.perf_counter()-start)
    samples=defaultdict(list)
    for case in cases:
        for action in case['common']:
            samples[(case['behavior'],action)].append((case['phase'],case['scenario'],case['x'],abs(case['c'][action]-case['c0'][action])))
    models={}
    for key,rows in samples.items():
        fit=[r for r in rows if r[0]=='fit'];cal=[r for r in rows if r[0]=='residual']
        if len(fit)<8 or not cal:continue
        coef=nnls(np.stack([r[2] for r in fit]),np.array([r[3] for r in fit]))[0]
        byscene=defaultdict(list)
        for _,scene,x,y in cal:byscene[scene].append(y-float(x@coef))
        residuals=sorted(max(v) for v in byscene.values())
        rank=min(len(residuals),math.ceil((len(residuals)+1)*.99))
        # Empirical held-out scenario-maximum residual inflation. At small
        # calibration N the exact 99% conformal rank may require infinity.
        inflation=max(0.,residuals[rank-1])
        models[key]=(coef,inflation)
    validation=Counter();bounds=[]
    for case in cases:
        if case['phase']!='validation':continue
        validation['states']+=1
        if set(case['c0'])!=set(case['c']) or case['chosen'] not in case['c0']:
            validation['common_set_premise_failed']+=1;continue
        keys=[(case['behavior'],a) for a in case['c']]
        if any(k not in models for k in keys):validation['envelope_fit_unavailable']+=1;continue
        errors={a:float(case['x']@models[(case['behavior'],a)][0]+models[(case['behavior'],a)][1]) for a in case['c']}
        cover=all(abs(case['c'][a]-case['c0'][a])<=errors[a]+1e-8 for a in case['c'])
        selected=case['chosen']
        bound=max([0.]+[case['c0'][selected]-case['c0'][a]+errors[selected]+errors[a] for a in case['c'] if a!=selected])
        actual=case['c'][selected]-min(case['c'].values())
        validation['eligible_envelope_test_states']+=1;validation['all_action_envelopes_cover']+=cover
        validation['regret_bound_at_most_5']+=bound<=5
        validation['empirical_bound_violations']+=actual>bound+1e-8
        bounds.append(bound)
    save_json(OUT/'premise_envelopes_report.json',{'scope':'development-only empirical envelope feasibility audit',
        'sampling':'all three receiver acceleration variants, all five behaviors, ages 0.4/2/4 seconds, available labels from 240 training and 80 validation scenarios',
        'sampled_premises':dict(counts),'validation':dict(validation),'bound_quantiles':np.quantile(bounds,[.5,.95,1]).tolist() if bounds else None,
        'fitting':'nonnegative least squares on 160 training scenarios; inflation by empirical 99th percentile of per-scenario maximum residual in the remaining 80 training scenarios',
        'calibration_rank_can_require_infinite_bound':'finite empirical inflation is used solely as a feasibility diagnostic; no 99-percent conformal or uniform guarantee is claimed',
        'uniform_envelopes_established':False,'learned_region_containment_established':False,
        'envelope_model_count':len(models),'elapsed_s':time.perf_counter()-start})

def runtime():
    torch.set_num_threads(2)
    base=ORIGINAL/'outputs/study_b/controlled_development_v2'
    with gzip.open(sorted(base.glob('validation_*.json.gz'))[0],'rt',encoding='utf-8') as f:scene=json.load(f)
    ref=scene['references'][0];point=next(p for p in scene['points'] if p['reference_id']==ref['reference_id'] and p['cache_age_s']>.1)
    z=point['normalized_drift'];current=point['observable_current_features']
    result={}
    methods=('frozen_surface','frozen_scalar_ttl','flexible_mlp','flexible_tree','cam_fixed','normalized_drift','age_only','observable_cost_trigger')
    for name in methods:
        model=LocalRule(name)
        for _ in range(100):state=model.prepare(ref);model.score(state,z,current)
        model.timing_prepare_ns=[];model.timing_check_ns=[]
        for _ in range(1000):
            state=model.prepare(ref)
            model.score(state,z,current)
        result[name]={'prepare_us_p50_p95_p99':(np.quantile(model.timing_prepare_ns,[.5,.95,.99])/1000).tolist(),
            'check_us_p50_p95_p99':(np.quantile(model.timing_check_ns,[.5,.95,.99])/1000).tolist(),
            'reference_model_inferences_per_prepare':int(model.model is not None),
            'model_inferences_per_check':int(name in ('flexible_mlp','flexible_tree'))}
    result={'methods':result,'warmup':100,'repetitions':1000,'execution_device':'CPU','torch_threads':2,
        'platform':platform.platform(),'processor':platform.processor(),
        'includes':'Python feature construction, preprocessing and local policy call; not training, model loading, sensor processing or end-to-end network delay',
        'model_file_bytes':{p.name:p.stat().st_size for p in OUT.glob('*') if p.suffix in ('.pt','.joblib')}}
    save_json(OUT/'runtime_report.json',result);log(stage='runtime_complete')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['premises','runtime']);args=parser.parse_args()
    premises() if args.phase=='premises' else runtime()
