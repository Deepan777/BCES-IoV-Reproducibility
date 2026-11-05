"""Prespecified sensor-error sensitivity on completed independent slot traces.

Measured inputs are rebuilt before inference and counterfactual planning. The
label evaluator retains the clean simulated future world. Each noise condition
has one sampled slot per scenario; its repeated frames are not independent N.
"""
from collections import defaultdict
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import beta
from development_study import ORIGINAL,OUT,save_json,log,sha
from local_policies import LocalRule
from confirmation_study import REG,verify_registration
from bces.geometry.drift import DriftScales,compute_drift
from bces.oracle.world import WorldObject
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.simulation.controlled_contract import ego_from_sample,local_observations,make_message
from bces.simulation.controlled_decisions import ControlledDecisionPlanner,reference_contract,decision_pair,kinematic,route_for
from bces.simulation.scenario_builder import scenario_spec

ERRORS={
    'nominal':dict(position=.0,yaw=.0,velocity=.0,miss=.0,track=.0),
    'localization':dict(position=.25,yaw=1.,velocity=.2,miss=.0,track=.0),
    'missed_detection':dict(position=.0,yaw=.0,velocity=.0,miss=.1,track=.0),
    'tracking':dict(position=.0,yaw=.0,velocity=.0,miss=.0,track=.1),
    'combined':dict(position=.25,yaw=1.,velocity=.2,miss=.1,track=.1),
}
METHODS=('frozen_surface','frozen_scalar_ttl','flexible_mlp','flexible_tree','cam_fixed','normalized_drift','age_only','observable_cost_trigger')

def rng_for(*parts):
    seed=int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')
    return np.random.default_rng(seed)

def perturb_objects(items,error,scenario,timestamp,role):
    result=[]
    for o in items:
        rng=rng_for(scenario,timestamp,role,o.track_id,error)
        if rng.random()<error['miss']:continue
        identity=o.track_id+f':switch:{timestamp}:{role}' if rng.random()<error['track'] else o.track_id
        angle=rng.normal(0,math.radians(error['yaw']))
        vx=o.vx_mps*math.cos(angle)-o.vy_mps*math.sin(angle)+rng.normal(0,error['velocity'])
        vy=o.vx_mps*math.sin(angle)+o.vy_mps*math.cos(angle)+rng.normal(0,error['velocity'])
        result.append(replace(o,track_id=identity,x_m=o.x_m+rng.normal(0,error['position']),y_m=o.y_m+rng.normal(0,error['position']),vx_mps=vx,vy_mps=vy))
    return tuple(result)

def main():
    reg=json.loads(REG.read_text());verify_registration(reg)
    collected=json.loads((OUT/'fresh_slots/collection_manifest.json').read_text())
    assert collected['status']=='COMPLETE'
    torch.set_num_threads(2)
    planner=ControlledDecisionPlanner(PlannerConfig.from_yaml(ORIGINAL/'configs/planner/kinematic.yaml'))
    scales=DriftScales(40,5,5,.7,3,.1,4)
    config=yaml.safe_load((ORIGINAL/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    models={m:LocalRule(m) for m in METHODS};rows=[]
    out=OUT/'fresh_robustness';out.mkdir(parents=True,exist_ok=True)
    for number,path in enumerate(sorted((OUT/'fresh_slots').glob('*.json.gz')),1):
        with gzip.open(path,'rt',encoding='utf-8') as f:stored=json.load(f)
        raw=stored['raw'];seed=raw['scenario_id'];trace=raw['trace']
        t0=round(config['reference_time_s']*1000);t=t0+round(raw['age_s']*1000)
        clean_world={int(k):tuple(WorldObject(**o) for o in frame['objects']) for k,frame in trace.items()}
        spec=scenario_spec(raw['family']);hidden=(('person:' if spec.pedestrian else 'vehicle:')+spec.actor_id,) if spec.hidden_from_local else ()
        for condition,error in ERRORS.items():
            accepted={name:False for name in METHODS};valid=None;point=None;status='reference_unavailable'
            if raw['status']!='abstain':
                # A fixed per-case pose bias is causal at both reference and
                # current times and avoids creating yaw-rate noise from truth.
                bias=rng_for(seed,condition,'ego_bias')
                dx,dy=bias.normal(0,error['position'],2);dh=bias.normal(0,math.radians(error['yaw']))
                def measured(timestamp):
                    ego=ego_from_sample(trace[str(timestamp)]['receiver'])
                    ego=replace(ego,x_m=ego.x_m+dx,y_m=ego.y_m+dy,heading_rad=ego.heading_rad+dh)
                    truth_ego=ego_from_sample(trace[str(timestamp)]['receiver'])
                    local=local_observations(clean_world[timestamp],truth_ego,range_m=30,hidden_ids=hidden)
                    remote=tuple(o for o in clean_world[timestamp] if math.hypot(o.x_m-truth_ego.x_m,o.y_m-truth_ego.y_m)<=120)
                    local=perturb_objects(local,error,seed,timestamp,'local')
                    remote=perturb_objects(remote,error,seed,timestamp,'remote')
                    return ego,local,make_message(remote,timestamp,message_id=int(seed))
                try:
                    ego0,local0,m0=measured(t0)
                    reference,query,cached=reference_contract(scenario=seed,split='revision_error_sensitivity',behavior=raw['behavior'],timestamp=t0,
                        ego=ego0,local=local0,message=m0,planner=planner,scales=scales)
                    ego,local,fresh=measured(t)
                    z=np.array(compute_drift(query.reference_state,kinematic(ego,t),scales).normalized,np.float32)
                    try:features=observable_planner_features(planner,local,tuple(o.propagated(raw['age_s']) for o in cached),ego,route_for(ego,raw['behavior'],planner))
                    except RuntimeError:features=None
                    for name,model in models.items():
                        if name=='observable_cost_trigger' and features is None:continue
                        accepted[name]=bool(model.score(model.prepare(reference),z,features)>=model.threshold)
                    status='oracle_unavailable'
                    try:
                        point=decision_pair(reference=reference,query=query,cached_reference=cached,ego=ego,local=local,fresh_message=fresh,
                            future_world=clean_world,planner=planner,thresholds=ValidityThresholds(5,.4,2))
                        valid=bool(point['valid']);status='evaluated'
                    except RuntimeError:pass
                except RuntimeError:pass
            rows.append({'scenario_id':seed,'condition':condition,'status':status,'valid':valid,'accepted':accepted,
                'regret':point['cost_regret'] if point else None,'risk':point['cached_risk'] if point else None,'deviation_m':point['trajectory_deviation_m'] if point else None})
            if condition=='nominal':
                assert valid==stored['row']['valid'],('nominal label reconstruction mismatch',seed)
                assert all(accepted[m]==stored['row']['methods'][m]['accepted'] for m in METHODS),('nominal decision reconstruction mismatch',seed)
        if number%100==0:log(stage='robustness',completed=number,total=reg['slot_scenarios'])
    report={}
    for condition in ERRORS:
        values=[r for r in rows if r['condition']==condition]
        result={}
        for name in METHODS:
            a=sum(r['accepted'][name] for r in values);k=sum(r['accepted'][name] and r['valid'] is not True for r in values)
            result[name]={'scenarios':len(values),'accepted':a,'coverage':a/len(values),'invalid_or_unknown_accepted':k,
                'invalid_reuse':k/a if a else None,'family_adjusted_upper95':float(beta.ppf(1-.05/(len(ERRORS)*len(METHODS)),k+1,a-k)) if k<a else 1.}
        report[condition]=result
    save_json(out/'rows.json',rows)
    save_json(out/'report.json',{'conditions':ERRORS,'methods':report,'registration_sha256':sha(REG),
        'error_parameters_measured_from_real_sensors':False,'causal_input_corruption':True,'label_world_clean_simulated_truth':True,
        'interpretation':'independent scenario sensitivity to specified synthetic errors, not calibrated deployment robustness'})
    log(stage='robustness_complete',scenarios=reg['slot_scenarios'])

if __name__=='__main__':main()
