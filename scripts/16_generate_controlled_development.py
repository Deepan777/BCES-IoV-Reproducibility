#!/usr/bin/env python3
"""Registered controlled traces and same-future-world decision diagnostics."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from dataclasses import asdict,replace
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bces.geometry.drift import DriftScales
from bces.oracle.observational import BEHAVIOR_IDS
from bces.oracle.planner import PlannerConfig
from bces.oracle.validity import ValidityThresholds
from bces.simulation.branching import _populate
from bces.simulation.scenario_builder import FAMILIES,scenario_spec
from bces.simulation.sumo_adapter import sumo_version
from bces.simulation.controlled_contract import (start_controlled,receiver_state,centered_world,
    ego_from_sample,local_observations,make_message)
from bces.simulation.controlled_decisions import ControlledDecisionPlanner,reference_contract,decision_pair
from bces.utils.budget import Budget,require_headroom,build_budget_report
from bces.utils.hashing import canonical_json_hash,sha256_file
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic

CONFIG=ROOT/'configs/evaluation/study_b_controlled_v2.yaml'


def sample_scenario(seed,config):
    rng=np.random.default_rng(seed)
    family=FAMILIES[int(rng.integers(len(FAMILIES)))]
    spec=scenario_spec(family)
    stress=bool(rng.random()<config['stress_probability'])
    jitter=config['stress_actor_position_jitter_m'] if stress else config['nominal_actor_position_jitter_m']
    speed_jitter=config['pedestrian_speed_jitter_mps'] if spec.pedestrian else config['vehicle_speed_jitter_mps']
    spec=replace(spec,actor_depart_position_m=max(5.,spec.actor_depart_position_m+float(rng.uniform(*jitter))),
                 actor_depart_speed_mps=max(.5,spec.actor_depart_speed_mps+float(rng.uniform(*speed_jitter))))
    return spec,{'benchmark_stratum':'stress' if stress else 'nominal',
        'ego_initial_speed_mps':float(rng.uniform(*config['ego_initial_speed_mps'])),
        'ego_post_reference_acceleration_mps2':float(rng.choice(config['ego_post_reference_accelerations_mps2'])),
        'braking_event_delay_s':float(rng.uniform(*config['braking_event_delay_s']))}


def collect_trace(seed,spec,parameters,config,horizon):
    connection=start_controlled(network=ROOT/config['network'],label=f'controlled_development_{seed}',
                                seed=seed,step_s=config['step_s'])
    frames={}; previous=None; event_applied=False
    end=config['reference_time_s']+max(config['ages_s'])+horizon
    try:
        _populate(connection,spec)
        for _ in range(round(end/config['step_s'])):
            now=connection.simulation.getTime()
            if 'ego' in connection.vehicle.getIDList():
                age=max(0.,now-config['reference_time_s'])
                speed=max(0.,min(20.,parameters['ego_initial_speed_mps']+
                                parameters['ego_post_reference_acceleration_mps2']*age))
                connection.vehicle.setSpeed('ego',speed)
            if spec.uncontrolled_signal:
                for signal in connection.trafficlight.getIDList():
                    current=connection.trafficlight.getRedYellowGreenState(signal)
                    connection.trafficlight.setRedYellowGreenState(signal,'G'*len(current))
            if (spec.braking_event and not event_applied and
                    now+1e-9>=config['reference_time_s']+parameters['braking_event_delay_s'] and
                    spec.actor_id in connection.vehicle.getIDList()):
                connection.vehicle.slowDown(spec.actor_id,0.,1.)
                event_applied=True
            connection.simulationStep()
            timestamp=round(connection.simulation.getTime()*1000)
            receiver=None
            if 'ego' in connection.vehicle.getIDList():
                receiver=receiver_state(connection,previous); previous=receiver
            frames[timestamp]={'receiver':receiver,'objects':centered_world(connection)}
    finally:
        connection.close()
    return frames


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--max-scenes',type=int)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config_path=args.config.resolve()
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    output=(args.output or ROOT/config['output_root']).resolve()
    if not output.is_relative_to(ROOT/'outputs/study_b'):
        raise ValueError('output outside Study B')
    state=git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean source required before generation')
    audit_path=ROOT/config['contract_audit']
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if (audit['status']!='PASS' or audit['approved_branch_initialization']!='deterministic_complete_prefix_replay'
            or audit['contract_source_sha256']!=sha256_file(ROOT/'bces/simulation/controlled_contract.py')):
        raise RuntimeError('current controlled contract must pass its replay audit')
    if sumo_version()!=audit['sumo_version']: raise RuntimeError('SUMO version changed')
    if (output/'run_manifest.json').exists(): raise FileExistsError('completed data are immutable')
    budget=Budget(workspace_root=ROOT,data_root=ROOT/'data'); require_headroom(budget,incoming_bytes=150_000_000)
    planner=ControlledDecisionPlanner(PlannerConfig.from_yaml(ROOT/config['planner']))
    raw_thresholds=yaml.safe_load((ROOT/config['validity']).read_text(encoding='utf-8'))
    thresholds=ValidityThresholds(*[raw_thresholds[k] for k in ('max_cost_regret','max_cached_risk','max_trajectory_deviation_m')])
    scales=DriftScales(*config['drift_scales'])
    assignments=[split for split,count in config['partitions'].items() for _ in range(count)]
    if args.max_scenes is not None:
        if args.max_scenes<1: raise ValueError('positive max-scenes required')
        assignments=assignments[:args.max_scenes]
    contract={'config_sha256':sha256_file(config_path),'source_git':state['commit'],
              'audit_sha256':sha256_file(audit_path),'assignments':assignments}
    output.mkdir(parents=True,exist_ok=True)
    path=output/'generation_contract.json'
    if path.exists() and json.loads(path.read_text())!=contract: raise RuntimeError('cannot resume changed experiment')
    if not path.exists(): write_json_atomic(path,contract)
    counts=Counter(); artifacts={}; rejections=Counter()
    for index,split in enumerate(assignments):
        seed=config['scenario_seed_start']+index; scenario=str(seed)
        path=output/f'{split}_{scenario}.json.gz'
        if path.exists():
            with gzip.open(path,'rt',encoding='utf-8') as f: scene=json.load(f)
            if scene['contract_sha256']!=canonical_json_hash(contract): raise RuntimeError('resume contract mismatch')
        else:
            spec,parameters=sample_scenario(seed,config)
            scene={'scenario_id':scenario,'split':split,'family':spec.family,
                'benchmark_stratum':parameters['benchmark_stratum'],'sampled_parameters':parameters,
                'scenario_spec':asdict(spec),'contract_sha256':canonical_json_hash(contract),
                'references':[],'points':[],'rejections':{},'traces':{}}
            failures=Counter(); reference_time=round(config['reference_time_s']*1000)
            hidden=(('person:' if spec.pedestrian else 'vehicle:')+spec.actor_id,) if spec.hidden_from_local else ()
            references={}
            variants=config.get('receiver_drift_variants',[parameters['ego_post_reference_acceleration_mps2']])
            for variant,acceleration in enumerate(variants):
                variant_parameters={**parameters,'ego_post_reference_acceleration_mps2':float(acceleration)}
                trace=collect_trace(seed,spec,variant_parameters,config,planner.config.horizon_s)
                future={t:frame['objects'] for t,frame in trace.items()}
                scene['traces'][str(variant)]={str(t):{'receiver':f['receiver'],'objects':[asdict(o) for o in f['objects']]} for t,f in trace.items()}
                def observed(timestamp):
                    frame=trace[timestamp]
                    if frame['receiver'] is None: raise ValueError('receiver_unavailable')
                    ego=ego_from_sample(frame['receiver'])
                    local=local_observations(frame['objects'],ego,range_m=config['local_range_m'],hidden_ids=hidden)
                    cooperative=tuple(o for o in frame['objects'] if math.hypot(o.x_m-ego.x_m,o.y_m-ego.y_m)<=config['cooperative_range_m'])
                    return ego,local,make_message(cooperative,timestamp,message_id=seed)
                for behavior in BEHAVIOR_IDS:
                    try:
                        ego,local,message=observed(reference_time)
                        ref,query,cached=reference_contract(scenario=scenario,split=split,behavior=behavior,
                            timestamp=reference_time,ego=ego,local=local,message=message,planner=planner,scales=scales)
                    except (ValueError,RuntimeError) as exc:
                        failures[f'reference:{behavior}:{type(exc).__name__}:{exc}']+=1
                        continue
                    identity=ref['reference_id']
                    if identity in references and references[identity]!=ref:
                        raise AssertionError('drift variants changed the frozen reference payload/query')
                    references[identity]=ref
                    for age in config['ages_s']:
                        timestamp=reference_time+round(age*1000)
                        try:
                            ego,local,message=observed(timestamp)
                            point=decision_pair(reference=ref,query=query,cached_reference=cached,
                                ego=ego,local=local,fresh_message=message,future_world=future,
                                planner=planner,thresholds=thresholds)
                        except (ValueError,RuntimeError,KeyError) as exc:
                            failures[f'point:{behavior}:{type(exc).__name__}:{exc}']+=1
                            continue
                        point['point_id']+=f':drift{variant}'
                        point['receiver_drift_variant']=variant
                        point['benchmark_stratum']=parameters['benchmark_stratum']
                        scene['points'].append(point)
            scene['references']=list(references.values())
            scene['rejections']=dict(failures)
            with gzip.open(path,'xt',encoding='utf-8') as f: json.dump(scene,f,separators=(',',':'),sort_keys=True,allow_nan=False)
        for point in scene['points']:
            counts[f'{split}:points']+=1; counts[f'{split}:valid']+=int(point['valid'])
            if point['cache_age_s']==0:
                counts[f'{split}:origin_points']+=1; counts[f'{split}:origin_invalid']+=int(not point['valid'])
        counts[f'{split}:scenarios']+=1; counts[f'{split}:references']+=len(scene['references'])
        counts['family:'+scene['family']]+=1; counts['stratum:'+scene['benchmark_stratum']]+=1
        artifacts[path.name]=sha256_file(path); rejections.update(scene['rejections'])
        if (index+1)%10==0 or index+1==len(assignments):
            require_headroom(budget)
            print(json.dumps({'processed':index+1,'total':len(assignments),'points':sum(v for k,v in counts.items() if k.endswith(':points'))}),flush=True)
    report={'schema_version':1,'status':'PASS','scientific_success':False,'manuscript_allowed':False,
        'scope':'development_only','original_test_accessed':False,'causal_derivatives':True,
        'receiver_role_verified':True,'deployed_receiver_evidence':False,
        'evaluation_world':'recorded_simulated_future_open_loop_not_closed_loop',
        'receiver_setup':'simulated_ego_with_declared_ideal_local_sensor_and_occlusion',
        'counts':dict(counts),'rejections':dict(rejections),'artifact_sha256':artifacts,
        'config_sha256':sha256_file(config_path),'contract_audit_sha256':sha256_file(audit_path),
        'maximum_points_per_scenario':5*len(config['ages_s'])*len(config.get('receiver_drift_variants',[0])),
        'policy_hash':planner.policy_hash,'git':state,'completed_utc':utc_now(),
        'budget':build_budget_report(budget)['measurements'],
        'limitations':['fixed non-ego future trace does not react to counterfactual ego plans',
            'labels are discrete-time open-loop trajectory costs, not closed-loop collisions',
            'nominal and stress sampling distributions are designed, not measured traffic frequencies',
            'motion uncertainty and risk-control envelopes are not established',
            'margin sidecar wire binding and complete deployment accounting remain unfinished']}
    report['report_sha256']=canonical_json_hash(report)
    write_json_atomic(output/'run_manifest.json',report)
    print(json.dumps({'status':'PASS','counts':report['counts'],'rejections':report['rejections']},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
