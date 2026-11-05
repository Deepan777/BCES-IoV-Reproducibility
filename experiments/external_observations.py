"""Causal-motion, frame-paired V2X-Real annotation-view stress evaluation.

This is external to simulator training but retrospective relative to earlier
source inspections. Annotation union is partial world evidence, not exhaustive
ground truth, and nominal 10-Hz frame alignment is not a measured clock bound.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch
import yaml
from development_study import ORIGINAL,OUT,save_json,log,sha,report_metrics
from local_policies import LocalRule
from bces.geometry.drift import DriftScales,compute_drift
from bces.oracle.planner import PlannerConfig,_rect_overlap
from bces.oracle.world import WorldObject
from bces.oracle.validity import ValidityThresholds
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.simulation.controlled_decisions import ControlledDecisionPlanner,decision_pair,kinematic,route_for
from bces.simulation.controlled_contract import make_message

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('historical_external_schema',ORIGINAL/'scripts/51_run_v2xreal_position_proxy.py')
schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)
NOISE={
    'nominal':dict(position_sd_m=0.,yaw_sd_deg=0.,velocity_sd_mps=0.,miss_probability=0.,id_break_probability=0.),
    'localization':dict(position_sd_m=.25,yaw_sd_deg=1.,velocity_sd_mps=.2,miss_probability=0.,id_break_probability=0.),
    'missed_objects':dict(position_sd_m=0.,yaw_sd_deg=0.,velocity_sd_mps=0.,miss_probability=.1,id_break_probability=0.),
    'track_association':dict(position_sd_m=0.,yaw_sd_deg=0.,velocity_sd_mps=0.,miss_probability=0.,id_break_probability=.1),
    'combined':dict(position_sd_m=.25,yaw_sd_deg=1.,velocity_sd_mps=.2,miss_probability=.1,id_break_probability=.1),
}

def stable_seed(*values):return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8],'big')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-scenes',type=int);args=parser.parse_args()
    torch.set_num_threads(2)
    archive=ORIGINAL/'data/raw/v2x_real_lidar64/val.zip'
    folder=OUT/'external_annotation_views';folder.mkdir(parents=True,exist_ok=True)
    methods=('frozen_surface','frozen_scalar_ttl','flexible_mlp','flexible_tree','cam_inspired','normalized_drift','age_only','observable_cost_trigger')
    policies={m:LocalRule(m) for m in methods}
    protocol={'status':'FROZEN_BEFORE_METHOD_COMPARISON','scope':'retrospective external annotation-view evaluation with synthetic errors',
        'archive':'official V2X-Real LiDAR-64 validation ZIP','archive_sha256':sha(archive),
        'script_sha256':sha(__file__),'sample_indices':'reference every 20 frames starting 20; ages 4,10,20 frames; complete 30-frame future',
        'timing':'nominal 0.1 s per matched frame index, supported by provider sensor specification; residual synchronization error unmeasured',
        'sender_view':'roadside agent -1, its existing annotation list intersected with 40-m source-centered nominal range and 120-m receiver relevance radius',
        'receiver_view':'agent 1 if available, otherwise agent 2; existing per-agent annotation list within 30 m',
        'online_velocity':'same-view same-ID backward difference using only previous frame; static annotation gives zero; missing history rejects reference',
        'label_world':'known annotation union, receiver view precedence; label-only forward derivative allowed for newly observed tracks',
        'errors':NOISE,'outcome_tuning':False,'independent_untouched_dataset_claim':False,
        'error_parameters_are_measured_sensor_distributions':False,'full_world_safety_claim':False,
        'provider':'https://mobility-lab.seas.ucla.edu/v2x-real/','methods':methods}
    protocol_path=folder/'protocol.json'
    if protocol_path.exists():assert json.loads(protocol_path.read_text())==json.loads(json.dumps(protocol))
    else:save_json(protocol_path,protocol)
    zf=zipfile.ZipFile(archive)
    grouped=defaultdict(lambda:defaultdict(dict))
    for name in zf.namelist():
        if name.endswith('.yaml'):
            parts=Path(name).parts
            grouped[parts[-3]][parts[-2]][int(Path(name).stem)]=name
    counts=Counter();rows=[];alignment=[]
    selected=sorted(grouped)
    if args.max_scenes is not None:selected=selected[:args.max_scenes]
    planner=ControlledDecisionPlanner(PlannerConfig.from_yaml(ORIGINAL/'configs/planner/kinematic.yaml'))
    scales=DriftScales(40,5,5,.7,3,.1,4)
    thresholds=ValidityThresholds(5,.4,2)
    for scene in selected:
        agents=grouped[scene]
        vehicle='1' if '1' in agents else '2'
        if vehicle not in agents or '-1' not in agents:
            counts['scenes_missing_pair']+=1;continue
        frames={agent:{i:yaml.load(zf.read(path),Loader=yaml.CSafeLoader) for i,path in agents[agent].items()} for agent in (vehicle,'-1')}
        common=sorted(set(frames[vehicle])&set(frames['-1']))
        if not common:counts['scenes_no_paired_indices']+=1;continue
        poses=[schema._pose(frames[vehicle][i]) for i in common]
        if max(math.dist(p[:2],poses[0][:2]) for p in poses)<=10:
            counts['static_or_low_displacement_scenes']+=1;continue
        counts['mobile_acquisition_scenes']+=1
        for i in common:
            local=frames[vehicle][i].get('vehicles',{});remote=frames['-1'][i].get('vehicles',{})
            distances=[math.dist(local[k]['location'][:2],remote[k]['location'][:2]) for k in set(local)&set(remote)]
            alignment.extend(distances);counts['paired_frames']+=1
        clean_objects={}
        def objects(agent,index,ego,radius,source,*,future_label=False):
            cache_key=(agent,index,future_label,round(ego.x_m,6),round(ego.y_m,6),radius)
            if cache_key not in clean_objects:
                frame=frames[agent].get(index)
                if frame is None:raise schema.UnknownInput('frame_missing')
                prev=frames[agent].get(index-1,{}).get('vehicles',{})
                following=frames[agent].get(index+1,{}).get('vehicles',{})
                items=[]
                for key,record in sorted(frame.get('vehicles',{}).items()):
                    xyz=record['location'];x,y=float(xyz[0]),float(xyz[1])
                    sx,sy=frame['lidar_pose'][:2]
                    if math.hypot(x-ego.x_m,y-ego.y_m)>radius or (agent=='-1' and math.hypot(x-sx,y-sy)>40):continue
                    if str(record.get('attribute','')).lower()=='static':vx=vy=0.
                    elif key in prev:
                        vx=(x-float(prev[key]['location'][0]))/.1;vy=(y-float(prev[key]['location'][1]))/.1
                    elif future_label and key in following:
                        vx=(float(following[key]['location'][0])-x)/.1;vy=(float(following[key]['location'][1])-y)/.1
                    else:raise schema.UnknownInput('object_velocity_history_missing')
                    if math.hypot(vx,vy)>45:raise schema.UnknownInput('object_velocity_implausible_or_track_switch')
                    ext=record['extent'];yaw=math.radians(float(record['angle'][1]));typ=record.get('obj_type','').lower()
                    items.append(WorldObject('source:'+str(key),x,y,vx,vy,yaw,2*float(ext[0]),2*float(ext[1]),int(any(t in typ for t in ('pedestrian','bicycle','scooter','motorcycle'))),1.,source))
                clean_objects[cache_key]=tuple(items)
            values=clean_objects[cache_key]
            sx,sy=frames[agent][index]['lidar_pose'][:2]
            return tuple(replace(o,source=source) for o in values if math.hypot(o.x_m-ego.x_m,o.y_m-ego.y_m)<=radius and (agent!='-1' or math.hypot(o.x_m-sx,o.y_m-sy)<=40))
        frame_lookup={id(f):(agent,i) for agent,ff in frames.items() for i,f in ff.items()}
        for reference_index in range(20,max(common)-50+1,20):
            for offset in (4,10,20):
                current_index=reference_index+offset
                counts['candidate_pairs']+=1
                try:
                    clean_reference_ego=schema.ego_at(frames[vehicle],reference_index)
                    clean_current_ego=schema.ego_at(frames[vehicle],current_index)
                    future={}
                    for index in range(current_index+2,current_index+31,2):
                        ego=schema.ego_at(frames[vehicle],index)
                        world={o.track_id:o for o in objects('-1',index,ego,120,'roadside_label',future_label=True)}
                        world.update({o.track_id:o for o in objects(vehicle,index,ego,120,'vehicle_label',future_label=True)})
                        future[index*100]=tuple(world.values())
                except (schema.UnknownInput,KeyError,ValueError) as exc:
                    counts['pair_unavailable:'+str(exc)]+=1;continue
                for noise_name,noise in NOISE.items():
                    def perturb(items,agent,index):
                        result=[]
                        for o in items:
                            rng=np.random.default_rng(stable_seed(scene,agent,index,o.track_id,noise_name))
                            if rng.random()<noise['miss_probability']:continue
                            track=o.track_id
                            if rng.random()<noise['id_break_probability']:track=track+f':break:{agent}:{index}'
                            angle=rng.normal(0,math.radians(noise['yaw_sd_deg']))
                            vx=o.vx_mps*math.cos(angle)-o.vy_mps*math.sin(angle)+rng.normal(0,noise['velocity_sd_mps'])
                            vy=o.vx_mps*math.sin(angle)+o.vy_mps*math.cos(angle)+rng.normal(0,noise['velocity_sd_mps'])
                            result.append(replace(o,track_id=track,x_m=o.x_m+rng.normal(0,noise['position_sd_m']),y_m=o.y_m+rng.normal(0,noise['position_sd_m']),vx_mps=vx,vy_mps=vy))
                        return tuple(result)
                    def ego_noisy(index):
                        e=schema.ego_at(frames[vehicle],index)
                        rng=np.random.default_rng(stable_seed(scene,vehicle,index,'ego',noise_name))
                        return replace(e,x_m=e.x_m+rng.normal(0,noise['position_sd_m']),y_m=e.y_m+rng.normal(0,noise['position_sd_m']),heading_rad=e.heading_rad+rng.normal(0,math.radians(noise['yaw_sd_deg'])))
                    def object_adapter(frame,ego,radius,source):
                        agent,index=frame_lookup[id(frame)]
                        return perturb(objects(agent,index,ego,radius,source),agent,index)
                    # The original schema adapter supplies fixed query fields;
                    # only online state/measurement construction is overridden.
                    original_ego=schema.ego_at;original_objects=schema.objects_at
                    try:
                        schema.objects_at=object_adapter
                        # Avoid recursive lookup: use the saved causal estimator.
                        def noisy_pose(ff,index):
                            e=original_ego(ff,index);rng=np.random.default_rng(stable_seed(scene,vehicle,index,'ego',noise_name))
                            return replace(e,x_m=e.x_m+rng.normal(0,noise['position_sd_m']),y_m=e.y_m+rng.normal(0,noise['position_sd_m']),heading_rad=e.heading_rad+rng.normal(0,math.radians(noise['yaw_sd_deg'])))
                        schema.ego_at=noisy_pose
                        reference,query,cached,message=schema.reference_for(scene,reference_index,frames[vehicle],frames['-1'],planner,scales)
                        current_ego=noisy_pose(frames[vehicle],current_index)
                        local=object_adapter(frames[vehicle][current_index],current_ego,30,'vehicle_annotation_input')
                        remote=object_adapter(frames['-1'][current_index],current_ego,120,'roadside_annotation_input')
                        fresh=make_message(remote,current_index*100,sender=schema.SENDER)
                        # Self/ego-body ambiguity is retained as a contract
                        # exclusion; a nearby object is not silently removed.
                        if any(_rect_overlap(clean_reference_ego.x_m,clean_reference_ego.y_m,clean_reference_ego.heading_rad,clean_reference_ego.length_m,clean_reference_ego.width_m,o.x_m,o.y_m,o.heading_rad,o.length_m,o.width_m,0) for o in cached):
                            raise schema.UnknownInput('initial_overlap_or_ego_identity_ambiguous')
                        z=np.array(compute_drift(query.reference_state,kinematic(current_ego,current_index*100),scales).normalized,np.float32)
                        current_features=observable_planner_features(planner,local,tuple(o.propagated(offset*.1) for o in cached),current_ego,route_for(current_ego,'keep',planner))
                        accepted={name:bool(pol.score(pol.prepare(reference),z,current_features)>=pol.threshold) for name,pol in policies.items()}
                    except (schema.UnknownInput,RuntimeError,ValueError,KeyError) as exc:
                        counts[noise_name+':reference_or_input_unavailable:'+str(exc)]+=1;continue
                    finally:schema.ego_at=original_ego;schema.objects_at=original_objects
                    try:
                        point=decision_pair(reference=reference,query=query,cached_reference=cached,ego=current_ego,local=local,fresh_message=fresh,future_world=future,planner=planner,thresholds=thresholds)
                        valid=bool(point['valid'])
                    except (RuntimeError,ValueError,KeyError) as exc:
                        point=None;valid=None;counts[noise_name+':label_unavailable:'+str(exc)]+=1
                    rows.append({'scene':scene,'reference_index':reference_index,'current_index':current_index,'noise':noise_name,'accepted':accepted,'valid':valid,
                        'regret':point['cost_regret'] if point else None,'risk':point['cached_risk'] if point else None,'deviation_m':point['trajectory_deviation_m'] if point else None})
        log(stage='external_scene',scene=scene,rows=len(rows))
    summaries={}
    for noise_name in NOISE:
        selected_rows=[r for r in rows if r['noise']==noise_name]
        if not selected_rows:continue
        y=np.array([r['valid'] is True for r in selected_rows]);scenes=np.array([r['scene'] for r in selected_rows])
        summaries[noise_name]={method:report_metrics(y,np.array([r['accepted'][method] for r in selected_rows],float),.5,scenes) for method in methods}
    save_json(folder/'rows.json',rows)
    save_json(folder/'report.json',{'protocol':protocol,'counts':dict(counts),'rows':len(rows),'methods_by_error':summaries,
        'cross_view_shared_track_position_disagreement_m_quantiles':np.quantile(alignment,[.5,.95,1]).tolist() if alignment else None,
        'unavailable_and_excluded_cases_retained_in_counts':True,'full_contract_independently_validated':False,
        'limits':['known-view annotation union is not exhaustive world truth','frame pairing does not bound hardware clock offset','injected errors are sensitivity parameters, not calibrated sensor error rates','actual detection, source coverage confidence, and driver intent are not jointly observed','few acquisition scenes limit clustered inference']})
    log(stage='external_complete',rows=len(rows),counts=dict(counts))

if __name__=='__main__':main()
