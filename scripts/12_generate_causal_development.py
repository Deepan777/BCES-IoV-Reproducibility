#!/usr/bin/env python3
"""Regenerate Study-B development examples with availability-safe motion inputs."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.drift import DriftScales
from bces.oracle.causal_diagnostics import causal_kinematic_states, observable_planner_features
from bces.oracle.observational import _snapshots
from bces.oracle.observational_v2 import BEHAVIORS, _focal_rows, _world_at, _ego_from_state, build_reference, observed_validity_point, route_for
from bces.oracle.planner import KinematicPlanner, PlannerConfig
from bces.oracle.vectorized_planner import VectorizedDiagnosticPlanner
from bces.oracle.view_association import AssociationConfig, associate_replay_views, audit_receiver_roles
from bces.oracle.validity import ValidityThresholds
from bces.utils.budget import Budget, build_budget_report, require_headroom
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic

CONFIG = ROOT / 'configs/evaluation/study_b_development_v1.yaml'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--max-scenes', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    output = (args.output or ROOT/config['output_root']).resolve()
    if not output.is_relative_to(ROOT/'outputs/study_b'):
        raise ValueError('output must stay within outputs/study_b')
    state = git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('commit the protocol and source before generating evidence')
    if (output/'run_manifest.json').exists():
        raise FileExistsError('completed generation is immutable')
    budget = Budget(workspace_root=ROOT, data_root=ROOT/'data')
    require_headroom(budget, incoming_bytes=50_000_000)
    before = build_budget_report(budget)['measurements']
    planner = VectorizedDiagnosticPlanner(PlannerConfig.from_yaml(ROOT/config['planner']))
    associated = config.get('association_repair', False)
    association_config = AssociationConfig(**config.get('association', {}))
    if associated:
        planner.policy_hash = int(canonical_json_hash({'base_policy':planner.policy_hash,
            'association_source_sha256':sha256_file(ROOT/'bces/oracle/view_association.py'),
            'association_config':config.get('association', {})})[:8],16)
    validity = yaml.safe_load((ROOT/config['validity']).read_text(encoding='utf-8'))
    thresholds = ValidityThresholds(validity['max_cost_regret'], validity['max_cached_risk'], validity['max_trajectory_deviation_m'])
    maximum = int(validity['maximum_objects_per_source'])
    scales = DriftScales(*config['drift_scales'])
    review = json.loads((ROOT/config['review']).read_text(encoding='utf-8'))
    records = sorted([r for r in review['records'] if r['split'] in config['partitions'] and r.get('valid_windows',0)>0],
                     key=lambda r: canonical_json_hash({'study_b_selection':r['scenario_id']}))
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError('max-scenes must be positive')
        records = records[:args.max_scenes]
    if len({str(r['scenario_id']) for r in records}) != len(records):
        raise ValueError('scenario IDs must be unique across partitions')
    output.mkdir(parents=True, exist_ok=True)
    contract = {'config_sha256':sha256_file(config_path), 'source_git':state['commit'],
                'review_sha256':sha256_file(ROOT/config['review']),
                'scenarios':[str(r['scenario_id']) for r in records]}
    plan = output/'generation_contract.json'
    if plan.exists() and json.loads(plan.read_text(encoding='utf-8')) != contract:
        raise RuntimeError('cannot resume under changed source/config/scenario plan')
    if not plan.exists():
        write_json_atomic(plan, contract)
    counts, rejected, artifacts, source_hashes = Counter(), Counter(), {}, {}
    started = utc_now()
    for number, record in enumerate(records,1):
        split, scenario = record['split'], str(record['scenario_id'])
        assert split != 'test'
        output_file = output/f'{split}_{scenario}.json.gz'
        relative = Path(record['source_file'])
        raw = ROOT/config['raw_root']
        paths = {'ego':raw/relative,
                 'vehicle':raw/'vehicle'/relative.parts[1]/relative.name,
                 'infrastructure':raw/'infrastructure'/relative.parts[1]/relative.name}
        hashes = {key:sha256_file(path) for key,path in paths.items()}
        source_hashes[scenario] = {key:{'path':str(path.relative_to(ROOT)), 'sha256':hashes[key]} for key,path in paths.items()}
        if output_file.exists():
            with gzip.open(output_file,'rt',encoding='utf-8') as handle:
                scene = json.load(handle)
            if scene['source_sha256'] != hashes or scene['contract_sha256'] != canonical_json_hash(contract):
                raise RuntimeError('resume source mismatch')
        else:
            ego_s, vehicle_s, infra_s = (_snapshots(paths[key]) for key in ('ego','vehicle','infrastructure'))
            focal_id = str(record['focal_actor_id'])
            role_audit = audit_receiver_roles(ego_s, focal_id)
            association_audit = None
            if associated:
                vehicle_s, infra_s, association_audit = associate_replay_views(
                    ego_s, vehicle_s, infra_s, focal_id, association_config)
            states = causal_kinematic_states(_focal_rows(ego_s,focal_id), config['history_frames'])
            times = sorted(states)
            scene = {'scenario_id':scenario,'split':split,'intersection_id':record['intersection_id'],
                     'source_sha256':hashes,'contract_sha256':canonical_json_hash(contract),
                     'references':[],'points':[],'rejections':{},
                     'receiver_role_audit':role_audit,'association_audit':association_audit}
            failures = Counter()
            reference_time = times[int(config['history_frames'])]
            for behavior in BEHAVIORS:
                try:
                    reference, _, query, cached, template = build_reference(
                        scenario_id=scenario, intersection_id=record['intersection_id'], split=split,
                        behavior=behavior, focal_id=focal_id, reference_timestamp=reference_time,
                        states=states, ego_snapshots=ego_s, vehicle_snapshots=vehicle_s,
                        infrastructure_snapshots=infra_s, scales=scales, planner=planner, maximum_objects=maximum)
                except RuntimeError:
                    failures[f'reference_infeasible:{behavior}'] += 1
                    continue
                reference['input_contract'] = 'causal_past_only_study_b_v1'
                if associated:
                    reference['input_contract'] = 'causal_associated_counterfactual_proxy_v2'
                reference['label_source'] = 'counterfactual_cross_view_perception_proxy'
                reference['receiver_role_verified'] = role_audit['receiver_role_verified']
                reference['motion_history_latest_ms'] = reference_time
                reference['motion_history_earliest_ms'] = times[0]
                ref_ego = _ego_from_state(states[reference_time],template)
                ref_local = _world_at(vehicle_s,reference_time,ref_ego,focal_id,maximum,'vehicle')
                reference['observable_margin_features'] = observable_planner_features(planner,ref_local,cached,ref_ego,route_for(ref_ego,behavior,planner))
                reference['additional_query_float32_bytes'] = 4*len(reference['observable_margin_features'])
                scene['references'].append(reference)
                for age in config['ages_s']:
                    timestamp = reference_time+round(age*1000)
                    point = observed_validity_point(reference=reference,query=query,cached_reference=cached,
                        template=template,current_timestamp=timestamp,states=states,focal_id=focal_id,
                        ego_snapshots=ego_s,vehicle_snapshots=vehicle_s,infrastructure_snapshots=infra_s,
                        planner=planner,thresholds=thresholds,maximum_objects=maximum)
                    if point is None:
                        failures[f'point_missing_or_infeasible:{behavior}'] += 1
                        continue
                    point['label_source'] = 'counterfactual_cross_view_perception_proxy'
                    ego = _ego_from_state(states[timestamp],template)
                    local = _world_at(vehicle_s,timestamp,ego,focal_id,maximum,'vehicle')
                    point['observable_current_features'] = observable_planner_features(planner,local,
                        tuple(item.propagated(age) for item in cached),ego,route_for(ego,behavior,planner))
                    point['observable_features_latest_ms'] = timestamp
                    point['motion_history_latest_ms'] = timestamp
                    point['violations'] = {
                        'regret':point['cost_regret']>thresholds.max_cost_regret+1e-10,
                        'absolute_risk':point['cached_risk']>thresholds.max_cached_risk+1e-10,
                        'deviation':point['trajectory_deviation_m']>thresholds.max_trajectory_deviation_m+1e-10}
                    if point['valid'] == any(point['violations'].values()):
                        raise AssertionError('validity decomposition mismatch')
                    scene['points'].append(point)
            scene['rejections'] = dict(failures)
            with gzip.open(output_file,'xt',encoding='utf-8') as handle:
                json.dump(scene,handle,separators=(',',':'),sort_keys=True,allow_nan=False)
        for point in scene['points']:
            counts[f'{split}:points'] += 1
            counts[f'{split}:valid'] += int(point['valid'])
            if point['cache_age_s'] == 0:
                counts[f'{split}:origin_points'] += 1
                counts[f'{split}:origin_invalid'] += int(not point['valid'])
        counts[f'{split}:references'] += len(scene['references'])
        counts[f'{split}:scenarios'] += 1
        for key,value in scene['receiver_role_audit']['counts'].items():
            counts[f'role_audit:{key}'] += value
        counts['role_audit:verified_receiver_scenarios'] += int(scene['receiver_role_audit']['receiver_role_verified'])
        if scene['association_audit']:
            for key,value in scene['association_audit']['counts'].items():
                counts[f'association:{key}'] += value
        rejected.update(scene['rejections'])
        artifacts[output_file.name] = sha256_file(output_file)
        if number%10==0 or number==len(records):
            require_headroom(budget)
            print(json.dumps({'processed':number,'total':len(records),'points':sum(v for k,v in counts.items() if k.endswith(':points'))}),flush=True)
    report = {'schema_version':1,'status':'PASS','scientific_success':False,'scope':'development_only',
              'original_test_accessed':False,'causal_derivatives':True,'counts':dict(counts),'rejections':dict(rejected),
              'artifact_sha256':artifacts,'source_files':source_hashes,'config_sha256':sha256_file(config_path),
              'association_repair':associated,'evaluation_world':'ego_view_perception_proxy_not_ground_truth',
              'receiver_setup':'forecasting_target_with_assigned_other_vehicle_observations_counterfactual',
              'deployed_receiver_evidence':False,'manuscript_allowed':False,
              'additional_margin_query_bytes':116,'current_features_sender_eligible':False,
              'policy_hash':planner.policy_hash,'git':state,'started_utc':started,'completed_utc':utc_now(),
              'budget_before':before,'budget_after':build_budget_report(budget)['measurements']}
    report['report_sha256'] = canonical_json_hash(report)
    write_json_atomic(output/'run_manifest.json',report)
    print(json.dumps({'status':report['status'],'counts':report['counts'],'output':str(output)},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
