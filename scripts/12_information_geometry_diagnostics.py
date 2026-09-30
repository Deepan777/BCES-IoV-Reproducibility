#!/usr/bin/env python3
"""Development-only information and empirical representability experiment."""
from __future__ import annotations
import gzip
import argparse
import json
import os
import sys
from collections import Counter,defaultdict
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')

import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from bces.evaluation.decision_geometry import fit_zero_error_surface, fit_zero_error_ttl
from bces.evaluation.study_b import feature_arrays, load_development, predictive_metrics, selective_metrics, scenario_ratio_upper
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.utils.budget import Budget,require_headroom,build_budget_report
from bces.utils.hashing import canonical_json_hash,sha256_file
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic

CONFIG = ROOT/'configs/evaluation/study_b_development_v1.yaml'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    output = ROOT/config['diagnostic_root']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('diagnostic output is immutable')
    state = git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('commit diagnostics before evaluation')
    budget = Budget(workspace_root=ROOT,data_root=ROOT/'data')
    require_headroom(budget,incoming_bytes=50_000_000)
    refs,points,source = load_development(ROOT/config['output_root'])
    arrays = feature_arrays(refs,points)
    labels = np.array([r['valid'] for r in points],bool)
    splits = np.array([r['split'] for r in points])
    scenarios = np.array([r['scenario_id'] for r in points])
    masks = {key:splits==key for key in ('train','validation','calibration')}
    output.mkdir(parents=True,exist_ok=True)
    reports,predictions = {},{}
    with threadpool_limits(limits=2):
        for view in ('frozen_inputs','reference_margins','current_receiver'):
            features = arrays[view]
            candidates = []
            for leaves in config['classifier_leaf_counts']:
                model = HistGradientBoostingClassifier(max_iter=config['classifier_iterations'],
                    max_leaf_nodes=leaves,learning_rate=.07,l2_regularization=1.,min_samples_leaf=40,
                    early_stopping=False,random_state=config['seed'])
                model.fit(features[masks['train']],labels[masks['train']])
                probability = model.predict_proba(features[masks['validation']])[:,1]
                validation = predictive_metrics(labels[masks['validation']],probability,scenarios[masks['validation']])
                candidates.append((validation['auprc'],leaves,model,validation))
                print(json.dumps({'view':view,'leaves':leaves,'validation_auroc':validation['auroc'],'validation_auprc':validation['auprc']}),flush=True)
            _,leaves,chosen,validation = max(candidates,key=lambda row:(row[0],-row[1]))
            all_probability = chosen.predict_proba(features)[:,1]
            predictions[view] = all_probability
            selected_validation = []
            for threshold in config['candidate_thresholds']:
                accepted = all_probability[masks['validation']]>=threshold
                values = selective_metrics(labels[masks['validation']],accepted,scenarios[masks['validation']])
                selected_validation.append({'threshold':threshold,**values})
            eligible = [r for r in selected_validation if r['accepted']>=config['minimum_accepted'] and r['uar']<=config['target_uar']]
            threshold = max(eligible,key=lambda r:r['coverage'])['threshold'] if eligible else None
            cal = masks['calibration']
            accept = all_probability[cal]>=threshold if threshold is not None else np.zeros(cal.sum(),bool)
            calibration = selective_metrics(labels[cal],accept,scenarios[cal])
            calibration['scenario_concentration_uar_upper'] = scenario_ratio_upper(labels[cal],accept,scenarios[cal],maximum_points=config.get('maximum_points_per_scenario',5*len(config['ages_s'])),comparisons=3)
            calibration['selected_threshold'] = threshold
            calibration['finite_sample_target_met'] = threshold is not None and calibration['scenario_concentration_uar_upper']<=config['target_uar']
            reports[view] = {'selected_leaves':leaves,'feature_count':features.shape[1],
                'selection_partition':'validation','validation':validation,'validation_thresholds':selected_validation,
                'calibration_fixed_rule':calibration,
                'candidates':[{'leaves':row[1],'validation':row[3]} for row in candidates],
                'eligible_as_sender_surface_input':view!='current_receiver'}
    by_ref = defaultdict(list)
    for index,point in enumerate(points):
        by_ref[point['reference_id']].append(index)
    selection = sorted(by_ref,key=lambda key:canonical_json_hash({'geometry_study_b':key}))[:config['geometry_reference_limit']]
    geometry = []
    for index,key in enumerate(selection,1):
        selected = by_ref[key]
        drift = arrays['drift'][selected].astype(float)
        valid = labels[selected]
        surface = fit_zero_error_surface(drift,valid,NORMAL_CODEBOOK_V1,time_limit=config['geometry_time_limit_s'])
        ttl = fit_zero_error_ttl([points[i]['cache_age_s'] for i in selected],valid)
        geometry.append({'reference_id':key,'split':refs[key]['split'],'points':len(selected),'valid':int(valid.sum()),
                         'surface':surface,'ttl':ttl,'point_ids':[points[i]['point_id'] for i in selected]})
        if index%25==0:
            print(json.dumps({'geometry_processed':index,'total':len(selection)}),flush=True)
    taxonomy = Counter()
    for point in points:
        for cause,violated in point['violations'].items():
            taxonomy[cause] += int(violated)
        if point['cache_age_s']==0:
            taxonomy['origin_total'] += 1
            taxonomy['origin_invalid'] += int(not point['valid'])
    with gzip.open(output/'predictions.jsonl.gz','xt',encoding='utf-8') as handle:
        for i,point in enumerate(points):
            row = {'point_id':point['point_id'],'reference_id':point['reference_id'],'scenario_id':point['scenario_id'],
                'split':point['split'],'valid':point['valid'],'probability':{name:float(p[i]) for name,p in predictions.items()}}
            handle.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
    write_json_atomic(output/'geometry.json',{'scope':'label_informed_sample_fit_only','references':geometry})
    totals = {'references':len(geometry),'points':sum(r['points'] for r in geometry),'valid':sum(r['valid'] for r in geometry),
        'surface_accepted_valid':sum(sum(r['surface']['accepted']) for r in geometry),
        'ttl_accepted_valid':sum(sum(r['ttl']['accepted']) for r in geometry),
        'surface_optimal_solutions':sum(r['surface']['optimal'] for r in geometry),
        'surface_abstentions':sum(r['surface']['abstain'] for r in geometry)}
    report = {'schema_version':1,'status':'PASS','scientific_success':False,'scope':'development_only',
        'original_test_accessed':False,'causal_input_contract':True,'information':reports,'geometry':totals,
        'violation_taxonomy':dict(taxonomy),'source_manifest_sha256':sha256_file(ROOT/config['output_root']/'run_manifest.json'),
        'config_sha256':sha256_file(config_path),'git':state,'completed_utc':utc_now(),
        'deployed_receiver_evidence':False,'evaluation_world':source.get('evaluation_world','unaudited_proxy'),
        'artifact_sha256':{name:sha256_file(output/name) for name in ('geometry.json','predictions.jsonl.gz')},
        'budget':build_budget_report(budget)['measurements'],'manuscript_allowed':False,
        'limitations':['representability results use the evaluated labels','development evidence is not fresh confirmation',
                      *source.get('limitations',[]),
                      *([] if source.get('receiver_role_verified') else [
                          'counterfactual target/local-view assignment is not a verified deployed receiver',
                          'evaluation perception stream is not complete physical ground truth']),
                      'finite-sample bound assumes IID scenarios; temporal/site overlap is not yet certified',
                      'current receiver features require additional computation and are not sender-time inputs',
                      'regret theorem remains conditional on justified uniform error envelopes']}
    report['report_sha256'] = canonical_json_hash(report)
    write_json_atomic(output/'diagnostics.json',report)
    print(json.dumps({'status':'PASS','geometry':totals,'violation_taxonomy':dict(taxonomy)},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
