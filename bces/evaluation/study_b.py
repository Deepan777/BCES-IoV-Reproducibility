"""Explicit feature allowlists and scenario-aware Study-B evaluation."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bces.utils.hashing import sha256_file


def reference_features(reference):
    """Only exact payload/query fields; exclude IDs, global coordinates and time."""
    batch = reference['frozen_input']['batch']
    state = np.asarray(batch['reference_state'],float)
    objects = np.asarray(batch['object_features'],float).copy()
    mask = np.asarray(batch['object_mask'],bool)
    cosine, sine = math.cos(state[3]), math.sin(state[3])
    for start in (0,2):
        x,y = objects[:,start].copy(),objects[:,start+1].copy()
        objects[:,start],objects[:,start+1] = cosine*x+sine*y,-sine*x+cosine*y
    objects[~mask] = 0
    path = np.asarray(batch['reference_path'],float)
    dx,dy = path[:,0]-state[0],path[:,1]-state[1]
    path_features = np.column_stack([cosine*dx+sine*dy,-sine*dx+cosine*dy,path[:,2],
        np.sin(path[:,3]-state[3]),np.cos(path[:,3]-state[3]),path[:,4],path[:,5]])
    context = [*state[2:6],*batch['drift_scales'],batch['identifiers'][0],batch['identifiers'][1],
               batch['object_count'],batch['occlusion_proxy'],batch['estimated_delay_s'],batch['map_context_flags']]
    return np.concatenate([objects.ravel(),mask.astype(float),path_features.ravel(),context]).astype(np.float32)


def load_development(root: Path):
    manifest = json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('original_test_accessed') is not False or not manifest.get('causal_derivatives'):
        raise ValueError('causal development manifest required')
    references,points,seen = {},[],set()
    for name,expected in sorted(manifest['artifact_sha256'].items()):
        path = root/name
        if path.parent != root or sha256_file(path) != expected:
            raise ValueError('source artifact hash/path mismatch')
        with gzip.open(path,'rt',encoding='utf-8') as handle:
            scene = json.load(handle)
        if scene['split'] not in {'train','validation','calibration'}:
            raise PermissionError('test data prohibited in development')
        if scene['scenario_id'] in seen:
            raise ValueError('duplicate scenario across partitions')
        seen.add(scene['scenario_id'])
        for ref in scene['references']:
            if ref['reference_id'] in references or ref['split'] != scene['split']:
                raise ValueError('reference identity/split mismatch')
            references[ref['reference_id']] = ref
        for point in scene['points']:
            if point['split'] != scene['split'] or point['scenario_id'] != scene['scenario_id']:
                raise ValueError('point identity/split mismatch')
            points.append(point)
    return references,points,manifest


def origin_validity(reference_ids,points):
    """All observed origin futures must be valid; absent origins abstain.

Several drift continuations share a frozen reference. Last-row-wins labels
would silently discard conflicting origin futures and depend on row order.
"""
    values={key:[] for key in reference_ids}
    for point in points:
        if point['cache_age_s']==0:
            values[point['reference_id']].append(bool(point['valid']))
    return np.asarray([bool(values[key]) and all(values[key]) for key in reference_ids],np.float32)


def feature_arrays(references, points):
    ids = sorted(references)
    index = {key:i for i,key in enumerate(ids)}
    base = np.stack([reference_features(references[key]) for key in ids])
    margin = np.asarray([references[key]['observable_margin_features'] for key in ids],np.float32)
    ref_index = np.array([index[row['reference_id']] for row in points])
    drift = np.asarray([row['normalized_drift'] for row in points],np.float32)
    current = np.asarray([row['observable_current_features'] for row in points],np.float32)
    return {'reference_ids':ids,'reference_index':ref_index,'base_reference':base,
            'margin_reference':np.column_stack([base,margin]),'drift':drift,
            'frozen_inputs':np.column_stack([base[ref_index],drift]),
            'reference_margins':np.column_stack([base[ref_index],margin[ref_index],drift]),
            'current_receiver':np.column_stack([base[ref_index],margin[ref_index],drift,current])}


def selective_metrics(valid, accepted, scenarios, *, seed=140926, replicates=1000):
    valid,accepted = np.asarray(valid,bool),np.asarray(accepted,bool)
    scenarios = np.asarray(scenarios)
    names = np.unique(scenarios)
    a = np.array([accepted[scenarios==key].sum() for key in names])
    u = np.array([(accepted & ~valid)[scenarios==key].sum() for key in names])
    n = np.array([(scenarios==key).sum() for key in names])
    count,unsafe = int(a.sum()),int(u.sum())
    draws = np.random.default_rng(seed).integers(len(names),size=(replicates,len(names)))
    den,num = a[draws].sum(axis=1),u[draws].sum(axis=1)
    rates = np.where(den>0,num/np.maximum(den,1),1.)
    return {'total':len(valid),'accepted':count,'unsafe_accepted':unsafe,
            'coverage':count/len(valid),'uar':unsafe/count if count else None,
            'scenario_count':len(names),'accepted_scenarios':int((a>0).sum()),
            'bootstrap_uar_interval_descriptive':np.quantile(rates,[.025,.975]).tolist(),
            'bootstrap_is_not_finite_sample_certification':True}


def scenario_ratio_upper(valid, accepted, scenarios, *, maximum_points, alpha=.05, comparisons=1):
    """Union-bound Hoeffding ratio for E[unsafe count]/E[accepted count].

    Independent identically distributed scenarios and a fixed rule are assumed.
    Each count is normalized by a predeclared upper bound on points per scenario.
    Dependence inside each scenario is unrestricted. May correctly return 1.
    """
    if maximum_points <= 0 or not 0 < alpha < 1 or comparisons < 1:
        raise ValueError('invalid concentration parameters')
    valid,accepted,scenarios = np.asarray(valid,bool),np.asarray(accepted,bool),np.asarray(scenarios)
    names = np.unique(scenarios)
    if not len(names):
        raise ValueError('at least one scenario required')
    a,u = [],[]
    for name in names:
        select = scenarios==name
        if select.sum() > maximum_points:
            raise ValueError('registered per-scenario bound exceeded')
        a.append(accepted[select].sum()/maximum_points)
        u.append((accepted[select]&~valid[select]).sum()/maximum_points)
    epsilon = math.sqrt(math.log(2*comparisons/alpha)/(2*len(names)))
    denominator = float(np.mean(a))-epsilon
    return min(1.,(float(np.mean(u))+epsilon)/denominator) if denominator>0 else 1.


def predictive_metrics(valid, probability, scenarios):
    valid,probability = np.asarray(valid,bool),np.asarray(probability,float)
    result = {'auroc':float(roc_auc_score(valid,probability)) if len(np.unique(valid))>1 else None,
              'auprc':float(average_precision_score(valid,probability)), 'matched_coverage':{}}
    order = np.argsort(-probability,kind='stable')
    for fraction in (.03,.1,.2,.5):
        accepted = np.zeros(len(valid),bool)
        accepted[order[:max(1,int(len(valid)*fraction))]] = True
        result['matched_coverage'][str(fraction)] = selective_metrics(valid,accepted,scenarios)
    return result
