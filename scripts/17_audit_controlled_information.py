#!/usr/bin/env python3
"""Separate same-age state effects, input ambiguity and sampled geometry limits."""
from __future__ import annotations

import json
import sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bces.evaluation.study_b import load_development
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.utils.hashing import canonical_json_hash,sha256_file
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic


def exact_drift_key(point):
    # This is the precision used by the actual learning features. Signed zeros
    # are the same numerical input. Reference context is fixed within a group.
    return tuple(float(x) for x in np.asarray(point['normalized_drift'],np.float32))


def main():
    root=ROOT/'outputs/study_b/controlled_development_v2'
    diagnostic_root=ROOT/'outputs/study_b/diagnostics_controlled_v2'
    output=ROOT/'outputs/study_b/controlled_information_audit_v2.json'
    if output.exists(): raise FileExistsError('completed audit is immutable')
    state=git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean audit source required')
    refs,points,manifest=load_development(root)
    diagnostic=json.loads((diagnostic_root/'diagnostics.json').read_text(encoding='utf-8'))
    if diagnostic['source_manifest_sha256']!=sha256_file(root/'run_manifest.json'):
        raise ValueError('diagnostic/source mismatch')
    geometry=json.loads((diagnostic_root/'geometry.json').read_text(encoding='utf-8'))
    if sha256_file(diagnostic_root/'geometry.json')!=diagnostic['artifact_sha256']['geometry.json']:
        raise ValueError('geometry artifact mismatch')
    by_age=defaultdict(list); by_state=defaultdict(list); strata=defaultdict(Counter)
    for point in points:
        by_age[(point['reference_id'],point['cache_age_s'])].append(point)
        by_state[(point['reference_id'],exact_drift_key(point))].append(point)
        group=strata[point['benchmark_stratum']]
        group['points']+=1; group['valid']+=point['valid']
        for cause,value in point['violations'].items(): group[cause]+=int(value)
    counts=Counter(); examples=[]
    normals=np.asarray(NORMAL_CODEBOOK_V1,float)
    for point in points:
        outside=bool(np.max(normals@np.asarray(point['normalized_drift'],float))>65535/32768)
        counts['outside_maximum_wire_surface']+=outside
        counts['valid_outside_maximum_wire_surface']+=outside and point['valid']
    for key,rows in sorted(by_age.items()):
        if len(rows)<2: continue
        counts['same_reference_and_age_groups']+=1
        distinct=len({exact_drift_key(p) for p in rows})>1
        mixed=len({p['valid'] for p in rows})>1
        counts['groups_with_distinct_receiver_states']+=distinct
        counts['groups_with_mixed_validity']+=mixed
        if distinct and mixed and len(examples)<3:
            examples.append({'reference_id':key[0],'age_s':key[1],
                'points':[{k:p[k] for k in ('point_id','receiver_drift_variant','valid','normalized_drift',
                    'cost_regret','cached_risk','trajectory_deviation_m','violations')} for p in rows]})
    for rows in by_state.values():
        if len({p['valid'] for p in rows})>1:
            counts['identical_float32_drift_conflict_groups']+=1
            counts['rows_in_identical_drift_conflicts']+=len(rows)
    point_map={p['point_id']:p for p in points}
    if len(point_map)!=len(points): raise ValueError('duplicate point identities')
    fitted=[]
    for row in geometry['references']:
        selected=[point_map[key] for key in row['point_ids']]
        grouped=defaultdict(list)
        for p in selected: grouped[exact_drift_key(p)].append(bool(p['valid']))
        flexible=sum(len(values) for values in grouped.values() if all(values))
        fitted.append({'reference_id':row['reference_id'],'points':len(selected),
            'valid':sum(p['valid'] for p in selected),'label_lookup_accepted_valid':flexible,
            'surface_accepted_valid':sum(row['surface']['accepted']),
            'ttl_accepted_valid':sum(row['ttl']['accepted'])})
    report={'schema_version':2,'status':'AUDIT_COMPLETE','scope':'development_only',
        'scientific_success':False,'manuscript_allowed':False,'original_test_accessed':False,
        'receiver_comparison_counts':dict(counts),'descriptive_examples':examples,
        'stratum_label_counts':{key:dict(value) for key,value in strata.items()},
        'geometry_totals':{key:sum(row[key] for row in fitted) for key in (
            'points','valid','label_lookup_accepted_valid','surface_accepted_valid','ttl_accepted_valid')},
        'geometry_per_reference':fitted,
        'information_validation_auroc':{k:v['validation']['auroc'] for k,v in diagnostic['information'].items()},
        'source_manifest_sha256':sha256_file(root/'run_manifest.json'),
        'diagnostics_sha256':sha256_file(diagnostic_root/'diagnostics.json'),
        'git':state,'completed_utc':utc_now(),
        'limitations':['exact-state grouping is within a frozen reference; other contexts are not equated',
            'label lookup uses evaluated labels and is not a deployable predictor or generalization estimate',
            'sampled label ambiguity may reflect unknown future world evolution',
            'geometry is sampled across controlled continuations, not a uniform safety region',
            'open-loop decisions do not establish closed-loop driving safety']}
    report['report_sha256']=canonical_json_hash(report)
    write_json_atomic(output,report)
    print(json.dumps({k:report[k] for k in ('status','receiver_comparison_counts','geometry_totals','information_validation_auroc')},indent=2))
    return 0


if __name__=='__main__': raise SystemExit(main())
