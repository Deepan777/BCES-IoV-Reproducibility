#!/usr/bin/env python3
"""Preserve paired identity-repair effects and actual Study-B evidence gates."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from bces.evaluation.study_b import load_development
from bces.utils.hashing import canonical_json_hash,sha256_file
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic


def main():
    old_root=ROOT/'outputs/study_b/causal_development_v1'
    new_root=ROOT/'outputs/study_b/causal_associated_v3'
    diagnostic_path=ROOT/'outputs/study_b/diagnostics_associated_v3/diagnostics.json'
    config_path=ROOT/'configs/training/decision_surface_associated_v3.yaml'
    output=ROOT/'outputs/study_b/association_evidence_audit_v3.json'
    if output.exists():
        raise FileExistsError('evidence audit is immutable')
    state=git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('clean committed audit source required')
    _,old,old_manifest=load_development(old_root)
    _,new,new_manifest=load_development(new_root)
    diagnostics=json.loads(diagnostic_path.read_text(encoding='utf-8'))
    if diagnostics['source_manifest_sha256'] != sha256_file(new_root/'run_manifest.json'):
        raise ValueError('diagnostics do not refer to this corrected source')
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    old={r['point_id']:r for r in old}
    new={r['point_id']:r for r in new}
    if set(old)!=set(new):
        raise ValueError('paired sensitivity requires exactly the same evaluated points')
    counts=Counter()
    by_split={name:Counter() for name in ('train','validation','calibration')}
    for key,row in new.items():
        previous=old[key]
        if row['normalized_drift']!=previous['normalized_drift'] or row['split']!=previous['split']:
            raise ValueError('identity-only comparison changed drift or partition')
        for group in (counts,by_split[row['split']]):
            group['points']+=1
            group['old_valid']+=previous['valid']
            group['new_valid']+=row['valid']
            group['labels_changed']+=previous['valid']!=row['valid']
            group[f"valid_transition:{int(previous['valid'])}->{int(row['valid'])}"]+=1
            if row['cache_age_s']==0:
                group['origins']+=1
                group['old_invalid_origins']+=not previous['valid']
                group['new_invalid_origins']+=not row['valid']
    information={key:diagnostics['information'][key]['validation']['auroc']
                 for key in ('frozen_inputs','reference_margins')}
    geometry=diagnostics['geometry']
    extra=geometry['surface_accepted_valid']-geometry['ttl_accepted_valid']
    ready=max(information.values())>=config['minimum_diagnostic_auroc'] and extra>=config['minimum_geometry_extra_valid_points']
    report={'schema_version':3,'status':'AUDIT_COMPLETE','scope':'development_only',
        'original_test_accessed':False,'scientific_success':False,'manuscript_allowed':False,
        'paired_counts':dict(counts),'paired_by_split':{k:dict(v) for k,v in by_split.items()},
        'role_and_association_counts':{k:v for k,v in new_manifest['counts'].items()
                                     if k.startswith(('role_audit:','association:'))},
        'information_validation_auroc':information,'geometry':geometry,
        'learning_prerequisite_met':ready,
        'registered_minimum_auroc':config['minimum_diagnostic_auroc'],
        'registered_minimum_geometry_extra_valid':config['minimum_geometry_extra_valid_points'],
        'publication_gate_failures':['unverified receiver/local-view assignment',
            'perception proxy is not complete physical truth','association uncertainty not independently validated',
            'no fresh confirmation','no demonstrated matched-risk communication benefit',
            'conditional regret envelopes not established for this planner/data'],
        'provenance':{'old_manifest_sha256':sha256_file(old_root/'run_manifest.json'),
                      'new_manifest_sha256':sha256_file(new_root/'run_manifest.json'),
                      'diagnostics_sha256':sha256_file(diagnostic_path),
                      'training_config_sha256':sha256_file(config_path)},
        'git':state,'completed_utc':utc_now()}
    report['report_sha256']=canonical_json_hash(report)
    write_json_atomic(output,report)
    print(json.dumps(report,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
