#!/usr/bin/env python3
"""Check economical one-slot generation against preserved full development traces."""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from bces.simulation.controlled_slots import generate_slot,decode_slot
from bces.utils.hashing import sha256_file,canonical_json_hash
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic


def main():
    state=git_state(ROOT)
    if state['dirty']: raise RuntimeError('clean source required')
    output=ROOT/'outputs/study_b/controlled_slot_contract_v2.json'
    if output.exists(): raise FileExistsError('completed audit is immutable')
    config_path=ROOT/'configs/evaluation/study_b_controlled_v2.yaml'
    config=yaml.safe_load(config_path.read_text())
    values=yaml.safe_load((ROOT/config['validity']).read_text())
    thresholds={k:values[k] for k in ('max_cost_regret','max_cached_risk','max_trajectory_deviation_m')}
    data_root=ROOT/config['output_root']
    manifest=json.loads((data_root/'run_manifest.json').read_text())
    seen=set(); cases=[]
    slots=[0,62,74,152,250,314]
    for name,digest in sorted(manifest['artifact_sha256'].items()):
        if not name.startswith('train_'): continue
        path=data_root/name
        if sha256_file(path)!=digest: raise ValueError('source artifact mismatch')
        with gzip.open(path,'rt') as f: scene=json.load(f)
        if scene['family'] in seen: continue
        slot=slots[len(seen)];seen.add(scene['family'])
        behavior,variant,age=decode_slot(slot,config)
        expected=[p for p in scene['points'] if p['behavior']==behavior
                  and p['receiver_drift_variant']==variant and p['cache_age_s']==age]
        start=time.perf_counter()
        actual=generate_slot(int(scene['scenario_id']),slot,config,thresholds,'train')
        elapsed=time.perf_counter()-start
        if expected:
            if len(expected)!=1 or actual['status']!='evaluated': raise AssertionError('slot availability mismatch')
            original=expected[0]
            for key,value in actual['point'].items():
                if key=='point_id':
                    if original[key]!=value+f':drift{variant}': raise AssertionError('point identity mismatch')
                elif canonical_json_hash(original[key])!=canonical_json_hash(value):
                    raise AssertionError(f'slot value mismatch: {key}')
            reference=next(r for r in scene['references'] if r['reference_id']==original['reference_id'])
            if canonical_json_hash(reference)!=canonical_json_hash(actual['reference']):
                raise AssertionError('frozen reference feature mismatch')
        elif actual['status'] not in ('abstain','oracle_unavailable'):
            raise AssertionError('infeasible slot was silently replaced')
        if actual['status']=='oracle_unavailable':
            reference=next(r for r in scene['references'] if r['reference_id']==actual['reference']['reference_id'])
            if canonical_json_hash(reference)!=canonical_json_hash(actual['reference']):
                raise AssertionError('unlabelled slot changed its observable reference')
        cases.append({'scenario_id':scene['scenario_id'],'family':scene['family'],'slot':slot,
            'status':actual['status'],'exact_comparison_passed':True,'wall_seconds':elapsed,
            'generated_record_sha256':canonical_json_hash(actual),'source_artifact_sha256':digest})
        print(json.dumps(cases[-1]),flush=True)
        if len(seen)==6: break
    if len(seen)!=6: raise RuntimeError('missing scenario family')
    report={'status':'PASS','scope':'engineering_development_only','scientific_success':False,
        'cases':cases,'no_fresh_confirmation_accessed':True,'original_test_accessed':False,
        'config_sha256':sha256_file(config_path),'source_manifest_sha256':sha256_file(data_root/'run_manifest.json'),
        'slot_generator_sha256':sha256_file(ROOT/'bces/simulation/controlled_slots.py'),
        'git':state,'completed_utc':utc_now()}
    write_json_atomic(output,report)
    return 0


if __name__=='__main__': raise SystemExit(main())
