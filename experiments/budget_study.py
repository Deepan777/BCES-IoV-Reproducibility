"""Development-only byte-budget matching; selection reads bytes, not risk labels."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from development_study import ORIGINAL,OUT,save_json,log,sha
from write_safely import save_json,save_gzip
from local_policies import LocalRule
from extended_closed_loop import run_branch
from driving_study import assignments,CONDITIONS

METHODS=('frozen_surface','frozen_scalar_ttl','flexible_mlp','flexible_tree','cam_inspired','cam_fixed','normalized_drift','age_only','observable_cost_trigger')

def main():
    torch.set_num_threads(2)
    config=yaml.safe_load((ORIGINAL/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    selected=assignments(1600000000,1,config)
    folder=OUT/'budget_development_v3';folder.mkdir(parents=True,exist_ok=True)
    previous_contract=json.loads((OUT/'budget_development_v2/contract.json').read_text())
    for source in ('extended_closed_loop.py','local_policies.py'):
        assert previous_contract['source_sha256'][source]==sha(Path(__file__).with_name(source))
    contract={'phase':'development_only','source_sha256':{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('extended_closed_loop.py'),Path(__file__).with_name('local_policies.py'))},
        'assignments':selected,'target':'15_percent_of_development_periodic_mean_generated_bytes','tolerance_relative':.1,
        'common_max_reuse_age_s':1.2,'minimum_refresh_intervals_ms':[0,400,800,1200,1600,2000,2400,2800],
        'source_generation_intervals_ms':[400,800,1200,1400,1600,2000,2400,2800],'periodic_intervals_ms':[200,400,800,1200,1400,1600,2000,2400],
        'reused_v2_report_sha256':sha(OUT/'budget_development_v2/report.json'),
        'selection':'minimum_absolute_byte_difference_then_lower_mean_bytes; no outcome labels in selection'}
    contract_path=folder/'contract.json'
    if contract_path.exists():assert json.loads(contract_path.read_text())==contract
    else:save_json(contract_path,contract)
    variants=[{'method':m,'parameter':interval,'kind':'min_refresh_ms'} for m in METHODS for interval in contract['minimum_refresh_intervals_ms']]
    variants += [{'method':'cpm_inspired_push','parameter':s,'kind':'source_generation_min_interval_ms'} for s in contract['source_generation_intervals_ms']]
    variants += [{'method':'periodic_payload','parameter':s,'kind':'periodic_interval_ms'} for s in contract['periodic_intervals_ms']]
    summaries=[];start=time.perf_counter();done=0
    for variant in variants:
        policy=LocalRule(variant['method'],max_age_s=1.2,min_refresh_ms=variant['parameter']) if variant['kind']=='min_refresh_ms' else None
        activeconfig={**config}
        if policy is None:activeconfig[variant['kind']]=variant['parameter']
        measurements=[]
        for case in selected:
            path=folder/f'{case["seed"]}_{variant["method"]}_{variant["parameter"]}.json.gz'
            historical=OUT/'budget_development_v2'/path.name
            if historical.exists():path=historical
            if path.exists():
                with gzip.open(path,'rt',encoding='utf-8') as f:row=json.load(f)
            else:
                row=run_branch(seed=case['seed'],method=variant['method'],config=activeconfig,condition=CONDITIONS['nominal'],network_seed=case['seed'],model=policy,horizon_s=30.)
                assert row['traffic']['byte_conservation_ok']
                save_gzip(path,row)
            measurements.append(row['traffic']['generated_bytes']);done+=1
            if done%10==0:log(stage='budget_development',completed=done,total=len(variants)*len(selected),seconds=time.perf_counter()-start)
        summaries.append({**variant,'mean_generated_bytes':float(np.mean(measurements)),'scenario_bytes':measurements})
    periodic=next(r['mean_generated_bytes'] for r in summaries if r['method']=='periodic_payload' and r['parameter']==200)
    target=.15*periodic
    choices={}
    for method in (*METHODS,'cpm_inspired_push','periodic_payload'):
        rows=[r for r in summaries if r['method']==method]
        chosen=min(rows,key=lambda r:(abs(r['mean_generated_bytes']-target),r['mean_generated_bytes']))
        error=abs(chosen['mean_generated_bytes']-target)/target
        choices[method]={**chosen,'common_max_reuse_age_s':1.2,'target_mean_generated_bytes':target,'relative_budget_error':error,'development_budget_match_within_10_percent':error<=.1}
    save_json(folder/'report.json',{'contract':contract,'candidates':summaries,'selected':choices,
        'all_methods_within_tolerance':all(c['development_budget_match_within_10_percent'] for c in choices.values()),
        'selection_uses_outcomes':False,'new_confirmation_required':True})
    log(stage='budget_complete',seconds=time.perf_counter()-start,target=target)

if __name__=='__main__':main()
