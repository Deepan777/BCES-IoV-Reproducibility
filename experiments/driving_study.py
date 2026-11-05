"""Longer-episode development and separately frozen independent evaluation."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from scipy.stats import beta
from development_study import ORIGINAL,OUT,save_json,log,sha
from write_safely import save_json,save_gzip
from local_policies import LocalRule
from extended_closed_loop import run_branch
from bces.simulation.controlled_slots import development_generator
from bces.network.events import NetworkCondition

FAMILIES=('straight_lead_braking','on_ramp_merge','unprotected_crossing','pedestrian_crossing')
CONDITIONS={'nominal':NetworkCondition(50,0,10),'impaired':NetworkCondition(100,.1,1,20,.05,.05)}
METHODS=('frozen_surface','frozen_scalar_ttl','flexible_mlp','flexible_tree','cam_inspired','cam_fixed','cpm_inspired_push','normalized_drift','age_only','observable_cost_trigger','periodic_payload','local_only')
ROOT=Path(__file__).resolve().parent

def assignments(start,n,config):
    selected=[];count=Counter();generator=development_generator()
    for seed in range(start,start+100000):
        spec,_=generator.sample_scenario(seed,config)
        if spec.family in FAMILIES and count[spec.family]<n:
            selected.append({'seed':seed,'family':spec.family});count[spec.family]+=1
        if len(selected)==n*len(FAMILIES):return selected
    raise RuntimeError('quota unavailable; preserve failure rather than replacing outcomes')

def summarize(records):
    groups=defaultdict(list)
    for row in records:groups[row['method']].append(row)
    periodic_name='periodic_payload_unbudgeted' if 'periodic_payload_unbudgeted' in groups else 'periodic_payload'
    reference_branches={r['scenario_id']+'|'+r['condition']:r for r in groups.get(periodic_name,[])}
    result={}
    for method,rows in groups.items():
        risks=[p['cached_risk'] for r in rows for p in r['planning_outcomes']]
        regret=[p['cost_regret'] for r in rows for p in r['planning_outcomes']]
        deviation=[p['planned_path_deviation_m'] for r in rows for p in r['planning_outcomes']]
        byscenario=defaultdict(list)
        for r in rows:byscenario[r['scenario_id']].append(r)
        events=[];bytes_=[];durations=[];essential=[]
        for key,branches in byscenario.items():
            adverse=any(r['counts'].get('geometric_overlap_ticks',0) or r['counts'].get('sumo_ego_collision_ticks',0) or not r['actuation_contract_passed']
                or (reference_branches.get(r['scenario_id']+'|'+r['condition'],r)['route_distance_lower_bound_m']-r['route_distance_lower_bound_m']>5) for r in branches)
            events.append(adverse);bytes_.append(float(np.mean([r['traffic']['generated_bytes'] for r in branches])))
            durations.append(float(np.mean([r['duration_s'] for r in branches])))
            essential.append(any(r['reference_diagnostic'].get('remote_information_essential_at_reference',False) for r in branches))
        n=len(events);k=sum(events)
        result[method]={'scenario_clusters':n,'branch_executions':len(rows),'adverse_scenarios':k,
            'adverse_rate_upper_97_5':float(beta.ppf(.975,k+1,n-k)) if k<n else 1.,
            'mean_generated_bytes':float(np.mean(bytes_)),'mean_duration_s':float(np.mean(durations)),
            'remote_essential_reference_scenarios':int(sum(essential)),
            'planning_outcome_decisions':len(risks),
            'mean_risk':float(np.mean(risks)) if risks else None,
            'mean_regret':float(np.mean(regret)) if regret else None,
            'mean_deviation_m':float(np.mean(deviation)) if deviation else None,
            'risk_quantiles':np.quantile(risks,[.5,.95,1]).tolist() if risks else None,
            'regret_quantiles':np.quantile(regret,[.5,.95,1]).tolist() if regret else None,
            'deviation_quantiles_m':np.quantile(deviation,[.5,.95,1]).tolist() if deviation else None,
            'outcome_counts':dict(Counter(r['outcome'] for r in rows)),
            'cause_counts':dict(sum((Counter(r['traffic'].get('refresh_causes',{})) for r in rows),Counter()))}
    # Within-scenario two-channel averages are the inferential observations.
    base=reference_branches
    for method,rows in groups.items():
        ratios=defaultdict(list)
        for r in rows:
            b=base.get(r['scenario_id']+'|'+r['condition'])
            if b and b['traffic']['generated_bytes']:
                ratios[r['scenario_id']].append(1-r['traffic']['generated_bytes']/b['traffic']['generated_bytes'])
        if ratios:
            values=np.array([np.mean(v) for v in ratios.values()])
            draws=np.random.default_rng(301026).integers(len(values),size=(2000,len(values)))
            result[method]['mean_byte_reduction']=float(values.mean())
            result[method]['byte_reduction_cluster_bootstrap_95_descriptive']=np.quantile(values[draws].mean(1),[.025,.975]).tolist()
    return result

def paired_analysis(records,registration):
    grouped=defaultdict(dict)
    for row in records:grouped[row['method']].setdefault(row['scenario_id'],[]).append(row)
    reference='frozen_surface';contrasts=registration['primary_slot_contrasts']['comparators']
    result={}
    for comparator in contrasts:
        differences=[];families=[];missing=[]
        shared=sorted(set(grouped[reference])&set(grouped[comparator]))
        for scene in shared:
            pair=[]
            for method in (reference,comparator):
                channel_means=[]
                for branch in grouped[method][scene]:
                    values=[max(0.,p['cost_regret']) for p in branch['planning_outcomes']]
                    if not values:break
                    channel_means.append(float(np.mean(values)))
                pair.append(float(np.mean(channel_means)) if len(channel_means)==2 else None)
            if any(value is None for value in pair):missing.append(scene);continue
            differences.append(pair[0]-pair[1]);families.append(grouped[reference][scene][0]['family'])
        differences=np.array(differences);families=np.array(families)
        rng=np.random.default_rng(301026);drawn=[]
        for family in FAMILIES:
            values=differences[families==family]
            if len(values):
                ids=rng.integers(len(values),size=(20000,len(values)))
                drawn.append(values[ids].mean(1))
        if len(drawn)==len(FAMILIES):
            bootstrap=np.mean(drawn,axis=0)
            interval=np.quantile(bootstrap,[.05/(2*len(contrasts)),1-.05/(2*len(contrasts))]).tolist()
            mean=float(np.mean([differences[families==f].mean() for f in FAMILIES]))
        else:interval=None;mean=None
        result[comparator]={'paired_scenarios':len(differences),'missing_followup_scenarios':missing,
            'BCES_minus_comparator_mean_positive_planning_regret':mean,
            'family_stratified_bootstrap_family95_interval':interval,
            'interpretation':'approximate interval for a planning-score contrast; not an interactive causal collision effect',
            'superiority_claim_blocked_by_missing_followup':bool(missing)}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','development','confirmation'],default='smoke');p.add_argument('--n-per-family',type=int,default=1);p.add_argument('--methods',nargs='+');args=p.parse_args()
    torch.set_num_threads(2)
    config=yaml.safe_load((ORIGINAL/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    methods=tuple(args.methods or (('frozen_surface','frozen_scalar_ttl','periodic_payload','local_only','cpm_inspired_push') if args.phase=='smoke' else METHODS))
    folder=OUT/('driving_'+args.phase);folder.mkdir(parents=True,exist_ok=True)
    if args.phase=='confirmation':
        registration=json.loads((ROOT/'confirmation_registration.json').read_text())
        from confirmation_study import verify_registration
        verify_registration(registration)
        assert registration['status']=='FROZEN_BEFORE_NEW_OUTCOMES'
        assert sha(__file__)==registration['source_sha256']['driving_study.py']
        selected=registration['driving_assignments'];methods=tuple(registration['driving_methods'])
    else:selected=assignments(1600000000,args.n_per_family,config)
    models={}
    for name in methods:
        base_name=name.removesuffix('_unbudgeted')
        if base_name not in ('periodic_payload','local_only','cpm_inspired_push'):
            budgeted=args.phase=='confirmation' and not name.endswith('_unbudgeted')
            cap=registration['budget_rules'][name]['common_max_reuse_age_s'] if budgeted else None
            cooldown=registration['budget_rules'][name]['parameter'] if budgeted else 0
            models[name]=LocalRule(base_name,max_age_s=cap,min_refresh_ms=cooldown)
    records=[];start=time.perf_counter();total=len(selected)*len(methods)*len(CONDITIONS)
    for selected_row in selected:
        for condition_name,condition in CONDITIONS.items():
            for method in methods:
                path=folder/f'{selected_row["seed"]}_{condition_name}_{method}.json.gz'
                if path.exists():
                    with gzip.open(path,'rt',encoding='utf-8') as f:row=json.load(f)
                    if args.phase=='confirmation':assert row['registration_sha256']==sha(ROOT/'confirmation_registration.json')
                else:
                    activeconfig={**config}
                    base_name=method.removesuffix('_unbudgeted')
                    if args.phase=='confirmation' and method in ('periodic_payload','cpm_inspired_push'):
                        choice=registration['budget_rules'][method];activeconfig[choice['kind']]=choice['parameter']
                    row=run_branch(seed=selected_row['seed'],method=base_name,config=activeconfig,condition=condition,
                        network_seed=selected_row['seed']+int(condition_name=='impaired'),model=models.get(method),horizon_s=30.)
                    row['method']=method
                    row['condition']=condition_name;row['revision_scope']='receiver_local_policies_30s_maximum'
                    if args.phase=='confirmation':row['registration_sha256']=sha(ROOT/'confirmation_registration.json')
                    assert row['traffic']['byte_conservation_ok'] and row['traffic']['component_conservation_ok']
                    assert all(r['used_timestamp_ms']<=r['available_ms'] for r in row['input_availability_audit'])
                    save_gzip(path,row)
                records.append(row)
                save_json(folder/'progress.json',{'completed':len(records),'total':total,'phase':args.phase,'elapsed_s':time.perf_counter()-start,'interim_outcomes_inspected':False})
                log(stage='driving',phase=args.phase,completed=len(records),total=total,seconds=time.perf_counter()-start)
    save_json(folder/'collection_manifest.json',{'status':'COMPLETE','executions':len(records),'artifacts':{p.name:sha(p) for p in folder.glob('*.json.gz')}})
    result={'phase':args.phase,'horizon_s':30,'scenario_clusters':len(selected),'summaries':summarize(records),
        'source_sha256':{p.name:sha(p) for p in (Path(__file__),ROOT/'extended_closed_loop.py',ROOT/'local_policies.py')},
        'termination_handling':'early route completion and failure retained; six-second comparisons remain historical',
        'new_architecture':'fixed predictor at receiver; causal refresh request and payload response; no source model query',
        'road_safety_certification':False}
    if args.phase=='confirmation':
        result['paired_primary_planning_score']=paired_analysis(records,registration)
        result['family_summaries']={family:summarize([r for r in records if r['family']==family]) for family in FAMILIES}
        result['actual_budget_comparison']={m:{'mean_generated_bytes':s['mean_generated_bytes'],
            'relative_error_to_development_target':abs(s['mean_generated_bytes']-registration['budget_rules'][m]['target_mean_generated_bytes'])/registration['budget_rules'][m]['target_mean_generated_bytes']}
            for m,s in result['summaries'].items() if m in registration['budget_rules']}
    save_json(folder/'report.json',result);log(stage='driving_complete',phase=args.phase,seconds=time.perf_counter()-start)

if __name__=='__main__':main()
