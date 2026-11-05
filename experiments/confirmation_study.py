"""Freeze revised policies, then collect fresh slots without interim outcomes."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from scipy.stats import beta,binomtest
from development_study import ORIGINAL,OUT,save_json,log,sha
from write_safely import save_json,save_gzip
from local_policies import LocalRule
from bces.simulation.controlled_slots import generate_slot
from bces.evaluation.slot_confirmation import random_slot,paired_difference_interval
from bces.simulation.controlled_decisions import kinematic,route_for
from bces.simulation.controlled_contract import ego_from_sample,local_observations,payload_objects
from bces.simulation.scenario_builder import scenario_spec
from bces.models.bound_policy import reference_query
from bces.oracle.causal_diagnostics import observable_planner_features
from bces.oracle.world import WorldObject
from bces.oracle.planner import PlannerConfig
from bces.simulation.controlled_decisions import ControlledDecisionPlanner
from bces.protocol.bound_exchange import decode_objects
from bces.protocol.bindings import sender_id_hash
from bces.geometry.drift import compute_drift

ROOT=Path(__file__).resolve().parent
REG=ROOT/'confirmation_registration.json'
SLOT_N=2400

def verify_registration(reg):
    assert sha(OUT/'development_report.json')==reg['development_report_sha256']
    assert sha(OUT/'budget_development_v3/report.json')==reg['budget_report_sha256']
    for name,expected in reg['source_sha256'].items():
        assert sha(ROOT/name)==expected,('changed registered source',name)
    for name,expected in reg['checkpoint_sha256'].items():
        assert sha(OUT/name)==expected,('changed registered checkpoint',name)
    for name,expected in reg['original_asset_sha256'].items():
        assert sha(ORIGINAL/name)==expected,('changed original asset',name)

def register():
    if REG.exists():raise FileExistsError('registration immutable; use the existing run')
    report=json.loads((OUT/'development_report.json').read_text())
    budget=json.loads((OUT/'budget_development_v3/report.json').read_text())
    from driving_study import assignments
    config=yaml.safe_load((ORIGINAL/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    sources={p.name:sha(p) for p in ROOT.glob('*.py')}
    files=[p for p in OUT.iterdir() if p.suffix in ('.pt','.joblib')]
    assets=['configs/evaluation/study_b_controlled_v2.yaml','configs/planner/kinematic.yaml','sumo/registered_grid_v1.net.xml',
        'bces/simulation/controlled_slots.py','bces/simulation/controlled_decisions.py','bces/oracle/planner.py','bces/oracle/vectorized_planner.py',
        'bces/simulation/controlled_contract.py','scripts/16_generate_controlled_development.py']
    historical=ORIGINAL/'outputs/study_b/slot_confirmation_v1/preregistration.json'
    assets.append(str(historical.relative_to(ORIGINAL)))
    assets.extend(rule['path'] for rule in json.loads(historical.read_text())['operating_rules'].values())
    assets.extend(str(p.relative_to(ORIGINAL)) for p in (ORIGINAL/'bces').rglob('*.py'))
    slot_methods=list(report['methods'])+['cam_fixed']
    # Power under a stated, useful paired alternative; this is a design
    # calculation, not a prediction of experimental success.
    rng=np.random.default_rng(301026)
    draws=rng.multinomial(SLOT_N,[.01,.025,.965],size=20000)
    alpha=.05/8
    power=float(np.mean([binomtest(int(a),int(a+b),.5,alternative='two-sided').pvalue<=alpha if a+b else False for a,b,_ in draws]))
    registration={'status':'FROZEN_BEFORE_NEW_OUTCOMES','registered_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sources,'checkpoint_sha256':{p.name:sha(p) for p in files},
        'original_asset_sha256':{n:sha(ORIGINAL/n) for n in assets},
        'development_report_sha256':sha(OUT/'development_report.json'),'budget_report_sha256':sha(OUT/'budget_development_v3/report.json'),
        'slot_scenarios':SLOT_N,'slot_seed_start':1700000000,'slot_phase_id':301026,'slot_randomization_seed':303026,
        'slot_methods':slot_methods,'driving_methods':list(budget['selected'])+['local_only','cam_fixed_unbudgeted','cpm_inspired_push_unbudgeted','periodic_payload_unbudgeted'],
        'driving_assignments':assignments(1800000000,30,config),'driving_horizon_s':30.,
        'driving_conditions':['nominal','impaired'],'budget_rules':budget['selected'],
        'slot_thresholds':{m:v['threshold'] for m,v in report['methods'].items()},
        'primary_slot_contrasts':{'reference':'frozen_surface','comparators':['frozen_scalar_ttl','flexible_mlp','flexible_tree','cam_inspired','cam_fixed','normalized_drift','age_only','observable_cost_trigger']},
        'primary_endpoint':'per-scenario accepted AND (invalid OR oracle-unavailable)',
        'conditional_invalid_reuse':'reported separately with family-adjusted exact upper bounds across all methods',
        'coverage_equivalence_band':.03,'multiple_comparisons':'two-sided exact paired discordance tests; Holm across eight contrasts',
        'power_design':{'BCES_only_event_probability':.01,'comparator_only_event_probability':.025,
            'minimally_useful_absolute_event_difference':.015,'worst_case_Bonferroni_alpha':alpha,'simulation_replicates':20000,'estimated_power':power},
        'stopping':'fixed N; no interim outcome inspection, policy tuning, replacement, family removal, or efficacy stopping',
        'budget_matching':'development nearest-byte choices; report actual confirmation mismatch, and call matched only within 10 percent of common target',
        'driving_inference':'scenario clusters; two channels nested; zero events do not establish equivalence or a 1-percent adverse bound at N=120',
        'driving_primary_metric':'per-scenario mean positive cost regret across available planning decisions, two channel means nested equally',
        'driving_primary_inference':'20,000 within-family whole-scenario bootstrap replicates; Bonferroni-adjusted intervals for BCES versus the eight prespecified comparator policies; approximate sampling intervals',
        'driving_secondary_metrics':['max cached-plan risk','max planned-path deviation','minimum longitudinal TTC','route progress','adverse episode including >5-m progress deficit versus unbudgeted periodic','actual generated bytes','request/payload latency'],
        'driving_budget_variants':'common 1.2-second age cap plus development-selected request cooldown; a rejected payload uses local fallback; unbudgeted CAM/CPM-inspired and 200-ms periodic controls are also retained',
        'driving_scope':'receiver-local variants; up to 30 seconds or route completion; four predefined map-compatible families',
        'new_seed_namespaces_disjoint':True,'historical_holdout_tuning':False}
    import re
    old_ids={int(value) for p in (ORIGINAL/'outputs/study_b').rglob('*') if p.is_file()
        for value in re.findall(r'(?<!\d)(\d{6,10})(?!\d)',p.name)}
    new_ids=set(range(registration['slot_seed_start'],registration['slot_seed_start']+SLOT_N))|{r['seed'] for r in registration['driving_assignments']}
    assert not old_ids&new_ids,'historical seed namespace overlap'
    registration['historical_filename_seed_candidates_checked']=len(old_ids)
    assert power>=.8
    save_json(REG,registration);log(stage='registered',slots=SLOT_N,driving_scenarios=len(registration['driving_assignments']),power=power)

def slot_summary(rows,reg):
    n=len(rows);methods=reg['slot_methods'];result={}
    for name in methods:
        a=sum(r['methods'][name]['accepted'] for r in rows);bad=sum(r['methods'][name]['adverse'] for r in rows)
        unknown=sum(r['methods'][name]['accepted'] and r['valid'] is None for r in rows)
        result[name]={'accepted':a,'coverage':a/n,'invalid_or_unknown_accepted':bad,'unknown_accepted':unknown,
            'observed_invalid_accepted':bad-unknown,'invalid_reuse':bad/a if a else None,
            'exact_upper_family95':float(beta.ppf(1-.05/len(methods),bad+1,a-bad)) if bad<a else 1.}
    paired={};primary=reg['primary_slot_contrasts']['reference']
    for other in reg['primary_slot_contrasts']['comparators']:
        first=sum(r['methods'][primary]['adverse'] and not r['methods'][other]['adverse'] for r in rows)
        second=sum(r['methods'][other]['adverse'] and not r['methods'][primary]['adverse'] for r in rows)
        p=float(binomtest(first,first+second,.5,alternative='two-sided').pvalue) if first+second else 1.
        a=sum(r['methods'][primary]['accepted'] and not r['methods'][other]['accepted'] for r in rows)
        b=sum(r['methods'][other]['accepted'] and not r['methods'][primary]['accepted'] for r in rows)
        ci=paired_difference_interval(a,b,n)
        paired[other]={'BCES_only_adverse':first,'comparator_only_adverse':second,'paired_event_difference':(first-second)/n,
            'paired_event_difference_interval':paired_difference_interval(first,second,n),'two_sided_exact_p':p,
            'coverage_difference_interval':ci,'coverage_equivalent_within_3pp':ci[0]>=-.03 and ci[1]<=.03}
    sorted_names=sorted(paired,key=lambda name:paired[name]['two_sided_exact_p']);previous=0
    for rank,name in enumerate(sorted_names):
        adjusted=min(1.,max(previous,(len(sorted_names)-rank)*paired[name]['two_sided_exact_p']));previous=adjusted
        paired[name]['holm_adjusted_p']=adjusted
    return {'phase':'fresh independent slots','scenario_clusters':n,'methods':result,'paired_comparisons':paired,
        'registration_sha256':sha(REG),'all_records_collected_before_analysis':True}

def slots():
    reg=json.loads(REG.read_text());verify_registration(reg)
    torch.set_num_threads(2);folder=OUT/'fresh_slots';folder.mkdir(parents=True,exist_ok=True)
    config=yaml.safe_load((ORIGINAL/'configs/evaluation/study_b_controlled_v2.yaml').read_text())
    models={m:LocalRule(m) for m in reg['slot_methods']}
    from bces.simulation.controlled_slots import development_generator
    from fast_sumo import start_controlled
    development_generator().start_controlled=start_controlled
    for name,model in models.items():
        if name!='cam_fixed':assert model.threshold==reg['slot_thresholds'][name]
    planner=ControlledDecisionPlanner(PlannerConfig.from_yaml(ORIGINAL/config['planner']))
    rows=[];start=time.perf_counter()
    for index in range(reg['slot_scenarios']):
        seed=reg['slot_seed_start']+index;slot=random_slot(seed,reg['slot_randomization_seed'],reg['slot_phase_id'])
        path=folder/f'{seed}.json.gz'
        if path.exists():
            with gzip.open(path,'rt',encoding='utf-8') as f:row=json.load(f)['row']
            assert row['registration_sha256']==sha(REG)
        else:
            raw=generate_slot(seed,slot,config,{'max_cost_regret':5.,'max_cached_risk':.4,'max_trajectory_deviation_m':2.},'revision_confirmation')
            accepted={m:False for m in models};scores={}
            if raw['status']!='abstain':
                message=decode_objects(bytes.fromhex(raw['payload_hex']))
                query=reference_query(raw['reference'],query_id=1,sender_hash=sender_id_hash(message.sender_id))
                ego=ego_from_sample(raw['current_receiver']);timestamp=query.reference_state.timestamp_ms+round(raw['age_s']*1000)
                current=kinematic(ego,timestamp);z=np.array(compute_drift(query.reference_state,current,query.drift_scales).normalized,np.float32)
                frame=raw['trace'][str(timestamp)];objects=tuple(WorldObject(**o) for o in frame['objects'])
                spec=scenario_spec(raw['family']);hidden=(('person:' if spec.pedestrian else 'vehicle:')+spec.actor_id,) if spec.hidden_from_local else ()
                local=local_observations(objects,ego,range_m=config['local_range_m'],hidden_ids=hidden)
                cached=tuple(o.propagated(raw['age_s']) for o in payload_objects(message))
                try:features=observable_planner_features(planner,local,cached,ego,route_for(ego,raw['behavior'],planner))
                except RuntimeError:features=None
                for name,model in models.items():
                    if name=='observable_cost_trigger' and features is None:continue
                    score=model.score(model.prepare(raw['reference']),z,features)
                    accepted[name]=bool(score>=model.threshold);scores[name]=float(score)
            # Label access follows all fixed online decisions.
            valid=bool(raw['point']['valid']) if raw['status']=='evaluated' else None
            row={'scenario_id':str(seed),'family':raw['family'],'slot':slot,'status':raw['status'],'valid':valid,
                'registration_sha256':sha(REG),'methods':{m:{'accepted':a,'adverse':a and valid is not True,'score':scores.get(m)} for m,a in accepted.items()}}
            save_gzip(path,{'row':row,'raw':raw})
            # Store raw data separately but expose only the row to the collector.
        if 'row' in row:row=row['row']
        rows.append(row)
        if (index+1)%25==0:
            progress={'completed':index+1,'total':reg['slot_scenarios'],'elapsed_s':time.perf_counter()-start,'interim_outcomes_inspected':False}
            save_json(folder/'progress.json',progress);log(stage='fresh_slot_collection',**progress)
    assert len(rows)==reg['slot_scenarios'] and len({r['scenario_id'] for r in rows})==len(rows)
    save_json(folder/'collection_manifest.json',{'status':'COMPLETE','scenarios':len(rows),'registration_sha256':sha(REG),
        'artifacts':{p.name:sha(p) for p in folder.glob('*.json.gz')}})
    save_json(folder/'report.json',slot_summary(rows,reg));log(stage='fresh_slot_analysis_complete',scenarios=len(rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['register','slots']);args=p.parse_args()
    register() if args.phase=='register' else slots()

if __name__=='__main__':main()
