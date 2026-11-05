"""Independent count/hash/timestamp checks, only on completed cohorts."""
from pathlib import Path
import argparse
from collections import Counter,defaultdict
import gzip
import json
import sys

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'experiments'))
from development_study import OUT,sha
from confirmation_study import REG,verify_registration,slot_summary
from write_safely import save_json

def verify_slots():
    reg=json.loads(REG.read_text());verify_registration(reg)
    folder=OUT/'fresh_slots';manifest=json.loads((folder/'collection_manifest.json').read_text())
    assert manifest['status']=='COMPLETE'
    expected={f'{reg["slot_seed_start"]+i}.json.gz' for i in range(reg['slot_scenarios'])}
    assert set(manifest['artifacts'])==expected
    rows=[]
    for name in sorted(expected):
        assert sha(folder/name)==manifest['artifacts'][name]
        with gzip.open(folder/name,'rt',encoding='utf-8') as f:stored=json.load(f)
        row=stored['row'];assert row['registration_sha256']==sha(REG)
        assert set(row['methods'])==set(reg['slot_methods'])
        rows.append(row)
    assert slot_summary(rows,reg)==json.loads((folder/'report.json').read_text())
    result={'cohort':'slots','integrity':'PASS','scenarios':len(rows),
        'unique_scenarios':len({r['scenario_id'] for r in rows}),
        'status_counts':dict(Counter(r['status'] for r in rows)),
        'registration_sha256':sha(REG),'all_registered_sources_unchanged':True}
    save_json(BASE/'operations'/'slot_integrity.json',result)
    print(json.dumps(result),flush=True)

def verify_driving():
    import numpy as np
    reg=json.loads(REG.read_text());verify_registration(reg)
    folder=OUT/'driving_confirmation';manifest=json.loads((folder/'collection_manifest.json').read_text())
    assert manifest['status']=='COMPLETE'
    expected={f'{s["seed"]}_{c}_{m}.json.gz' for s in reg['driving_assignments']
        for c in reg['driving_conditions'] for m in reg['driving_methods']}
    assert set(manifest['artifacts'])==expected
    groups=defaultdict(list);prefixes=defaultdict(set);audits=Counter()
    for name in sorted(expected):
        assert sha(folder/name)==manifest['artifacts'][name]
        with gzip.open(folder/name,'rt',encoding='utf-8') as f:r=json.load(f)
        assert r['registration_sha256']==sha(REG)
        assert r['traffic']['byte_conservation_ok'] and r['traffic']['component_conservation_ok']
        assert all(a['used_timestamp_ms']<=a['available_ms'] for a in r['input_availability_audit'])
        prefixes[r['scenario_id']].add(json.dumps(r['initial_contract'],sort_keys=True))
        audits['branches']+=1;audits['actuation_failures']+=not r['actuation_contract_passed']
        audits['censored_planning_decisions']+=r['planning_outcome_censored_decisions']
        audits['input_availability_checks']+=len(r['input_availability_audit'])
        groups[r['method']].append(r)
    assert all(len(values)==1 for values in prefixes.values()), 'unmatched branch prefixes'
    summaries={}
    for method,rows in groups.items():
        def quant(values):return np.quantile(values,[.5,.95,.99,1]).tolist() if values else None
        lat=[v for r in rows for v in r['traffic'].get('request_payload_latency_ms',[])]
        age=[v for r in rows for v in r['traffic'].get('reference_age_on_install_ms',[])]
        ttc=[v['events']['longitudinal_corridor_ttc_s'] for r in rows for v in r['rows']
            if v.get('events',{}).get('longitudinal_corridor_ttc_s') is not None]
        byscene=defaultdict(list)
        for r in rows:byscene[r['scenario_id']].append(r)
        positive=[]
        for branches in byscene.values():
            channel_means=[]
            for branch in branches:
                values=[max(0.,p['cost_regret']) for p in branch['planning_outcomes']]
                if values:channel_means.append(float(np.mean(values)))
            if len(channel_means)==2:positive.append(float(np.mean(channel_means)))
        risk_max=[max(p['cached_risk'] for p in r['planning_outcomes']) for r in rows if r['planning_outcomes']]
        dev_max=[max(p['planned_path_deviation_m'] for p in r['planning_outcomes']) for r in rows if r['planning_outcomes']]
        summaries[method]={
            'request_payload_latency_ms_p50_p95_p99_max':quant(lat),'completed_roundtrips':len(lat),
            'reference_age_on_install_ms_p50_p95_p99_max':quant(age),'installed_references':len(age),
            'branches_without_completed_roundtrip':sum(not r['traffic'].get('request_payload_latency_ms') for r in rows),
            'mean_scenario_route_distance_m':float(np.mean([np.mean([b['route_distance_lower_bound_m'] for b in v]) for v in byscene.values()])),
            'duration_s_min_p50_p95_max':np.quantile([r['duration_s'] for r in rows],[0,.5,.95,1]).tolist(),
            'per_decision_longitudinal_ttc_s_p50_p95_p99_max':quant(ttc),
            'min_longitudinal_ttc_s':min(ttc) if ttc else None,
            'mean_scenario_positive_regret':float(np.mean(positive)) if positive else None,
            'scenarios_with_two_channel_planning_followup':len(positive),
            'branches_with_planning_followup':len(risk_max),
            'mean_per_branch_max_risk':float(np.mean(risk_max)) if risk_max else None,
            'mean_per_branch_max_deviation_m':float(np.mean(dev_max)) if dev_max else None,
            'actuation_failures':sum(not r['actuation_contract_passed'] for r in rows),
            'component_bytes':dict(sum((Counter(r['traffic']['component_bytes']) for r in rows),Counter())),
            'completed_roundtrip_quantiles_are_conditional':True,
            'refresh_causes':dict(sum((Counter(r['traffic'].get('refresh_causes',{})) for r in rows),Counter()))}
    result={'cohort':'driving','integrity':'PASS','registration_sha256':sha(REG),
        'matched_prefix_scenarios':len(prefixes),'counts':dict(audits),'all_registered_sources_unchanged':True,
        'latency_quantiles_use_completed_request_response_samples':True,'summaries':summaries}
    save_json(BASE/'operations'/'driving_integrity_and_timing.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['slots','driving']);args=p.parse_args()
    verify_slots() if args.cohort=='slots' else verify_driving()
