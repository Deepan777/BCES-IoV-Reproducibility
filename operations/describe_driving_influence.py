"""Post-completion descriptive influence, not another confirmatory test."""
from pathlib import Path
from collections import defaultdict
import gzip
import json
import numpy as np
BASE=Path(__file__).resolve().parents[1]
folder=BASE/'experiments/results/driving_confirmation'
assert json.loads((folder/'collection_manifest.json').read_text())['status']=='COMPLETE'
reg=json.loads((BASE/'experiments/confirmation_registration.json').read_text())
rows=defaultdict(dict)
for p in folder.glob('*.json.gz'):
    with gzip.open(p,'rt',encoding='utf-8') as f:r=json.load(f)
    values=[max(0.,v['cost_regret']) for v in r['planning_outcomes']]
    rows[r['method']][(r['scenario_id'],r['condition'])]={
        'positive_regret':float(np.mean(values)) if values else None,
        'counts':r['counts'],'progress':r['route_distance_lower_bound_m'],
        'actuation':r['actuation_contract_passed'],'family':r['family']}
scenes=[str(a['seed']) for a in reg['driving_assignments']]
vectors={m:np.array([np.mean([values[(s,c)]['positive_regret'] for c in reg['driving_conditions']]) for s in scenes]) for m,values in rows.items()}
description={m:{'positive_regret_scenario_p50_p95_max':np.quantile(v,[.5,.95,1]).tolist(),
    'mean':float(v.mean()),'scenarios_with_nonzero_positive_regret':int((v>0).sum())} for m,v in vectors.items()}
influence={}
for m in reg['primary_slot_contrasts']['comparators']:
    difference=vectors['frozen_surface']-vectors[m];i=int(np.abs(difference).argmax())
    influence[m]={'largest_absolute_difference_scenario':scenes[i],
        'largest_case_difference':float(difference[i]),
        'largest_case_absolute_share_of_sum_absolute_differences':float(abs(difference[i])/np.abs(difference).sum()) if np.abs(difference).sum() else None,
        'mean_difference_all':float(difference.mean()),
        'mean_difference_without_largest_absolute_case_descriptive':float(np.delete(difference,i).mean()),
        'positive_zero_negative_case_counts':[int((difference>0).sum()),int((difference==0).sum()),int((difference<0).sum())]}
adverse={}
base=rows['periodic_payload_unbudgeted']
for m,values in rows.items():
    events=[]
    for s in scenes:
        causes=set()
        for c in reg['driving_conditions']:
            r=values[(s,c)]
            if r['counts'].get('geometric_overlap_ticks',0):causes.add('sampled geometric overlap')
            if r['counts'].get('sumo_ego_collision_ticks',0):causes.add('SUMO ego collision')
            if not r['actuation']:causes.add('actuation contract failure')
            if base[(s,c)]['progress']-r['progress']>5:causes.add('progress deficit >5 m')
        if causes:events.append({'scenario':s,'family':values[(s,reg['driving_conditions'][0])]['family'],'causes':sorted(causes)})
    adverse[m]=events
report={'scope':'post-completion descriptive influence; no new policy, exclusion or hypothesis test',
    'scenario_distributions':description,'paired_influence':influence,'adverse_causes':adverse}
(BASE/'operations/driving_influence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
