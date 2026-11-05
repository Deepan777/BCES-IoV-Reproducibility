"""Reviewer-requested development comparisons. Historical sources are read only.

No historical confirmation outcomes are loaded. All operating points are chosen
on validation; development calibration is reported without retuning.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
sys.dont_write_bytecode=True
ORIGINAL=Path(r'D:\VIT Vellore Research one\BCES-IoV')
sys.path.insert(0,str(ORIGINAL))
import joblib
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
from bces.evaluation.study_b import load_development,feature_arrays,origin_validity
from bces.models.decision_surface import DecisionSurfaceNet,decision_surface_loss,membership_score,CAP,GUARD,STEP
from bces.models.surfacenet import normal_tensor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SEEDS=(301026,301027,301028)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def log(**values):
    print(json.dumps(values),flush=True)

def data():
    refs,points,manifest=load_development(ORIGINAL/'outputs/study_b/controlled_development_v2')
    arrays=feature_arrays(refs,points)
    split=np.array([p['split'] for p in points])
    scenarios=np.array([p['scenario_id'] for p in points])
    sets={name:set(scenarios[split==name]) for name in ('train','validation','calibration')}
    assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','calibration'),('validation','calibration')])
    return refs,points,arrays,split,scenarios,manifest

def report_metrics(y,score,threshold,scenarios,*,replicates=1000):
    y=np.asarray(y,bool);score=np.asarray(score,float)
    accept=score>=threshold
    names,index=np.unique(scenarios,return_inverse=True)
    a=np.bincount(index,weights=accept,minlength=len(names))
    bad=np.bincount(index,weights=accept&~y,minlength=len(names))
    draws=np.random.default_rng(301026).integers(len(names),size=(replicates,len(names)))
    den=a[draws].sum(1);ratios=bad[draws].sum(1)/np.maximum(den,1)
    n=int(accept.sum());k=int((accept&~y).sum())
    return {'states':len(y),'scenario_clusters':len(names),'accepted':n,
        'invalid_accepted':k,'valid_excluded':int((~accept&y).sum()),
        'coverage':n/len(y),'invalid_reuse':k/n if n else None,
        'invalid_reuse_bootstrap_95_descriptive':np.quantile(ratios,[.025,.975]).tolist(),
        'auroc':float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
        'auprc':float(average_precision_score(y,score)) if y.any() else None}

def operating_threshold(score,coverage=.8):
    # Ties are not broken using outcomes. Accept all values strictly above
    # the largest excluded order statistic; attainable coverage can be smaller.
    order=np.sort(np.asarray(score,float))
    index=max(0,min(len(order)-1,int(np.ceil((1-coverage)*len(order)))-1))
    return float(np.nextafter(order[index],np.inf))

class FlexibleMLP(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,128),nn.SiLU(),nn.LayerNorm(128),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def infer(model,x,device,batch=2048):
    model.eval();scores=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            scores.append(model(torch.from_numpy(x[start:start+batch]).to(device)).cpu().numpy())
    return np.concatenate(scores)

def train_flexible(x,y,split,device):
    train=split=='train';val=split=='validation'
    mean=x[train].mean(0);std=x[train].std(0).clip(.01)
    x=np.clip((x-mean)/std,-10,10).astype(np.float32)
    runs=[];best=None
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed)
        model=FlexibleMLP(x.shape[1]).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
        tx=torch.from_numpy(x[train]).to(device);ty=torch.from_numpy(y[train]).to(device)
        best_loss=float('inf');stale=0;weights=None
        for epoch in range(1,101):
            model.train();order=torch.randperm(len(tx),device=device)
            for j in range(0,len(tx),512):
                ids=order[j:j+512];pred=model(tx[ids]);w=torch.where(ty[ids]>.5,1.,4.)
                loss=nn.functional.binary_cross_entropy_with_logits(pred,ty[ids],weight=w)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            pred=infer(model,x[val],device)
            vloss=float(nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(pred),torch.from_numpy(y[val]),weight=torch.where(torch.from_numpy(y[val])>.5,1.,4.)))
            if vloss<best_loss-1e-5:
                best_loss=vloss;weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else: stale+=1
            if stale>=15:break
        run={'seed':seed,'validation_loss':best_loss,'epochs':epoch}
        runs.append(run);log(stage='mlp',**run)
        if best is None or (best_loss,seed)<(best['validation_loss'],best['seed']):
            best={**run,'state':weights}
    model=FlexibleMLP(x.shape[1]);model.load_state_dict(best['state'])
    saved={'state':best['state'],'mean':mean,'std':std,'features':x.shape[1],'seed':best['seed'],'runs':runs}
    torch.save(saved,OUT/'flexible_mlp.pt')
    return infer(model,x,'cpu'),saved

def surface_predictions(model,xref,ref_index,z,originless=False):
    model.eval();device=next(model.parameters()).device
    with torch.inference_mode():
        offsets,logits=model(torch.from_numpy(xref).to(device))
    offsets=offsets.cpu();logits=logits.cpu()
    if originless:logits=torch.ones_like(logits)
    scores=membership_score(offsets[ref_index],logits[ref_index],torch.from_numpy(z),scalar=model.scalar).numpy()
    return scores,offsets.numpy(),logits.numpy()

def train_ablations(refs,points,a,split,device):
    ids=a['reference_ids'];idx=a['reference_index'];z=a['drift'].astype(np.float32)
    y=np.array([p['valid'] for p in points],np.float32)
    origin=origin_validity(ids,points)[idx]
    source=ORIGINAL/'outputs/study_b/decision_surface_controlled_v2'
    manifest=json.loads((source/'run_manifest.json').read_text())
    assert sha(source/'training_teachers.json')==manifest['artifacts']['training_teachers.json']
    teachers=json.loads((source/'training_teachers.json').read_text())
    idmap={key:i for i,key in enumerate(ids)}
    targets={False:np.zeros((len(ids),16),np.float32),True:np.zeros((len(ids),1),np.float32)}
    for row in teachers['references']:
        i=idmap[row['reference_id']]
        if not row['surface']['abstain']:targets[False][i]=np.minimum(CAP,np.array(row['surface']['offsets'])+GUARD)
        if not row['ttl']['abstain']:targets[True][i,0]=min(CAP,row['ttl']['threshold']+GUARD)
        assert refs[row['reference_id']]['split']=='train'
    tr=split=='train';va=split=='validation'
    reftrain=np.array([refs[k]['split']=='train' for k in ids])
    behavior_column=a['base_reference'].shape[1]-6
    # Context tail is [v,heading,a,curvature,seven scales,behavior,risk,
    # object count,occlusion,delay,map]. Explicit behavior lies six from end.
    assert all(a['base_reference'][i,behavior_column]==refs[k]['behavior_id'] for i,k in enumerate(ids))
    results={};prediction={}
    for behavior in (True,False):
        raw=a['margin_reference'].copy()
        if not behavior:raw[:,behavior_column]=0
        mean=raw[reftrain].mean(0);std=raw[reftrain].std(0).clip(.01)
        normalized=np.clip((raw-mean)/std,-10,10).astype(np.float32)
        for scalar in (False,True):
            for teacher in (True,False):
                name=f'{"ttl" if scalar else "surface"}_teacher{int(teacher)}_behavior{int(behavior)}'
                allruns=[];best=None
                for seed in SEEDS:
                    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                    model=DecisionSurfaceNet(raw.shape[1],scalar).to(device)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.001)
                    tx=torch.from_numpy(normalized[idx[tr]]).to(device);tz=torch.from_numpy(z[tr]).to(device)
                    ty=torch.from_numpy(y[tr]).to(device);to=torch.from_numpy(origin[tr]).to(device)
                    tt=torch.from_numpy(targets[scalar][idx[tr]]).to(device)
                    vx=torch.from_numpy(normalized[idx[va]]).to(device);vz=torch.from_numpy(z[va]).to(device)
                    vy=torch.from_numpy(y[va]).to(device);vo=torch.from_numpy(origin[va]).to(device)
                    best_loss=float('inf');weights=None;stale=0
                    for epoch in range(1,101):
                        model.train();order=torch.randperm(len(tx),device=device)
                        for j in range(0,len(tx),512):
                            ii=order[j:j+512];b,g=model(tx[ii]);loss=decision_surface_loss(b,g,tz[ii],ty[ii],to[ii],scalar=scalar,teacher=tt[ii] if teacher else None)
                            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        model.eval()
                        with torch.inference_mode():
                            # Batched validation avoids retaining the full graph.
                            b,g=model(vx);vloss=float(decision_surface_loss(b,g,vz,vy,vo,scalar=scalar))
                        if vloss<best_loss-1e-5:
                            best_loss=vloss;weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                        else:stale+=1
                        if stale>=15:break
                    run={'seed':seed,'epochs':epoch,'validation_loss':best_loss};allruns.append(run)
                    if best is None or (best_loss,seed)<(best['validation_loss'],best['seed']):best={**run,'state':weights}
                    log(stage='ablation',method=name,**run)
                    del model,optimizer,tx,tz,ty,to,tt,vx,vz,vy,vo
                    if device=='cuda':torch.cuda.empty_cache()
                model=DecisionSurfaceNet(raw.shape[1],scalar);model.load_state_dict(best['state'])
                score,offsets,logits=surface_predictions(model,normalized,idx,z)
                saved={'model_state':best['state'],'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                    'view':'reference_margins','scalar':scalar,'feature_count':raw.shape[1],'seed':best['seed'],
                    'behavior_column':behavior_column,'explicit_behavior':behavior,'teacher':teacher,'runs':allruns}
                torch.save(saved,OUT/(name+'.pt'))
                prediction[name]=score;results[name]={'runs':allruns,'selected_seed':best['seed']}
                if name=='surface_teacher1_behavior1':
                    np.savez_compressed(OUT/'surface_geometry_predictions.npz',offsets=offsets,logits=logits,z=z,ref_index=idx,y=y,split=split)
    return prediction,results

def heuristic_scores(a):
    z=a['drift'].astype(float)
    # Higher score always means more eligible reuse.
    return {'age_only':-z[:,6],
        'normalized_drift':-np.max(np.abs(z),axis=1),
        'cam_inspired':-np.maximum.reduce([np.hypot(z[:,0]*40,z[:,1]*5)/4,
            np.abs(z[:,2]*5)/.5,np.abs(z[:,3]*.7)/np.deg2rad(4),z[:,6]*4]),
        # Cost is estimated using local+cached information, no fresh oracle.
        'observable_cost_trigger':-np.maximum(np.abs(a['current_receiver'][:,-29+6]-a['margin_reference'][a['reference_index'],-29+6])/5,z[:,6])}

def geometry_audit(refs,points,a,split):
    source=ORIGINAL/'outputs/study_b/decision_surface_controlled_v2/training_teachers.json'
    fit=json.loads(source.read_text());rows=fit['references'];counts=defaultdict(int)
    byref=defaultdict(list)
    for i,p in enumerate(points):byref[p['reference_id']].append(i)
    normals=normal_tensor().numpy()
    for row in rows:
        pp=[points[i] for i in byref[row['reference_id']]]
        yy=np.array([p['valid'] for p in pp],bool)
        counts['training_references']+=1;counts['sampled_valid_states']+=int(yy.sum())
        counts['solver_proven_optimal']+=bool(row['surface'].get('optimal'))
        counts['abstaining_fits']+=bool(row['surface']['abstain'])
        accepted=np.asarray(row['surface']['accepted'],bool)
        counts['fit_invalid_accepted']+=int((accepted&~yy).sum())
        counts['fit_valid_excluded']+=int((~accepted&yy).sum())
        origin=[p['valid'] for p in pp if p['cache_age_s']==0]
        veto=not origin or not all(origin)
        counts['origin_veto_references']+=veto
        counts['origin_veto_valid_states']+=int(yy.sum()) if veto else 0
        for variant in sorted({p['receiver_drift_variant'] for p in pp}):
            trajectory=sorted([p for p in pp if p['receiver_drift_variant']==variant],key=lambda p:p['cache_age_s'])
            seen_invalid=False;recovered=0
            for p in trajectory:
                if not p['valid']:seen_invalid=True
                elif seen_invalid:recovered+=1
            counts['recovered_valid_states']+=recovered
            counts['trajectories_with_recovery']+=recovered>0
            counts['trajectories']+=1
    pred=np.load(OUT/'surface_geometry_predictions.npz')
    offsets=pred['offsets'];logits=pred['logits'];z=pred['z'];idx=pred['ref_index'];y=pred['y'].astype(bool)
    quant=np.floor(offsets/STEP)*STEP
    unquant=(offsets[idx]-z@normals.T).min(1)>=GUARD
    quantaccept=(quant[idx]-z@normals.T).min(1)>=GUARD
    originaccept=logits[idx]>=0
    counts['learned_validation_states']=int((split=='validation').sum())
    v=split=='validation'
    counts['validation_valid_excluded_by_quantization']=int((v&y&unquant&~quantaccept&originaccept).sum())
    counts['validation_valid_excluded_by_origin_gate']=int((v&y&quantaccept&~originaccept).sum())
    counts['validation_valid_excluded_by_learned_geometry']=int((v&y&~quantaccept).sum())
    counts['validation_invalid_accepted_unshrunk']=int((v&~y&quantaccept&originaccept).sum())
    return {'scope':'training_label_fit_and_validation_learned_boundary','counts':dict(counts),
        'geometry_and_origin_exclusions_overlap':'origin-gate count is conditional on geometric acceptance; teacher-fit origin-veto counts are descriptive and can overlap fit exclusions',
        'nonconvex_population_domain_certified':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-ablations',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    start=time.perf_counter();refs,points,a,split,scenarios,manifest=data()
    y=np.array([p['valid'] for p in points],np.float32)
    log(stage='data_loaded',states=len(y),references=len(refs),device=device)
    predictions=heuristic_scores(a)
    full=a['reference_margins'].astype(np.float32)
    mlp,_=train_flexible(full,y,split,device);predictions['flexible_mlp']=mlp
    tree_runs=[];best=None
    for leaves in (15,31):
        model=HistGradientBoostingClassifier(max_iter=180,max_leaf_nodes=leaves,learning_rate=.1,
            l2_regularization=1.,early_stopping=False,random_state=SEEDS[0])
        model.fit(full[split=='train'],y[split=='train'],sample_weight=np.where(y[split=='train']>.5,1.,4.))
        score=model.predict_proba(full)[:,1];auc=float(roc_auc_score(y[split=='validation'],score[split=='validation']))
        tree_runs.append({'leaves':leaves,'validation_auroc':auc});log(stage='tree',**tree_runs[-1])
        if best is None or (auc,-leaves)>(best[0],-best[1]):best=(auc,leaves,model,score)
    joblib.dump(best[2],OUT/'flexible_tree.joblib');predictions['flexible_tree']=best[3]
    ablations={}
    if not args.skip_ablations:
        more,ablations=train_ablations(refs,points,a,split,device);predictions.update(more)
    # Apply the historical models and shrinkages as frozen comparators.
    registration=json.loads((ORIGINAL/'outputs/study_b/slot_confirmation_v1/preregistration.json').read_text())
    for name,rule in registration['operating_rules'].items():
        path=ORIGINAL/rule['path'];assert sha(path)==rule['sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=DecisionSurfaceNet(saved['feature_count'],saved['scalar']);model.load_state_dict(saved['model_state'])
        x=np.clip((a['margin_reference']-saved['mean'].numpy())/saved['std'].numpy(),-10,10).astype(np.float32)
        _,offsets,logits=surface_predictions(model,x,a['reference_index'],a['drift'].astype(np.float32))
        predictions['frozen_'+name]=membership_score(torch.from_numpy(np.maximum(0,offsets-rule['shrinkage']))[a['reference_index']],
            torch.from_numpy(logits)[a['reference_index']],torch.from_numpy(a['drift'].astype(np.float32)),scalar=saved['scalar']).numpy()
    methods={}
    for name,score in predictions.items():
        frozen=name.startswith('frozen_')
        threshold=0. if frozen else operating_threshold(score[split=='validation'])
        methods[name]={'threshold':threshold,'selection':'historical_frozen_rule' if frozen else 'validation_coverage_at_most_80_percent_no_outcome_tie_break',
            'validation':report_metrics(y[split=='validation'],score[split=='validation'],threshold,scenarios[split=='validation']),
            'development_calibration':report_metrics(y[split=='calibration'],score[split=='calibration'],threshold,scenarios[split=='calibration'])}
    # Threshold sensitivity is a retrospective development analysis, not an
    # operating-point search on confirmation outcomes.
    sensitivity={}
    for eps,h,d in [(2.5,.4,2),(5,.2,2),(5,.4,1),(5,.4,2),(10,.4,2),(5,.6,2),(5,.4,3)]:
        label=np.array([p['cost_regret']<=eps and p['cached_risk']<=h and p['trajectory_deviation_m']<=d for p in points])
        mask=split=='calibration'
        sensitivity[f'{eps}/{h}/{d}']={n:report_metrics(label[mask],predictions[n][mask],methods[n]['threshold'],scenarios[mask],replicates=500)
            for n in ('frozen_surface','frozen_scalar_ttl','flexible_tree','flexible_mlp')}
    result={'scope':'development_only_no_historical_confirmation_loaded','methods':methods,'tree_runs':tree_runs,
        'ablations':ablations,'threshold_sensitivity':sensitivity,
        'behavior_ablation_scope':'explicit behavior ID removed; intended path and observed planner features can still reveal maneuver; not a full behavior-blind counterfactual',
        'platform':platform.platform(),'device':torch.cuda.get_device_name(0) if device=='cuda' else platform.processor(),
        'wall_seconds':time.perf_counter()-start,'source_data_manifest_sha256':sha(ORIGINAL/'outputs/study_b/controlled_development_v2/run_manifest.json'),
        'script_sha256':sha(__file__),'fresh_confirmation_required':True}
    if not args.skip_ablations:result['geometry_audit']=geometry_audit(refs,points,a,split)
    save_json(OUT/'development_report.json',result)
    np.savez_compressed(OUT/'development_predictions.npz',y=y,split=split,scenarios=scenarios,**predictions)
    log(stage='complete',wall_seconds=result['wall_seconds'],methods=len(methods))

if __name__=='__main__':main()
