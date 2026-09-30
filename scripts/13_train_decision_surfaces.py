#!/usr/bin/env python3
"""Run the learning diagnostic only after information and geometry headroom."""
from __future__ import annotations

import gzip
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader,TensorDataset

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from bces.evaluation.decision_geometry import fit_zero_error_surface,fit_zero_error_ttl
from bces.evaluation.study_b import feature_arrays,load_development,selective_metrics,scenario_ratio_upper,predictive_metrics,origin_validity
from bces.geometry.codebooks import NORMAL_CODEBOOK_V1
from bces.models.decision_surface import DecisionSurfaceNet,decision_surface_loss,membership_score,CAP,GUARD
from bces.models.surfacenet import trainable_parameter_count
from bces.utils.budget import Budget,require_headroom,build_budget_report
from bces.utils.hashing import canonical_json_hash,sha256_file
from bces.utils.reproducibility import git_state,utc_now,write_json_atomic

CONFIG=ROOT/'configs/training/decision_surface_development_v1.yaml'


def evaluate(model,x,drift,valid,origin,device):
    model.eval()
    offsets,logits=[],[]
    total=0.
    with torch.no_grad():
        for start in range(0,len(x),1024):
            end=start+1024
            b=torch.from_numpy(x[start:end]).to(device)
            o,g=model(b)
            z=torch.from_numpy(drift[start:end]).to(device)
            y=torch.from_numpy(valid[start:end]).to(device)
            v=torch.from_numpy(origin[start:end]).to(device)
            total+=float(decision_surface_loss(o,g,z,y,v,scalar=model.scalar))*len(b)
            offsets.append(o.cpu()); logits.append(g.cpu())
    return total/len(x),torch.cat(offsets),torch.cat(logits)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=CONFIG)
    args=parser.parse_args()
    config_path=args.config.resolve()
    config=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    diagnostics=json.loads((ROOT/config['diagnostics']).read_text(encoding='utf-8'))
    if diagnostics['source_manifest_sha256'] != sha256_file(ROOT/config['data_root']/'run_manifest.json'):
        raise ValueError('diagnostic prerequisite must refer to the exact training data version')
    if diagnostics['original_test_accessed'] or diagnostics['scope']!='development_only':
        raise ValueError('development diagnostics required')
    geometry=diagnostics['geometry']
    information=max(diagnostics['information'][v]['validation']['auroc'] for v in ('frozen_inputs','reference_margins'))
    if information<config['minimum_diagnostic_auroc'] or geometry['surface_accepted_valid']-geometry['ttl_accepted_valid']<config['minimum_geometry_extra_valid_points']:
        print(json.dumps({'status':'DIAGNOSTIC_STOP','information_auroc':information,'geometry':geometry,
                          'reason':'registered learning prerequisite failed','manuscript_allowed':False},indent=2))
        return 2
    state=git_state(ROOT)
    if state['dirty']:
        raise RuntimeError('clean committed revision required')
    output=ROOT/config['output_root']
    if output.exists():
        raise FileExistsError('learning artifact immutable')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('registered single CUDA device required')
    if '3060' not in torch.cuda.get_device_name(0):
        raise RuntimeError('registered RTX 3060 required')
    device=torch.device('cuda:0')
    torch.set_num_threads(2)
    budget=Budget(workspace_root=ROOT,data_root=ROOT/'data')
    require_headroom(budget,incoming_bytes=50_000_000)
    refs,points,source=load_development(ROOT/config['data_root'])
    arrays=feature_arrays(refs,points)
    ids=arrays['reference_ids']; ref_index=arrays['reference_index']
    split=np.array([p['split'] for p in points])
    scenarios=np.array([p['scenario_id'] for p in points])
    valid=np.array([p['valid'] for p in points],np.float32)
    drift=arrays['drift'].astype(np.float32)
    masks={name:split==name for name in ('train','validation','calibration')}
    index_by_ref={key:i for i,key in enumerate(ids)}
    origin_by_ref=origin_validity(ids,points)
    by_ref=defaultdict(list)
    for i,p in enumerate(points):
        by_ref[p['reference_id']].append(i)
    origin=origin_by_ref[ref_index]
    train_refs=np.array([refs[key]['split']=='train' for key in ids])
    output.mkdir(parents=True)
    surface_teacher=np.zeros((len(ids),16),np.float32)
    scalar_teacher=np.zeros((len(ids),1),np.float32)
    teacher_audit=[]
    for number,key in enumerate(np.array(ids)[train_refs],1):
        selection=by_ref[key]
        row=index_by_ref[key]
        target=fit_zero_error_surface(drift[selection],valid[selection].astype(bool),NORMAL_CODEBOOK_V1,time_limit=3.)
        ttl=fit_zero_error_ttl(drift[selection,6],valid[selection].astype(bool))
        if not target['abstain']:
            surface_teacher[row]=np.minimum(CAP,np.array(target['offsets'])+GUARD)
        if not ttl['abstain']:
            scalar_teacher[row,0]=min(CAP,ttl['threshold']+GUARD)
        teacher_audit.append({'reference_id':str(key),'surface':target,'ttl':ttl})
        if number%200==0:
            print(json.dumps({'teacher_references':number,'total':int(train_refs.sum())}),flush=True)
    write_json_atomic(output/'training_teachers.json',{'scope':'training_labels_only','references':teacher_audit})
    runs,selected_models={},{}
    for view in config['feature_views']:
        reference=arrays['base_reference'] if view=='frozen_inputs' else arrays['margin_reference']
        mean=reference[train_refs].mean(0)
        std=reference[train_refs].std(0).clip(.01)
        normalized=((reference-mean)/std).clip(-10,10).astype(np.float32)
        x=normalized[ref_index]
        for family in config['families']:
            scalar=family=='scalar_ttl'
            teachers=scalar_teacher if scalar else surface_teacher
            train=TensorDataset(torch.from_numpy(x[masks['train']]),torch.from_numpy(drift[masks['train']]),
                torch.from_numpy(valid[masks['train']]),torch.from_numpy(origin[masks['train']]),
                torch.from_numpy(teachers[ref_index[masks['train']]]))
            key=f'{view}:{family}'
            runs[key]=[]
            best_run=None
            for seed in config['seeds']:
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
                model=DecisionSurfaceNet(reference.shape[1],scalar=scalar).to(device)
                count=trainable_parameter_count(model)
                if count>config['max_parameters']:
                    raise RuntimeError('model parameter cap exceeded')
                loader=DataLoader(train,batch_size=config['batch_size'],shuffle=True,num_workers=0,
                                  generator=torch.Generator().manual_seed(seed))
                optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
                scaler=torch.amp.GradScaler('cuda')
                best_loss=float('inf');best_weights=None;stale=0;history=[]
                for epoch in range(1,config['epochs']+1):
                    model.train()
                    loss_sum=0.
                    for batch in loader:
                        b,z,y,v,t=(part.to(device) for part in batch)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda',dtype=torch.float16):
                            offsets,logits=model(b)
                        loss=decision_surface_loss(offsets.float(),logits.float(),z,y,v,scalar=scalar,teacher=t)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        scaler.step(optimizer);scaler.update()
                        loss_sum+=float(loss.detach())*len(b)
                    vloss,_,_=evaluate(model,x[masks['validation']],drift[masks['validation']],valid[masks['validation']],origin[masks['validation']],device)
                    if not np.isfinite(vloss):
                        raise RuntimeError('nonfinite validation loss')
                    history.append({'epoch':epoch,'training_loss':loss_sum/len(train),'validation_loss':vloss})
                    if vloss<best_loss-1e-5:
                        best_loss=vloss;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
                    else:
                        stale+=1
                    if epoch%10==0:
                        print(json.dumps({'run':key,'seed':seed,'epoch':epoch,'validation_loss':vloss}),flush=True)
                    if torch.cuda.max_memory_allocated()>config['max_cuda_allocated_bytes']:
                        raise RuntimeError('CUDA cap exceeded')
                    if stale>=config['patience']:
                        break
                model.load_state_dict(best_weights)
                _,offsets,logits=evaluate(model,x,drift,valid,origin,device)
                score=membership_score(offsets,logits,torch.from_numpy(drift),scalar=scalar).numpy()
                metrics={}
                for partition,mask in masks.items():
                    metrics[partition]=selective_metrics(valid[mask],score[mask]>=0,scenarios[mask])
                    metrics[partition]['ranking']=predictive_metrics(valid[mask],1/(1+np.exp(-np.clip(score[mask]*5,-60,60))),scenarios[mask])
                path=output/f'{view}_{family}_{seed}.pt'
                torch.save({'model_state':best_weights,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),
                            'view':view,'scalar':scalar,'feature_count':reference.shape[1],'seed':seed},path)
                run={'seed':seed,'parameter_count':count,'validation_loss':best_loss,'epochs':epoch,
                     'history':history,'metrics':metrics,'checkpoint_sha256':sha256_file(path),
                     'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
                runs[key].append(run)
                if best_run is None or (best_loss,seed)<(best_run['validation_loss'],best_run['seed']):
                    best_run=run;selected_models[key]=(score,offsets.numpy(),logits.numpy())
                del model,optimizer,scaler
                torch.cuda.empty_cache()
                print(json.dumps({'completed':key,'seed':seed,'validation':metrics['validation']}),flush=True)
            runs[key]={'runs':runs[key],'selected_seed':best_run['seed']}
    calibration={}
    for key,(score,offsets,logits) in selected_models.items():
        scalar=key.endswith(':scalar_ttl')
        curve=[]
        for shrink in config['shrinkages']:
            candidate=membership_score(torch.from_numpy(np.maximum(0,offsets-shrink)),torch.from_numpy(logits),torch.from_numpy(drift),scalar=scalar).numpy()
            mask=masks['calibration']
            result=selective_metrics(valid[mask],candidate[mask]>=0,scenarios[mask])
            result['shrinkage']=shrink
            result['finite_sample_uar_upper']=scenario_ratio_upper(valid[mask],candidate[mask]>=0,scenarios[mask],maximum_points=config.get('maximum_points_per_scenario',105),comparisons=len(selected_models)*len(config['shrinkages']))
            curve.append(result)
        eligible=[r for r in curve if r['accepted']>=config['minimum_accepted'] and r['finite_sample_uar_upper']<=config['target_uar']]
        calibration[key]={'curve':curve,'selected':max(eligible,key=lambda r:r['coverage']) if eligible else None,
                          'target_met':bool(eligible),'action_if_target_unmet':'abstain'}
    with gzip.open(output/'predictions.jsonl.gz','xt',encoding='utf-8') as handle:
        for i,point in enumerate(points):
            handle.write(json.dumps({'point_id':point['point_id'],'scenario_id':point['scenario_id'],'split':point['split'],
                'valid':point['valid'],'score':{key:float(value[0][i]) for key,value in selected_models.items()}},separators=(',',':'))+'\n')
    report={'schema_version':1,'status':'PASS','scope':'development_only','scientific_success':False,
            'original_test_accessed':False,'manuscript_allowed':False,'runs':runs,'calibration':calibration,
            'regret_theorem_applied_as_certificate':False,'additional_reference_margin_query_bytes':116,
            'fresh_confirmation_required':True,'data_manifest_sha256':sha256_file(ROOT/config['data_root']/'run_manifest.json'),
            'diagnostics_sha256':sha256_file(ROOT/config['diagnostics']),'config_sha256':sha256_file(config_path),
            'deployed_receiver_evidence':False,'evaluation_world':diagnostics.get('evaluation_world','unaudited_proxy'),
            'artifacts':{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()},
            'git':state,'completed_utc':utc_now(),'budget':build_budget_report(budget)['measurements']}
    report['report_sha256']=canonical_json_hash(report)
    write_json_atomic(output/'run_manifest.json',report)
    print(json.dumps({'status':'PASS','calibration_target_met':{k:v['target_met'] for k,v in calibration.items()},'manuscript_allowed':False},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
