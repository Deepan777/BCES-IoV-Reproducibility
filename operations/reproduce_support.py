"""Recompute supporting experimental metrics and frozen teacher-model inference."""
from collections import defaultdict
import gzip,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(x) for x in f]
def main():
    base=ROOT/'outputs/study_b';info=base/'diverse_empty_info_geometry_dev_v1';r=load(info/'report.json');pred=rows(info/'predictions.jsonl.gz');views=defaultdict(list)
    for x in pred:views[x['view']].append(x)
    metrics={k:float(roc_auc_score([x['valid'] for x in a],[x['oof_valid_probability'] for x in a])) for k,a in views.items()}
    assert all(metrics[k]==r['information'][k]['auroc'] for k in metrics)
    geometry=load(info/'geometry_references.json');capacity={k:sum(x[k] for x in geometry) for k in ['surface_accepted_valid','ttl_accepted_valid']}
    assert len(geometry)==r['geometry']['references']==192
    assert capacity=={k:r['geometry'][k] for k in capacity}
    print(json.dumps({'information_auroc':metrics,'geometry_capacity':capacity,'matches':True}),flush=True)
    plain=base/'diverse_empty_surface_ttl_learning_dev_v1';plain_report=load(plain/'report.json');plain_rows=rows(plain/'predictions.jsonl.gz')
    for family in ['surface','scalar_ttl']:
        for fraction in [.2,.4,.6,.8]:
            accepted=invalid=0
            for fold in range(8):
                block=[x for x in plain_rows if x['seed']%8==fold]
                order=np.argsort(-np.array([x['score'][family] for x in block]),kind='stable')[:math.floor(fraction*len(block))]
                accepted+=len(order);invalid+=sum(not block[i]['valid'] for i in order)
            expected=plain_report['summary'][family]['matched_coverage'][str(fraction)]
            assert (accepted,invalid)==(expected['accepted'],expected['invalid_accepted'])
    transfer=base/'diverse_empty_teacher_transfer_analysis_dev_v1';report=load(transfer/'report.json');pred=rows(transfer/'predictions.jsonl.gz');counts={}
    for family in ['surface_teacher','scalar_ttl_teacher','surface_no_teacher','scalar_ttl_no_teacher']:
        chosen=[x for x in pred if x['accepted'][family]['0.4']];counts[family]={'accepted':len(chosen),'invalid':sum(not x['valid'] for x in chosen)}
    assert [counts[k]['invalid'] for k in counts]==[1,22,19,124] and all(x['accepted']==1075 for x in counts.values())
    from reproduce import module
    import torch
    torch.set_num_threads(2)
    m=module(ROOT/'scripts/143_analyze_empty_teacher_transfer_dev.py','teacher_replay');raw=m._rows(load(m.SOURCE/'report.json'))
    assert [x['observable_key'] for x in raw]==[x['observable_key'] for x in pred]
    x=np.asarray([r['reference_vector'] for r in raw],dtype=np.float32);drift=np.asarray([r['drift'] for r in raw],dtype=np.float32)
    valid=np.asarray([r['valid'] for r in raw],bool);seeds=np.asarray([r['seed'] for r in raw])
    scores=m._inference(x,drift,load(m.MODELS/'report.json'),load(m.MODELS/'protocol.json'));masks={}
    for family,values in scores.items():
        assert np.allclose(values,[r['score'][family] for r in pred],rtol=0,atol=1e-5)
        masks[family]=m._accepted(values,.4)
        for fraction in [.2,.4,.6,.8]:assert m._counts(m._accepted(values,fraction),valid)==report['summary'][family]['matched_coverage'][str(fraction)]
    assert m._bootstrap(masks,valid,seeds)==report['cluster_bootstrap_40_percent']
    out={'information_auroc':metrics,'geometry_capacity':capacity,'plain_fold_learning':'matches','teacher_transfer_counts_40_percent':counts,'teacher_checkpoint_inference':'PASS','seed_cluster_bootstrap':'matches'}
    p=ROOT/'reproduced/supporting_diagnostics.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
