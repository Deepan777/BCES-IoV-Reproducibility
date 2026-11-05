"""Receiver-local policies sharing payloads, causal context and refresh transport.

The original source-query architecture remains a historical comparator. These
new variants execute the fixed predictor on the receiver; a refresh request and
response payload are still charged, while there is no artificial 948-byte query.
"""
from __future__ import annotations
import math
from pathlib import Path
from types import SimpleNamespace
import time
import hashlib
import json

import joblib
import numpy as np
import torch
from development_study import ORIGINAL,OUT,FlexibleMLP
from bces.evaluation.study_b import reference_features
from bces.geometry.drift import compute_drift
from bces.models.decision_surface import DecisionSurfaceNet,CAP,STEP,GUARD
from bces.models.surfacenet import normal_tensor
from bces.protocol.receiver import DecisionState
from bces.network.wire_channel import refresh_control
from bces.protocol.bound_exchange import decode_objects

class LocalRule:
    def __init__(self,name,*,threshold=None,max_age_s=None,min_refresh_ms=0):
        if name=='cam_fixed':name='cam_inspired';threshold=-1.
        self.name=name;self.timing_prepare_ns=[];self.timing_check_ns=[]
        self.max_age_s=max_age_s
        self.min_refresh_ms=min_refresh_ms
        self.policy_hash=json.loads((ORIGINAL/'outputs/study_b/slot_confirmation_v1/preregistration.json').read_text())['policy_hash']
        self.model=None;self.mean=None;self.std=None;self.saved=None;self.scalar=False;self.explicit_behavior=True
        if name.startswith('frozen_'):
            family=name[len('frozen_'):]
            rule=json.loads((ORIGINAL/'outputs/study_b/slot_confirmation_v1/preregistration.json').read_text())['operating_rules'][family]
            path=ORIGINAL/rule['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==rule['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            self.shrinkage=rule['shrinkage'];self.threshold=0.
            self._surface(saved)
        elif name.startswith(('surface_teacher','ttl_teacher')):
            self._surface(torch.load(OUT/(name+'.pt'),map_location='cpu',weights_only=False))
            self.shrinkage=0.
        elif name=='flexible_mlp':
            saved=torch.load(OUT/'flexible_mlp.pt',map_location='cpu',weights_only=False)
            self.saved=saved;self.mean=saved['mean'];self.std=saved['std']
            self.model=FlexibleMLP(saved['features']);self.model.load_state_dict(saved['state']);self.model.eval()
        elif name=='flexible_tree':self.model=joblib.load(OUT/'flexible_tree.joblib')
        if not name.startswith('frozen_'):
            report=OUT/'development_report.json'
            self.threshold=float(threshold if threshold is not None else json.loads(report.read_text())['methods'][name]['threshold'])
        elif threshold is not None:self.threshold=float(threshold)

    def _surface(self,saved):
        self.saved=saved;self.scalar=saved['scalar']
        self.mean=saved['mean'].numpy();self.std=saved['std'].numpy()
        self.model=DecisionSurfaceNet(saved['feature_count'],self.scalar)
        self.model.load_state_dict(saved['model_state']);self.model.eval()
        self.explicit_behavior=saved.get('explicit_behavior',True)

    def prepare(self,reference):
        start=time.perf_counter_ns()
        base=reference_features(reference)
        if not self.explicit_behavior:base[-6]=0
        features=np.concatenate([base,np.array(reference['observable_margin_features'],np.float32)])
        state={'reference':reference,'features':features}
        if isinstance(self.model,DecisionSurfaceNet):
            x=np.clip((features-self.mean)/self.std,-10,10).astype(np.float32)
            with torch.inference_mode():offsets,logits=self.model(torch.from_numpy(x)[None])
            offsets=np.maximum(0,offsets[0].numpy()-self.shrinkage)
            state.update(offsets=np.floor(offsets/STEP).clip(0,255)*STEP,origin=bool(logits[0]>=0))
        self.timing_prepare_ns.append(time.perf_counter_ns()-start)
        return state

    def score(self,state,z,current_features=None):
        start=time.perf_counter_ns();z=np.asarray(z,np.float32)
        if self.max_age_s is not None and z[6]*4>self.max_age_s+1e-7:
            self.timing_check_ns.append(time.perf_counter_ns()-start)
            return -1e9
        if isinstance(self.model,DecisionSurfaceNet):
            if not state['origin']:score=-CAP-GUARD
            elif self.scalar:score=float(state['offsets'][0]-z[6]-GUARD)
            else:score=float((state['offsets']-normal_tensor().numpy()@z).min()-GUARD)
        elif self.name=='flexible_mlp':
            x=np.concatenate([state['features'],z]);x=np.clip((x-self.mean)/self.std,-10,10).astype(np.float32)
            with torch.inference_mode():score=float(self.model(torch.from_numpy(x)[None])[0])
        elif self.name=='flexible_tree':
            score=float(self.model.predict_proba(np.concatenate([state['features'],z])[None])[0,1])
        elif self.name=='age_only':score=-float(z[6])
        elif self.name=='normalized_drift':score=-float(np.abs(z).max())
        elif self.name=='cam_inspired':score=-max(math.hypot(z[0]*40,z[1]*5)/4,abs(z[2]*5)/.5,abs(z[3]*.7)/math.radians(4),z[6]*4)
        elif self.name=='observable_cost_trigger':
            if current_features is None:raise ValueError('observable VoI proxy requires cached/local planner features')
            score=-max(abs(current_features[6]-state['reference']['observable_margin_features'][6])/5,z[6])
        else:raise ValueError(self.name)
        self.timing_check_ns.append(time.perf_counter_ns()-start)
        return score

class LocalSession:
    """Causal request/payload state machine, fixed timeout and one retry.

    All delivered datagrams are processed at their event times. A received
    payload is tied to its sampled historical reference; it is never rebased to
    arrival time. A failed reference or missing input leaves local fallback.
    """
    def __init__(self,channel,policy,build_reference,payload_provider,*,timeout_ms=400):
        self.channel=channel;self.policy=policy;self.build_reference=build_reference
        self.payload_provider=payload_provider;self.timeout_ms=timeout_ms
        self.receiver=SimpleNamespace(payload=None);self.reference=None;self.state=None;self.query=None
        self.pending_since=None;self.first_request=None;self.attempt=0;self.sequence=0
        self.last_request_ms=-1e12
        self.counts={};self.latency_ms=[];self.reference_age_ms=[];self.last_timestamp=-1

    def advance(self,now):
        while (event:=self.channel.next_delivery(now)) is not None:
            if event.kind=='refresh':
                self.channel.send('payload',self.payload_provider(event.delivered_ms),event.delivered_ms,attempt=event.attempt)
            elif event.kind=='payload':
                message=decode_objects(event.body)
                if message.timestamp_ms<=self.last_timestamp:
                    self.channel.discard(event);continue
                self.last_timestamp=message.timestamp_ms
                try:reference,query=self.build_reference(event.body,event.delivered_ms)
                except (ValueError,RuntimeError):
                    self.state=None;self.counts['reference_unavailable']=self.counts.get('reference_unavailable',0)+1
                else:
                    self.reference,self.query=reference,query
                    self.state=self.policy.prepare(reference);self.receiver.payload=event.body
                    self.counts['new_references']=self.counts.get('new_references',0)+1
                    self.reference_age_ms.append(event.delivered_ms-message.timestamp_ms)
                if self.first_request is not None:self.latency_ms.append(event.delivered_ms-self.first_request)
                self.pending_since=None;self.first_request=None;self.attempt=0

    def decision(self,current,behavior,current_features=None):
        now=current.timestamp_ms;self.advance(now)
        eligible=self.state is not None and int(self.query.behavior)==int(behavior)
        accepted=False;reason='no_reference'
        if eligible:
            z=compute_drift(self.query.reference_state,current,self.query.drift_scales).normalized
            accepted=self.policy.score(self.state,z,current_features)>=self.policy.threshold
            reason='accepted' if accepted else 'policy_rejection'
        if not accepted:
            self.counts[reason]=self.counts.get(reason,0)+1
            if now-self.last_request_ms<getattr(self.policy,'min_refresh_ms',0):
                self.counts['cooldown_local_fallback']=self.counts.get('cooldown_local_fallback',0)+1
                return SimpleNamespace(state=DecisionState.REFRESH)
            if self.pending_since is None:
                self.sequence+=1;self.pending_since=now;self.first_request=now;self.attempt=0
                self.last_request_ms=now
                self.channel.send('refresh',refresh_control(0,self.sequence),now)
            elif now-self.pending_since>=self.timeout_ms:
                self.counts['timeouts']=self.counts.get('timeouts',0)+1
                if self.attempt<1:
                    self.attempt+=1;self.pending_since=now
                    self.last_request_ms=now
                    self.channel.send('refresh',refresh_control(0,self.sequence),now,attempt=self.attempt)
                else:self.pending_since=None;self.first_request=None
        return SimpleNamespace(state=DecisionState.ACCEPT_REUSE if accepted else DecisionState.REFRESH)

    def report(self):
        return {**self.channel.report(),'receiver_local':True,'query_bytes':0,'refresh_causes':self.counts,
            'request_payload_latency_ms':self.latency_ms,'reference_age_on_install_ms':self.reference_age_ms,
            'source_inference_required':False}
