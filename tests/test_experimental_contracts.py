"""Meaningful checks of causality and transport in the new study code."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from types import SimpleNamespace
from local_policies import LocalSession,LocalRule
from development_study import operating_threshold
from bces.network.wire_channel import WireChannel,refresh_control
from bces.network.events import NetworkCondition

def test_operating_point_does_not_split_tied_scores_by_labels():
    score=np.array([0.,0.,0.,1.,1.])
    threshold=operating_threshold(score,.8)
    assert (score>=threshold).tolist()==[False,False,False,True,True]

def test_cam_fixed_uses_physical_units():
    policy=LocalRule('cam_fixed')
    # 3.9 m displacement is below the 4 m trigger; 4.1 m is above it.
    z=np.zeros(7);z[0]=3.9/40
    assert policy.score({},z)>=policy.threshold
    z[0]=4.1/40
    assert policy.score({},z)<policy.threshold
    z[:]=0;z[3]=np.deg2rad(4.1)/.7
    assert policy.score({},z)<policy.threshold

def test_future_generation_is_not_counted_before_horizon():
    channel=WireChannel(NetworkCondition(50,0,10),seed=1,security_bytes=64)
    channel.send('refresh',refresh_control(0,1),1000)
    assert channel.next_delivery(500) is None
    assert channel.report()['generated_bytes']==0
    assert channel.next_delivery(1100).kind=='refresh'
    assert channel.report()['generated_bytes']==13+10+64
    assert channel.report()['byte_conservation_ok']

def test_no_reference_requests_then_waits_for_causal_delivery():
    channel=WireChannel(NetworkCondition(50,0,10),seed=1,security_bytes=64)
    provided=[]
    def payload(now):
        provided.append(now)
        # The test deliberately stops before delivery to the receiver.
        return b'payload-test'
    session=LocalSession(channel,None,None,payload)
    current=SimpleNamespace(timestamp_ms=0)
    session.decision(current,0)
    assert not provided
    session.advance(49)
    assert not provided
    session.advance(60)
    assert len(provided)==1 and 50<provided[0]<60
    assert session.reference is None

def test_refresh_cooldown_uses_local_fallback_without_reusing_rejected_payload():
    from bces.protocol.receiver import DecisionState
    channel=WireChannel(NetworkCondition(50,0,10),seed=1,security_bytes=64)
    policy=SimpleNamespace(min_refresh_ms=1200)
    session=LocalSession(channel,policy,None,lambda t:b'payload-test')
    first=session.decision(SimpleNamespace(timestamp_ms=0),0)
    second=session.decision(SimpleNamespace(timestamp_ms=20),0)
    assert first.state==second.state==DecisionState.REFRESH
    assert channel.counts['refresh_packets']==1
    assert session.counts['cooldown_local_fallback']==1

def test_live_policy_features_match_saved_development_predictions():
    import gzip,json
    from development_study import ORIGINAL,OUT
    path=sorted((ORIGINAL/'outputs/study_b/controlled_development_v2').glob('validation_*.json.gz'))[0]
    with gzip.open(path,'rt',encoding='utf-8') as f:scene=json.load(f)
    refs={r['reference_id']:r for r in scene['references']}
    saved=np.load(OUT/'development_predictions.npz')
    indices=np.flatnonzero(saved['scenarios']==scene['scenario_id'])
    assert len(indices)==len(scene['points'])
    for name in ('frozen_surface','frozen_scalar_ttl','flexible_mlp','flexible_tree','surface_teacher1_behavior0'):
        policy=LocalRule(name)
        for j in (0,len(indices)//2,len(indices)-1):
            point=scene['points'][j];reference=refs[point['reference_id']]
            score=policy.score(policy.prepare(reference),point['normalized_drift'],point['observable_current_features'])
            assert abs(score-float(saved[name][indices[j]]))<1e-4,(name,score,saved[name][indices[j]])
