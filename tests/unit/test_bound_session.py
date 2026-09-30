from dataclasses import replace
from types import SimpleNamespace
import pytest

from bces.network.bound_session import BoundSession, ReferenceUnavailable
from bces.network.events import NetworkCondition
from bces.network.wire_channel import WireChannel
from bces.protocol.bound_exchange import encode_response
from bces.protocol.receiver import DecisionState
from tests.unit.test_bound_exchange import fixture


def session(*,loss=0.,latency=10.,timeout=100.,retries=1,unavailable=False):
    bound,payload,extension = fixture('scalar_ttl')
    # Scalar replies have no embedded query ID, but retain full response binding.
    policy = SimpleNamespace(family=bound.family,sha256=bound.model_sha256,
        policy_hash=bound.query.policy_hash,issue=lambda wire,raw:encode_response(wire,extension))
    def build(raw,query_id,arrival):
        if unavailable: raise ReferenceUnavailable('known reference infeasibility')
        return replace(bound,query=replace(bound.query,query_id=query_id),generated_ms=arrival,
                       payload_received_ms=arrival,context_available_ms=arrival)
    channel = WireChannel(NetworkCondition(latency,loss,10),seed=33,security_bytes=64)
    app = BoundSession(channel=channel,policy=policy,build_query=build,
        payload_provider=lambda now:payload,sender_id='rsu-phase1',
        computation_delay_ms=6.,timeout_ms=timeout,max_retries=retries)
    return app,bound.query.reference_state,payload


def test_four_leg_refresh_exchange_is_causal_and_pending_is_local_only():
    app,state,payload = session()
    decision = app.decision(state,0)
    assert decision.state == DecisionState.REFRESH
    app.advance(state.timestamp_ms+80)
    current = replace(state,timestamp_ms=state.timestamp_ms+80)
    assert app.decision(current,0).state == DecisionState.ACCEPT_REUSE
    report = app.report()
    assert report['byte_conservation_ok'] and report['component_conservation_ok']
    assert report['generated_packets']==4
    assert report['generated_bytes']==13+len(payload)+948+74+4*(10+64)
    assert report['session_counts']['installed_responses']==1
    assert report['session_counts']['reason:scalar_inside']>=1
    issued = next(e for e in report['timeline'] if e['event']=='response_scheduled')
    assert issued['time_ms']==issued['query_received_ms']+6.
    assert all(e['time_ms']>=e['generated_ms'] for e in report['timeline'] if 'generated_ms' in e)


def test_total_loss_counts_retry_and_eventually_allows_new_refresh():
    app,state,_ = session(loss=1.)
    assert app.decision(state,0).request_refresh
    app.advance(state.timestamp_ms+250)
    assert app.pending is None
    assert app.report()['session_counts']['request_timeouts']==1
    assert app.report()['session_counts']['request_retries']==1
    assert app.report()['generated_packets']==2
    assert app.report()['retry_bytes']>0
    assert app.decision(replace(state,timestamp_ms=state.timestamp_ms+250),0).request_refresh
    assert app.report()['generated_packets']==3


def test_unavailable_reference_never_issues_model_query():
    app,state,_ = session(unavailable=True)
    app.decision(state,0)
    app.advance(state.timestamp_ms+80)
    assert app.receiver.query_wire is None
    assert app.report()['session_counts']['reference_unavailable']==1
    assert app.report()['generated_packets']==2


def test_late_and_duplicate_replies_do_not_install_twice():
    app,state,_ = session(latency=30.,timeout=50.)
    app.decision(state,0)
    app.advance(state.timestamp_ms+350)
    report = app.report()
    assert report['session_counts']['installed_responses']==1
    assert report['session_counts'].get('duplicate_response',0)>=1
    assert report['byte_conservation_ok'] and report['component_conservation_ok']


def test_rejects_backwards_time_and_bad_timing():
    app,state,_ = session()
    app.advance(state.timestamp_ms)
    with pytest.raises(ValueError): app.advance(state.timestamp_ms-1)
    with pytest.raises(ValueError): session(timeout=0.)
