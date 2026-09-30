from dataclasses import replace
import hashlib
import numpy as np
import pytest

from bces.data.objects import ObjectMessage
from bces.models.decision_surface import encode_conservative_surface
from bces.protocol.bindings import digest32
from bces.protocol.bound_exchange import (BoundQuery, BoundReceiver, encode_response, decode_response,
    scalar_extension, QUERY_HEADER, RESPONSE_HEADER)
from bces.protocol.receiver import DecisionState
from tests.helpers_phase1 import make_query, make_state, SENDER_ID


def fixture(family='surface', view='reference_margins'):
    q = make_query()
    payload = ObjectMessage(1, SENDER_ID, q.reference_state.timestamp_ms, ()).encode()
    bound = BoundQuery(q, hashlib.sha256(payload).hexdigest(), 'a'*64, view, family, 0.,
        tuple(np.arange(29)/7) if view == 'reference_margins' else (), q.reference_state.timestamp_ms, 0., 0., 1)
    if family == 'surface':
        extension = encode_conservative_surface([1.]*16, origin_permitted=True,
            bindings=dict(behavior_id=int(q.behavior), query_id=q.query_id,
                sender_id_hash=q.expected_sender_id_hash, payload_digest=digest32(payload),
                policy_hash=q.policy_hash, t_ref_ms=q.wire_t_ref_ms, risk_class=q.risk_class,
                calibration_id=q.calibration_id))
    else:
        extension = scalar_extension(1., True)
    return bound, payload, extension


def test_exact_sidecar_response_sizes_and_float32_roundtrip():
    b, payload, extension = fixture()
    wire = b.encode()
    assert BoundQuery.decode(wire) == b
    assert np.array_equal(np.asarray(b.margins, np.float32), np.arange(29, dtype=float).astype(np.float32)/np.float32(7))
    _, _, n, side = QUERY_HEADER.unpack_from(wire)
    assert side == 116 and len(wire) == QUERY_HEADER.size+n+116
    assert len(wire) == 948
    response = encode_response(wire, extension)
    assert len(extension) == 48 and len(response) == RESPONSE_HEADER.size+48
    assert decode_response(response, wire) == extension
    assert b.validate_payload(payload).sender_id == SENDER_ID


def test_binary_query_preserves_float64_context_and_fair_query_size():
    b, _, _ = fixture()
    b = replace(b,query=replace(b.query,reference_state=replace(b.query.reference_state,x_m=1.2345678901234567)))
    assert BoundQuery.decode(b.encode()).query.reference_state.x_m == b.query.reference_state.x_m
    assert len(replace(b,family='scalar_ttl').encode()) == len(b.encode())
    base = replace(b,feature_view='frozen_inputs',margins=())
    assert len(base.encode()) == len(b.encode())-116


@pytest.mark.parametrize('field,value', [('margins', (0.,)*29), ('model_sha256', 'b'*64),
    ('shrinkage', .2), ('occlusion_proxy', .5), ('map_context_flags', 2)])
def test_full_context_mutations_cannot_reuse_original_response(field, value):
    b, _, extension = fixture()
    response = encode_response(b.encode(), extension)
    changed = replace(b, **{field:value})
    with pytest.raises(ValueError): decode_response(response, changed.encode())
    receiver = BoundReceiver()
    receiver.register(b.encode())
    with pytest.raises(ValueError): receiver.register(changed.encode())


@pytest.mark.parametrize('family', ['surface', 'scalar_ttl'])
def test_end_to_end_binding_and_pending_fallback(family):
    b, payload, extension = fixture(family)
    receiver = BoundReceiver()
    receiver.register(b.encode())
    q = b.query
    args = (q.reference_state, q.behavior, q.policy_hash, SENDER_ID)
    assert receiver.evaluate(*args).state == DecisionState.REFRESH
    response = encode_response(b.encode(), extension)
    with pytest.raises(ValueError): receiver.install(response, payload+b' ')
    receiver.install(response, payload)
    assert receiver.evaluate(*args).state == DecisionState.ACCEPT_REUSE
    assert receiver.evaluate(args[0], args[1], args[2]+1, args[3]).state == DecisionState.INVALID
    next_query = replace(b, query=replace(q, query_id=q.query_id+1))
    receiver.register(next_query.encode())
    assert receiver.evaluate(*args).state == DecisionState.REFRESH
    with pytest.raises(ValueError): receiver.install(response, payload)
    with pytest.raises(ValueError): receiver.register(b.encode())


def test_future_nonfinite_and_malformed_inputs_rejected():
    b, _, _ = fixture()
    with pytest.raises(ValueError): replace(b, context_available_ms=b.context_available_ms+1)
    with pytest.raises(ValueError): replace(b, margins=(float('nan'),)*29)
    with pytest.raises(ValueError): BoundQuery.decode(b.encode()+b'x')
    with pytest.raises(ValueError): BoundQuery.decode(b.encode()[:5])
