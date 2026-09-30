from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from bces.network.adaptive_margin_policy import AdaptiveMarginWirePolicy, select_reference_method
from bces.protocol.bound_exchange import BoundReceiver, decode_response, encode_response
from bces.protocol.receiver import DecisionState
from tests.unit.test_bound_exchange import fixture
from tests.helpers_phase1 import SENDER_ID


def reference(*, x=0.0, y=30.0, vx=0.0, vy=8.0, class_id=0, gap=0.25):
    return {"frozen_input": {"batch": {
        "reference_state": [0.0, 0.0, 8.0, 1.5707963267948966, 0.0, 0.0, 5000],
        "object_features": [[x, y, vx, vy, 5.0, 1.8, class_id]],
        "object_mask": [True],
    }}, "observable_margin_features": [0.0] * 12 + [gap]}


def test_lead_following_with_margin_selects_ttl():
    method, evidence = select_reference_method(reference())
    assert method == "scalar_ttl" and evidence["lead_like"]


def test_crossing_motion_does_not_select_ttl():
    assert select_reference_method(reference(vx=10.0, vy=0.0))[0] == "surface"


def test_low_margin_or_lateral_object_retains_surface():
    assert select_reference_method(reference(gap=-0.1))[0] == "surface"
    assert select_reference_method(reference(x=6.0))[0] == "surface"
    assert select_reference_method(reference(class_id=1))[0] == "surface"


def fake_pair():
    surface_bound, _, surface_extension = fixture("surface")
    ttl_bound, _, ttl_extension = fixture("scalar_ttl")
    policy_hash = surface_bound.query.policy_hash
    surface = SimpleNamespace(family="surface", view="reference_margins", policy_hash=policy_hash,
                              sha256="a" * 64, issue=lambda wire, payload: encode_response(wire, surface_extension))
    ttl = SimpleNamespace(family="scalar_ttl", view="reference_margins", policy_hash=policy_hash,
                          sha256="b" * 64, issue=lambda wire, payload: encode_response(wire, ttl_extension))
    return AdaptiveMarginWirePolicy(surface=surface, scalar_ttl=ttl,
                                    shrinkages={"surface": 0.0, "scalar_ttl": 0.0})


@pytest.mark.parametrize("family", ["surface", "scalar_ttl"])
def test_hybrid_rebinds_selected_extension_to_exact_query(family):
    policy = fake_pair()
    bound, payload, _ = fixture(family)
    bound = replace(bound, model_sha256=policy.sha256)
    response = policy.issue(bound.encode(), payload)
    receiver = BoundReceiver()
    receiver.register(bound.encode())
    receiver.install(response, payload)
    assert receiver.evaluate(bound.query.reference_state, bound.query.behavior,
                             bound.query.policy_hash, SENDER_ID).state == DecisionState.ACCEPT_REUSE
    other = replace(bound, family="surface" if family == "scalar_ttl" else "scalar_ttl")
    with pytest.raises(ValueError):
        decode_response(response, other.encode())


def test_hybrid_rejects_model_or_shrinkage_rebinding():
    policy = fake_pair()
    bound, payload, _ = fixture("surface")
    bound = replace(bound, model_sha256=policy.sha256)
    for changed in (replace(bound, model_sha256="c" * 64), replace(bound, shrinkage=0.1)):
        with pytest.raises(ValueError):
            policy.issue(changed.encode(), payload)
