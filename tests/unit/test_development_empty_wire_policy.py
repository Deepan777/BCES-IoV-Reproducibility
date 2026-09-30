from dataclasses import asdict, replace
import hashlib

import numpy as np
import pytest
import torch

from bces.data.objects import ObjectMessage
from bces.models.decision_surface import DecisionSurfaceNet
from bces.models.development_empty_wire_policy import (
    DevelopmentEmptyWirePolicy, OLD_LABEL_REPORT_SHA256,
    raw_reference_vector, raw_wire_features,
)
from bces.models.reference_input import freeze_reference_input
from bces.protocol.bound_exchange import BoundQuery, BoundReceiver, decode_response
from bces.protocol.query import Behavior
from bces.utils.hashing import sha256_file
from tests.helpers_phase1 import SENDER_ID, make_query


def fixture(tmp_path, family):
    scalar = family.startswith("scalar_ttl")
    query = make_query(calibration_id=0)
    message = ObjectMessage(1, SENDER_ID, query.reference_state.timestamp_ms, ())
    payload = message.encode()
    model = DecisionSurfaceNet(448, scalar=scalar)
    checkpoint = tmp_path / "candidate.pt"
    torch.save({
        "model_state": model.state_dict(),
        "mean": torch.zeros(448), "std": torch.ones(448),
        "family": family, "scalar": scalar, "feature_count": 448,
        "policy_hash": query.policy_hash,
        "source_report_sha256": OLD_LABEL_REPORT_SHA256,
    }, checkpoint)
    policy = DevelopmentEmptyWirePolicy(
        checkpoint, expected_sha256=sha256_file(checkpoint),
        policy_hash=query.policy_hash)
    bound = BoundQuery(
        query, hashlib.sha256(payload).hexdigest(), policy.sha256,
        "frozen_inputs", policy.family, 0.0, (),
        query.reference_state.timestamp_ms, 0.0, 0.0, 1,
    )
    return policy, bound, payload, message


@pytest.mark.parametrize("family", ["surface_teacher", "scalar_ttl_teacher"])
def test_exact_raw_wire_features_and_bound_response(tmp_path, family):
    policy, bound, payload, message = fixture(tmp_path, family)
    wire = bound.encode()
    frozen = freeze_reference_input(
        message, bound.query,
        generated_at_ms=bound.generated_ms,
        query_available_at_ms=bound.generated_ms,
        context_available_at_ms=bound.context_available_ms,
        query_provenance="received_query", occlusion_proxy=0.0,
        estimated_delay_s=0.0, map_context_flags=1,
    )
    expected = raw_reference_vector(asdict(frozen.batch))
    assert expected.shape == (448,)
    assert np.array_equal(raw_wire_features(BoundQuery.decode(wire), payload), expected)
    offsets, origin = policy.predict(wire, payload)
    assert len(offsets) == (1 if policy.family == "scalar_ttl" else 16)
    assert type(origin) is bool
    response = policy.issue(wire, payload)
    assert len(decode_response(response, wire)) == (2 if policy.family == "scalar_ttl" else 48)
    receiver = BoundReceiver()
    receiver.register(wire)
    receiver.install(response, payload)


@pytest.mark.parametrize("family", ["surface_teacher", "scalar_ttl_teacher"])
def test_mismatched_payload_model_family_and_context_fail_closed(tmp_path, family):
    policy, bound, payload, _ = fixture(tmp_path, family)
    changed = [
        (bound, payload + b" "),
        (replace(bound, model_sha256="0" * 64), payload),
        (replace(bound, family=("scalar_ttl" if policy.family == "surface" else "surface")), payload),
        (replace(bound, map_context_flags=0), payload),
        (replace(bound, query=replace(bound.query, behavior=Behavior.BRAKE)), payload),
    ]
    for other, candidate_payload in changed:
        with pytest.raises(ValueError):
            policy.predict(other.encode(), candidate_payload)


def test_raw_reference_rejects_malformed_or_nonfinite_batch(tmp_path):
    _, bound, payload, message = fixture(tmp_path, "surface_teacher")
    frozen = freeze_reference_input(
        message, bound.query,
        generated_at_ms=bound.generated_ms,
        query_available_at_ms=bound.generated_ms,
        context_available_at_ms=bound.context_available_ms,
        query_provenance="received_query", occlusion_proxy=0.0,
        estimated_delay_s=0.0, map_context_flags=1,
    )
    malformed = asdict(frozen.batch)
    malformed["object_features"] = []
    with pytest.raises(ValueError, match="448"):
        raw_reference_vector(malformed)
    nonfinite = asdict(frozen.batch)
    nonfinite["object_features"] = list(nonfinite["object_features"])
    nonfinite["object_features"][0] = tuple([float("nan")] + list(nonfinite["object_features"][0][1:]))
    with pytest.raises(ValueError, match="finite"):
        raw_reference_vector(nonfinite)
