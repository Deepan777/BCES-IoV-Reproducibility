from __future__ import annotations

from dataclasses import replace

import pytest

from bces.data.objects import ObjectMessage, ObjectSource, PerceivedObject
from bces.models.reference_input import freeze_reference_input
from tests.helpers_phase1 import SENDER_ID, make_query

pytestmark = pytest.mark.phase4


def build(message=None, query=None, **overrides):
    query = query or make_query()
    message = message or ObjectMessage(1, SENDER_ID, 1_000_000, (
        PerceivedObject("actor", 0, 12, 1, 3, 0, 4.5, 1.8, 0.9, ObjectSource.INFRASTRUCTURE),
    ))
    options = {
        "generated_at_ms": 1_000_000,
        "query_available_at_ms": 1_000_000,
        "context_available_at_ms": 1_000_000,
        "query_provenance": "received_query",
        "occlusion_proxy": 0.0,
        "estimated_delay_s": 0.0,
    }
    options.update(overrides)
    return freeze_reference_input(message, query, **options)


def test_reference_input_preserves_object_path_and_payload_query_binding():
    first = build()
    assert first == build()
    assert len(first.batch.object_features) == 32
    assert len(first.batch.object_features[0]) == 10
    assert len(first.batch.reference_path) == 12
    assert first.batch.object_count == 1
    changed_query = build(query=make_query(query_id=8))
    assert changed_query.payload_sha256 == first.payload_sha256
    assert changed_query.query_sha256 != first.query_sha256
    # Passing time does not change the same message/query features.
    assert build(generated_at_ms=1_000_200).feature_sha256 == first.feature_sha256


@pytest.mark.parametrize("field", ["query_available_at_ms", "context_available_at_ms"])
def test_future_sender_context_is_rejected(field):
    with pytest.raises(ValueError, match="unavailable"):
        build(**{field: 1_000_001})


def test_future_message_and_wrong_sender_are_rejected():
    with pytest.raises(ValueError, match="unavailable"):
        build(message=ObjectMessage(1, SENDER_ID, 1_000_001, ()))
    with pytest.raises(ValueError, match="sender"):
        build(message=ObjectMessage(1, "another-rsu", 1_000_000, ()))


def test_observed_future_behavior_or_path_cannot_be_a_query_proxy():
    with pytest.raises(ValueError, match="future observed"):
        build(query_provenance="whole_window_behavior_proxy")
    with pytest.raises(TypeError):
        build(current_receiver_state=replace(make_query().reference_state, x_m=20))
