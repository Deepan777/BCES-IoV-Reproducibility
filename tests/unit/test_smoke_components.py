from __future__ import annotations

import socket

import pytest

from bces.data.objects import ObjectMessage
from bces.data.smoke import SMOKE_OCCLUDED_TRACK_ID, generate_smoke_fixture
from bces.evaluation.validity import compute_validity_metrics
from bces.models.batch import MAX_MESSAGE_OBJECTS, build_model_batch_row
from bces.oracle.validity import (
    ValidityObservation,
    ValidityThresholds,
    evaluate_validity,
)
from bces.protocol.query import Behavior
from bces.protocol.receiver import DecisionState
from bces.smoke.network_guard import NetworkAccessProhibited, deny_network_access
from bces.smoke.pipeline import build_smoke_query, run_smoke_pipeline

pytestmark = pytest.mark.phase2


def test_generated_fixture_has_registered_contents() -> None:
    fixture = generate_smoke_fixture()
    assert len(fixture.frames) == 20
    assert {frame.intended_behavior for frame in fixture.frames} == {
        Behavior.KEEP,
        Behavior.BRAKE,
    }
    classes = [item.class_id for item in fixture.cached_message.objects]
    assert classes.count(1) == 2
    assert classes.count(2) == 1
    assert all(
        frame.occluded_track_ids == (SMOKE_OCCLUDED_TRACK_ID,)
        for frame in fixture.frames
    )
    assert fixture.evidence_use == "software_verification_only"


def test_object_message_encoding_is_deterministic() -> None:
    message = generate_smoke_fixture().cached_message
    assert message.encode() == message.encode()
    with pytest.raises(ValueError, match="unique"):
        ObjectMessage(
            message_id=message.message_id,
            sender_id=message.sender_id,
            timestamp_ms=message.timestamp_ms,
            objects=(message.objects[0], message.objects[0]),
        )


def test_validity_rule_is_registered_conjunction() -> None:
    thresholds = ValidityThresholds(0.1, 0.2, 0.3)
    assert evaluate_validity(ValidityObservation(1.1, 1.0, 0.2, 0.3), thresholds)
    assert not evaluate_validity(
        ValidityObservation(1.100001, 1.0, 0.2, 0.3), thresholds
    )
    assert not evaluate_validity(ValidityObservation(1.0, 1.0, 0.21, 0.3), thresholds)
    assert not evaluate_validity(ValidityObservation(1.0, 1.0, 0.2, 0.31), thresholds)


def test_model_batch_is_padded_and_inference_only() -> None:
    fixture = generate_smoke_fixture()
    result = run_smoke_pipeline(fixture)
    assert len(result["rows"]) == 20
    batch = build_model_batch_row(
        fixture.cached_message,
        build_smoke_query(fixture, Behavior.KEEP),
        occlusion_proxy=1 / 3,
        estimated_delay_s=0.0,
    )
    assert len(batch.object_features) == MAX_MESSAGE_OBJECTS
    assert batch.object_count == 3
    assert sum(batch.object_mask) == 3
    assert len(batch.reference_path) == 12


def test_metric_denominators() -> None:
    metrics = compute_validity_metrics(
        [
            (DecisionState.ACCEPT_REUSE, True),
            (DecisionState.ACCEPT_REUSE, False),
            (DecisionState.REFRESH, False),
            (DecisionState.INVALID, False),
        ]
    )
    assert metrics.coverage == 0.5
    assert metrics.unsafe_accept_rate == 0.5
    assert metrics.accepted_count == 2
    zero = compute_validity_metrics([(DecisionState.INVALID, False)])
    assert zero.unsafe_accept_rate is None


def test_network_guard_fails_closed() -> None:
    with deny_network_access() as state, pytest.raises(NetworkAccessProhibited):
        socket.create_connection(("example.invalid", 80))
    assert state.attempts == 1
