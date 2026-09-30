from __future__ import annotations

from dataclasses import replace

import pytest

from bces.protocol.codec import encode_packet
from bces.protocol.query import Behavior
from bces.protocol.receiver import (
    CacheOffer,
    DecisionState,
    EvidenceMode,
    PacketCache,
    ReceiverStateMachine,
)
from tests.helpers_phase1 import (
    PAYLOAD,
    POLICY_HASH,
    SENDER_ID,
    make_packet,
    make_path,
    make_query,
    make_state,
)

pytestmark = pytest.mark.phase1


def evaluate(machine: ReceiverStateMachine, **overrides: object):
    query = overrides.pop("query", make_query())
    packet = overrides.pop("packet", make_packet(query))
    arguments: dict[str, object] = {
        "encoded_packet": encode_packet(packet),
        "cooperative_payload": PAYLOAD,
        "query": query,
        "current_state": make_state(),
        "current_behavior": Behavior.KEEP,
        "current_policy_hash": POLICY_HASH,
        "actual_sender_id": SENDER_ID,
    }
    arguments.update(overrides)
    return machine.evaluate(**arguments)  # type: ignore[arg-type]


def test_query_requires_twelve_monotonic_path_points() -> None:
    with pytest.raises(ValueError, match="exactly 12"):
        make_query(reference_path=make_path()[:-1])
    path = list(make_path())
    path[2] = replace(path[2], time_offset_s=path[1].time_offset_s)
    with pytest.raises(ValueError, match="strictly increasing"):
        make_query(reference_path=tuple(path))


def test_receiver_accept_refresh_and_invalid_states() -> None:
    machine = ReceiverStateMachine(refresh_guard_band=0.1)
    accepted = evaluate(machine)
    assert accepted.state is DecisionState.ACCEPT_REUSE
    assert accepted.evidence_mode is EvidenceMode.COOPERATIVE_REUSE
    refreshed = evaluate(machine, current_state=make_state(x_m=9.5))
    assert refreshed.state is DecisionState.REFRESH
    assert refreshed.evidence_mode is EvidenceMode.LOCAL_ONLY
    assert refreshed.request_refresh
    invalid = evaluate(machine, current_state=make_state(x_m=11.0))
    assert invalid.state is DecisionState.INVALID
    assert invalid.reason == "surface_violated"
    assert invalid.request_refresh


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"cooperative_payload": b"different"}, "payload_mismatch"),
        ({"current_policy_hash": POLICY_HASH + 1}, "current_policy_mismatch"),
        ({"current_behavior": Behavior.BRAKE}, "current_behavior_mismatch"),
        ({"actual_sender_id": "another-rsu"}, "sender_mismatch"),
    ],
)
def test_runtime_binding_mismatches_invalidate(
    override: dict[str, object], reason: str
) -> None:
    decision = evaluate(ReceiverStateMachine(refresh_guard_band=0.1), **override)
    assert decision.state is DecisionState.INVALID
    assert decision.evidence_mode is EvidenceMode.LOCAL_ONLY
    assert decision.reason == reason


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("query_id", 8, "query_id_mismatch"),
        ("payload_digest", 0, "payload_mismatch"),
        ("policy_hash", 0, "query_policy_mismatch"),
        ("behavior_id", int(Behavior.BRAKE), "query_behavior_mismatch"),
        ("sender_id_hash", 0, "query_sender_mismatch"),
        ("calibration_id", 3, "calibration_mismatch"),
    ],
)
def test_packet_binding_mismatches_invalidate(
    field: str, value: int, reason: str
) -> None:
    packet = make_packet(**{field: value})
    decision = evaluate(ReceiverStateMachine(refresh_guard_band=0.1), packet=packet)
    assert decision.state is DecisionState.INVALID
    assert decision.reason == reason


def test_malformed_and_unsupported_packets_invalidate() -> None:
    machine = ReceiverStateMachine(refresh_guard_band=0.1)
    malformed = evaluate(machine, encoded_packet=b"short")
    assert malformed.state is DecisionState.INVALID
    encoded = bytearray(encode_packet(make_packet()))
    encoded[0] = 2
    unsupported = evaluate(machine, encoded_packet=bytes(encoded))
    assert unsupported.state is DecisionState.INVALID


def test_duplicate_is_idempotent_and_stale_packet_cannot_rollback() -> None:
    machine = ReceiverStateMachine(refresh_guard_band=0.1)
    first = evaluate(machine)
    duplicate = evaluate(machine)
    assert first.cache_offer is CacheOffer.INSERTED
    assert duplicate.cache_offer is CacheOffer.DUPLICATE
    old_query = make_query(query_id=6)
    stale = evaluate(machine, query=old_query, packet=make_packet(old_query))
    assert stale.state is DecisionState.INVALID
    assert stale.cache_offer is CacheOffer.STALE
    assert machine.cache.current(make_query().expected_sender_id_hash).query_id == 7  # type: ignore[union-attr]


def test_conflicting_duplicate_is_rejected_without_cache_mutation() -> None:
    machine = ReceiverStateMachine(refresh_guard_band=0.1)
    evaluate(machine)
    conflict_packet = make_packet(quantized_offsets=(1,) * 16)
    conflict = evaluate(machine, packet=conflict_packet)
    assert conflict.state is DecisionState.INVALID
    assert conflict.cache_offer is CacheOffer.CONFLICT
    assert machine.cache.current(make_query().expected_sender_id_hash) == make_packet()


def test_uint32_serial_wrap_replaces_newer_query() -> None:
    cache = PacketCache()
    near_wrap = make_packet(make_query(query_id=0xFFFFFFFF))
    after_wrap = make_packet(make_query(query_id=0))
    assert cache.offer(near_wrap) is CacheOffer.INSERTED
    assert cache.offer(after_wrap) is CacheOffer.REPLACED
