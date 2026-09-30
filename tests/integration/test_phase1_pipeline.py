from __future__ import annotations

import json
from pathlib import Path

import pytest

from bces.geometry.drift import compute_drift
from bces.geometry.surfaces import ExpirySurface
from bces.protocol.codec import decode_packet, dequantize_offsets, encode_packet
from bces.protocol.query import Behavior
from bces.protocol.receiver import DecisionState, ReceiverStateMachine
from tests.helpers_phase1 import (
    PAYLOAD,
    POLICY_HASH,
    SENDER_ID,
    make_packet,
    make_query,
    make_state,
)

pytestmark = pytest.mark.phase1
ROOT = Path(__file__).resolve().parents[2]


def test_query_drift_surface_codec_receiver_pipeline() -> None:
    query = make_query()
    packet = make_packet(query)
    encoded = encode_packet(packet)
    decoded = decode_packet(encoded)
    surface = ExpirySurface(
        offsets=dequantize_offsets(decoded),
        codebook_id=decoded.normal_codebook_id,
        calibration_id=decoded.calibration_id,
    )
    zero_drift = compute_drift(query.reference_state, make_state(), query.drift_scales)
    assert surface.contains(zero_drift.normalized)

    machine = ReceiverStateMachine(refresh_guard_band=0.1)
    decisions = [
        machine.evaluate(
            encoded_packet=encoded,
            cooperative_payload=PAYLOAD,
            query=query,
            current_state=current,
            current_behavior=Behavior.KEEP,
            current_policy_hash=POLICY_HASH,
            actual_sender_id=SENDER_ID,
        ).state
        for current in (make_state(), make_state(x_m=9.5), make_state(x_m=11.0))
    ]
    assert decisions == [
        DecisionState.ACCEPT_REUSE,
        DecisionState.REFRESH,
        DecisionState.INVALID,
    ]


def test_golden_vector_decodes_independently_from_fixture() -> None:
    vector = json.loads(
        (ROOT / "tests/protocol_vectors/v1_golden.json").read_text(encoding="utf-8")
    )
    encoded = bytes.fromhex(vector["expected_hex"])
    packet = decode_packet(encoded)
    assert packet.query_id == 0x01020304
    assert packet.sender_id_hash == 0x11223344
    assert packet.payload_digest == 0x55667788
    assert packet.policy_hash == 0x99AABBCC
    assert len(encoded) == 48
