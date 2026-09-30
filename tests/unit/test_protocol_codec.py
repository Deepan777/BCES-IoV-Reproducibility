from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bces.protocol.codec import decode_packet, encode_packet, quantize_offsets
from bces.protocol.packet import (
    BCESPacket,
    MalformedPacketError,
    PacketError,
    UnsupportedVersionError,
)

pytestmark = pytest.mark.phase1
ROOT = Path(__file__).resolve().parents[2]


def golden() -> tuple[BCESPacket, str]:
    vector = json.loads(
        (ROOT / "tests/protocol_vectors/v1_golden.json").read_text(encoding="utf-8")
    )
    packet = BCESPacket(
        **{
            **vector["packet"],
            "quantized_offsets": tuple(vector["packet"]["quantized_offsets"]),
        }
    )
    return packet, vector["expected_hex"]


def test_exact_48_byte_golden_vector_and_network_order() -> None:
    packet, expected_hex = golden()
    encoded = encode_packet(packet)
    assert len(encoded) == 48
    assert encoded.hex() == expected_hex
    assert encoded[4:8] == bytes.fromhex("01020304")
    assert encoded[16:20] == bytes.fromhex("99aabbcc")
    assert decode_packet(encoded) == packet


@given(
    behavior=st.integers(min_value=0, max_value=4),
    query_id=st.integers(min_value=0, max_value=0xFFFFFFFF),
    sender=st.integers(min_value=0, max_value=0xFFFFFFFF),
    payload=st.integers(min_value=0, max_value=0xFFFFFFFF),
    policy=st.integers(min_value=0, max_value=0xFFFFFFFF),
    t_ref=st.integers(min_value=0, max_value=0xFFFFFFFF),
    scale=st.integers(min_value=1, max_value=0xFFFF),
    offsets=st.lists(st.integers(min_value=0, max_value=255), min_size=16, max_size=16),
)
def test_legal_packets_round_trip(
    behavior: int,
    query_id: int,
    sender: int,
    payload: int,
    policy: int,
    t_ref: int,
    scale: int,
    offsets: list[int],
) -> None:
    packet = BCESPacket(
        1,
        0,
        behavior,
        1,
        query_id,
        sender,
        payload,
        policy,
        t_ref,
        1,
        16,
        255,
        255,
        scale,
        0,
        tuple(offsets),
    )
    assert decode_packet(encode_packet(packet)) == packet


@pytest.mark.parametrize("length", [0, 1, 47, 49, 96])
def test_malformed_lengths_are_rejected(length: int) -> None:
    with pytest.raises(MalformedPacketError):
        decode_packet(bytes(length))


def test_unsupported_version_is_rejected() -> None:
    packet, _ = golden()
    encoded = bytearray(encode_packet(packet))
    encoded[0] = 2
    with pytest.raises(UnsupportedVersionError):
        decode_packet(bytes(encoded))


@pytest.mark.parametrize(
    ("index", "value", "error"),
    [(1, 1, "flags"), (25, 15, "K"), (28, 0, "offset_scale"), (30, 1, "reserved")],
)
def test_illegal_header_fields_are_rejected(index: int, value: int, error: str) -> None:
    packet, _ = golden()
    encoded = bytearray(encode_packet(packet))
    if index == 28:
        encoded[28:30] = b"\x00\x00"
    elif index == 30:
        encoded[30:32] = b"\x00\x01"
    else:
        encoded[index] = value
    with pytest.raises(PacketError, match=error):
        decode_packet(bytes(encoded))


def test_quantization_boundaries_and_scale_guard() -> None:
    zero = quantize_offsets((0.0,) * 16)
    assert zero.offset_scale_q15 == 1
    assert zero.quantized_offsets == (0,) * 16
    values = tuple(index / 15 for index in range(16))
    quantized = quantize_offsets(values)
    assert quantized.offset_scale_q15 == 32768
    assert quantized.quantized_offsets[0] == 0
    assert quantized.quantized_offsets[-1] == 255
    assert (
        max(abs(a - b) for a, b in zip(values, quantized.dequantized_offsets))
        <= 1.0 / 510.0
    )
    with pytest.raises(ValueError, match="smaller"):
        quantize_offsets((1.0,) * 16, scale=0.5)
    with pytest.raises(ValueError):
        quantize_offsets((-0.1,) + (0.0,) * 15)
