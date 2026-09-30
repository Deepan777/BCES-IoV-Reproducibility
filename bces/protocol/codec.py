"""Network-byte-order codec and Q15/uint8 surface quantization."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass

from .packet import (
    OFFSET_COUNT_V1,
    PACKET_BYTES_V1,
    BCESPacket,
    MalformedPacketError,
)

PACKET_STRUCT_V1 = struct.Struct("!BBBBIIIIIBBBBHH16B")
Q15_DENOMINATOR = 32768.0

if PACKET_STRUCT_V1.size != PACKET_BYTES_V1:  # deterministic import-time invariant
    raise RuntimeError("version-1 packet struct is not exactly 48 bytes")


@dataclass(frozen=True)
class QuantizedSurface:
    offset_scale_q15: int
    quantized_offsets: tuple[int, ...]
    dequantized_offsets: tuple[float, ...]

    @property
    def scale(self) -> float:
        return self.offset_scale_q15 / Q15_DENOMINATOR


def _round_nonnegative(value: float) -> int:
    return math.floor(value + 0.5)


def quantize_offsets(
    offsets: Iterable[float], *, scale: float | None = None
) -> QuantizedSurface:
    values = tuple(float(value) for value in offsets)
    if len(values) != OFFSET_COUNT_V1:
        raise ValueError(f"exactly {OFFSET_COUNT_V1} offsets are required")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("offsets must be finite and non-negative")
    maximum = max(values)
    if scale is None:
        scale_q15 = max(1, math.ceil(maximum * Q15_DENOMINATOR))
    else:
        scale_value = float(scale)
        if not math.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("scale must be finite and positive")
        scale_q15 = _round_nonnegative(scale_value * Q15_DENOMINATOR)
    if not 1 <= scale_q15 <= 0xFFFF:
        raise ValueError("offset scale is outside unsigned Q15 range")
    decoded_scale = scale_q15 / Q15_DENOMINATOR
    if maximum > decoded_scale:
        raise ValueError("offset scale is smaller than the largest offset")
    quantized = tuple(
        min(255, max(0, _round_nonnegative(255.0 * value / decoded_scale)))
        for value in values
    )
    dequantized = tuple(decoded_scale * value / 255.0 for value in quantized)
    return QuantizedSurface(
        offset_scale_q15=scale_q15,
        quantized_offsets=quantized,
        dequantized_offsets=dequantized,
    )


def dequantize_offsets(packet: BCESPacket) -> tuple[float, ...]:
    scale = packet.offset_scale_q15 / Q15_DENOMINATOR
    return tuple(scale * value / 255.0 for value in packet.quantized_offsets)


def encode_packet(packet: BCESPacket) -> bytes:
    encoded = PACKET_STRUCT_V1.pack(
        packet.protocol_version,
        packet.flags,
        packet.behavior_id,
        packet.drift_schema_id,
        packet.query_id,
        packet.sender_id_hash,
        packet.payload_digest,
        packet.policy_hash,
        packet.t_ref_ms,
        packet.normal_codebook_id,
        packet.k,
        packet.risk_class,
        packet.calibration_id,
        packet.offset_scale_q15,
        packet.reserved,
        *packet.quantized_offsets,
    )
    if len(encoded) != PACKET_BYTES_V1:
        raise RuntimeError("encoded packet length invariant failed")
    return encoded


def decode_packet(payload: bytes) -> BCESPacket:
    if not isinstance(payload, bytes):
        raise TypeError("encoded packet must be bytes")
    if len(payload) != PACKET_BYTES_V1:
        raise MalformedPacketError(
            f"packet length {len(payload)} does not equal {PACKET_BYTES_V1}"
        )
    try:
        unpacked = PACKET_STRUCT_V1.unpack(payload)
    except struct.error as exc:  # defensive; length check should make this unreachable
        raise MalformedPacketError(str(exc)) from exc
    return BCESPacket(
        protocol_version=unpacked[0],
        flags=unpacked[1],
        behavior_id=unpacked[2],
        drift_schema_id=unpacked[3],
        query_id=unpacked[4],
        sender_id_hash=unpacked[5],
        payload_digest=unpacked[6],
        policy_hash=unpacked[7],
        t_ref_ms=unpacked[8],
        normal_codebook_id=unpacked[9],
        k=unpacked[10],
        risk_class=unpacked[11],
        calibration_id=unpacked[12],
        offset_scale_q15=unpacked[13],
        reserved=unpacked[14],
        quantized_offsets=tuple(unpacked[15:]),
    )


def packet_from_offsets(
    *,
    behavior_id: int,
    query_id: int,
    sender_id_hash: int,
    payload_digest: int,
    policy_hash: int,
    t_ref_ms: int,
    risk_class: int,
    calibration_id: int,
    offsets: Iterable[float],
) -> BCESPacket:
    quantized = quantize_offsets(offsets)
    return BCESPacket(
        protocol_version=1,
        flags=0,
        behavior_id=behavior_id,
        drift_schema_id=1,
        query_id=query_id,
        sender_id_hash=sender_id_hash,
        payload_digest=payload_digest,
        policy_hash=policy_hash,
        t_ref_ms=t_ref_ms,
        normal_codebook_id=1,
        k=16,
        risk_class=risk_class,
        calibration_id=calibration_id,
        offset_scale_q15=quantized.offset_scale_q15,
        reserved=0,
        quantized_offsets=quantized.quantized_offsets,
    )
