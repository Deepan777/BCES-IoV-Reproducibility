"""Strict typed representation of the exact version-1 48-byte extension."""

from __future__ import annotations

from dataclasses import dataclass

from .query import PROTOCOL_VERSION_V1, Behavior

PACKET_BYTES_V1 = 48
OFFSET_COUNT_V1 = 16
SUPPORTED_FLAGS_V1 = 0


class PacketError(ValueError):
    pass


class MalformedPacketError(PacketError):
    pass


class UnsupportedVersionError(PacketError):
    pass


def _require_uint(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise PacketError(f"{name} must be uint{bits}")


@dataclass(frozen=True)
class BCESPacket:
    protocol_version: int
    flags: int
    behavior_id: int
    drift_schema_id: int
    query_id: int
    sender_id_hash: int
    payload_digest: int
    policy_hash: int
    t_ref_ms: int
    normal_codebook_id: int
    k: int
    risk_class: int
    calibration_id: int
    offset_scale_q15: int
    reserved: int
    quantized_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "protocol_version",
            "flags",
            "behavior_id",
            "drift_schema_id",
            "normal_codebook_id",
            "k",
            "risk_class",
            "calibration_id",
        ):
            _require_uint(name, getattr(self, name), 8)
        for name in (
            "query_id",
            "sender_id_hash",
            "payload_digest",
            "policy_hash",
            "t_ref_ms",
        ):
            _require_uint(name, getattr(self, name), 32)
        _require_uint("offset_scale_q15", self.offset_scale_q15, 16)
        _require_uint("reserved", self.reserved, 16)
        if self.protocol_version != PROTOCOL_VERSION_V1:
            raise UnsupportedVersionError(
                f"unsupported protocol version {self.protocol_version}"
            )
        if self.flags != SUPPORTED_FLAGS_V1:
            raise PacketError("version-1 flags must be zero")
        try:
            Behavior(self.behavior_id)
        except ValueError as exc:
            raise PacketError("unsupported behavior_id") from exc
        if self.drift_schema_id != 1:
            raise PacketError("unsupported drift_schema_id")
        if self.normal_codebook_id != 1:
            raise PacketError("unsupported normal_codebook_id")
        if self.k != OFFSET_COUNT_V1:
            raise PacketError(f"K must equal {OFFSET_COUNT_V1}")
        if self.offset_scale_q15 == 0:
            raise PacketError("offset_scale_q15 must be positive")
        if self.reserved != 0:
            raise PacketError("reserved field must be zero in version 1")
        offsets = tuple(self.quantized_offsets)
        if len(offsets) != OFFSET_COUNT_V1:
            raise PacketError(
                f"quantized_offsets must contain {OFFSET_COUNT_V1} values"
            )
        for index, value in enumerate(offsets):
            _require_uint(f"quantized_offsets[{index}]", value, 8)
        object.__setattr__(self, "quantized_offsets", offsets)
