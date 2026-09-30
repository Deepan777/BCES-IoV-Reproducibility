"""Binding-aware receiver state machine with anti-rollback packet caching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from bces.geometry.drift import KinematicState, compute_drift
from bces.geometry.surfaces import ExpirySurface

from .bindings import digest32, sender_id_hash
from .codec import decode_packet, dequantize_offsets
from .packet import BCESPacket, PacketError
from .query import Behavior, ReceiverQuery


class DecisionState(str, Enum):
    ACCEPT_REUSE = "ACCEPT_REUSE"
    REFRESH = "REFRESH"
    INVALID = "INVALID"


class EvidenceMode(str, Enum):
    COOPERATIVE_REUSE = "COOPERATIVE_REUSE"
    LOCAL_ONLY = "LOCAL_ONLY"


class CacheOffer(str, Enum):
    INSERTED = "INSERTED"
    REPLACED = "REPLACED"
    DUPLICATE = "DUPLICATE"
    STALE = "STALE"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class ReceiverDecision:
    state: DecisionState
    evidence_mode: EvidenceMode
    request_refresh: bool
    reason: str
    minimum_slack: float | None = None
    cache_offer: CacheOffer | None = None


def _serial_relation(candidate: int, current: int) -> int:
    difference = (candidate - current) & 0xFFFFFFFF
    if difference == 0:
        return 0
    return 1 if difference < 0x80000000 else -1


class PacketCache:
    """Keep the newest packet per sender without allowing duplicate/reordered rollback."""

    def __init__(self) -> None:
        self._entries: dict[int, BCESPacket] = {}

    def current(self, sender_hash: int) -> BCESPacket | None:
        return self._entries.get(sender_hash)

    def offer(self, packet: BCESPacket) -> CacheOffer:
        current = self._entries.get(packet.sender_id_hash)
        if current is None:
            self._entries[packet.sender_id_hash] = packet
            return CacheOffer.INSERTED
        query_relation = _serial_relation(packet.query_id, current.query_id)
        if query_relation > 0:
            self._entries[packet.sender_id_hash] = packet
            return CacheOffer.REPLACED
        if query_relation < 0:
            return CacheOffer.STALE
        time_relation = _serial_relation(packet.t_ref_ms, current.t_ref_ms)
        if time_relation > 0:
            self._entries[packet.sender_id_hash] = packet
            return CacheOffer.REPLACED
        if time_relation < 0:
            return CacheOffer.STALE
        return CacheOffer.DUPLICATE if packet == current else CacheOffer.CONFLICT


class ReceiverStateMachine:
    def __init__(
        self, *, refresh_guard_band: float, cache: PacketCache | None = None
    ) -> None:
        if not math.isfinite(refresh_guard_band) or refresh_guard_band < 0.0:
            raise ValueError("refresh_guard_band must be finite and non-negative")
        self.refresh_guard_band = float(refresh_guard_band)
        self.cache = cache or PacketCache()

    @staticmethod
    def _invalid(
        reason: str, cache_offer: CacheOffer | None = None
    ) -> ReceiverDecision:
        return ReceiverDecision(
            state=DecisionState.INVALID,
            evidence_mode=EvidenceMode.LOCAL_ONLY,
            request_refresh=False,
            reason=reason,
            cache_offer=cache_offer,
        )

    def evaluate(
        self,
        *,
        encoded_packet: bytes,
        cooperative_payload: bytes,
        query: ReceiverQuery,
        current_state: KinematicState,
        current_behavior: Behavior,
        current_policy_hash: int,
        actual_sender_id: str,
    ) -> ReceiverDecision:
        try:
            packet = decode_packet(encoded_packet)
        except (PacketError, TypeError) as exc:
            return self._invalid(
                f"malformed_or_unsupported_packet:{type(exc).__name__}"
            )

        try:
            actual_sender_hash = sender_id_hash(actual_sender_id)
            actual_payload_digest = digest32(cooperative_payload)
            runtime_behavior = Behavior(current_behavior)
        except (TypeError, ValueError) as exc:
            return self._invalid(f"invalid_runtime_context:{type(exc).__name__}")
        if (
            not isinstance(current_policy_hash, int)
            or not 0 <= current_policy_hash <= 0xFFFFFFFF
        ):
            return self._invalid("invalid_runtime_policy_hash")
        bindings = (
            (packet.query_id == query.query_id, "query_id_mismatch"),
            (
                packet.sender_id_hash == query.expected_sender_id_hash,
                "query_sender_mismatch",
            ),
            (packet.sender_id_hash == actual_sender_hash, "sender_mismatch"),
            (packet.payload_digest == actual_payload_digest, "payload_mismatch"),
            (packet.policy_hash == query.policy_hash, "query_policy_mismatch"),
            (packet.policy_hash == current_policy_hash, "current_policy_mismatch"),
            (packet.behavior_id == int(query.behavior), "query_behavior_mismatch"),
            (packet.behavior_id == int(runtime_behavior), "current_behavior_mismatch"),
            (packet.t_ref_ms == query.wire_t_ref_ms, "reference_time_mismatch"),
            (packet.drift_schema_id == query.drift_schema_id, "drift_schema_mismatch"),
            (
                packet.normal_codebook_id == query.normal_codebook_id,
                "codebook_mismatch",
            ),
            (packet.risk_class == query.risk_class, "risk_class_mismatch"),
            (packet.calibration_id == query.calibration_id, "calibration_mismatch"),
        )
        for matches, reason in bindings:
            if not matches:
                return self._invalid(reason)

        cache_offer = self.cache.offer(packet)
        if cache_offer in (CacheOffer.STALE, CacheOffer.CONFLICT):
            return self._invalid(f"cache_{cache_offer.value.lower()}", cache_offer)

        try:
            drift = compute_drift(
                query.reference_state, current_state, query.drift_scales
            )
            surface = ExpirySurface(
                offsets=dequantize_offsets(packet),
                codebook_id=packet.normal_codebook_id,
                calibration_id=packet.calibration_id,
            )
            slack = surface.minimum_slack(drift.normalized)
        except ValueError as exc:
            return self._invalid(
                f"drift_or_surface_error:{type(exc).__name__}", cache_offer
            )

        if slack < 0.0:
            return ReceiverDecision(
                state=DecisionState.INVALID,
                evidence_mode=EvidenceMode.LOCAL_ONLY,
                request_refresh=True,
                reason="surface_violated",
                minimum_slack=slack,
                cache_offer=cache_offer,
            )
        if slack < self.refresh_guard_band:
            return ReceiverDecision(
                state=DecisionState.REFRESH,
                evidence_mode=EvidenceMode.LOCAL_ONLY,
                request_refresh=True,
                reason="inside_refresh_guard_band",
                minimum_slack=slack,
                cache_offer=cache_offer,
            )
        return ReceiverDecision(
            state=DecisionState.ACCEPT_REUSE,
            evidence_mode=EvidenceMode.COOPERATIVE_REUSE,
            request_refresh=False,
            reason="inside_surface",
            minimum_slack=slack,
            cache_offer=cache_offer,
        )
