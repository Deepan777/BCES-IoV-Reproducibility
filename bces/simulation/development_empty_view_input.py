"""Fail-closed decoded-view input for an unregistered development planner.

The premise is an explicit *simulation assumption*, not a certificate that a
physical sensor saw every actor. No truth or future trajectory is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from bces.data.objects import ObjectMessage
from bces.simulation.controlled_contract import payload_objects
from bces.simulation.development_negative_evidence_planner import VerifiedEmptyObservation


@dataclass(frozen=True)
class DevelopmentEmptyViewPremise:
    """The exact decoded empty packet to which an ideal-view assumption applies."""

    payload_sha256: str
    sender_id: str
    timestamp_ms: int
    view_id: str
    complete_for_conflict_region: bool

    def __post_init__(self) -> None:
        if len(self.payload_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.payload_sha256
        ):
            raise ValueError("payload_sha256 must be a lowercase SHA-256 digest")
        if not self.sender_id or not self.view_id:
            raise ValueError("source and view identifiers are required")
        if type(self.timestamp_ms) is not int or self.timestamp_ms < 0:
            raise ValueError("premise timestamp must be a nonnegative integer")
        if type(self.complete_for_conflict_region) is not bool:
            raise ValueError("view completeness must be an explicit boolean")


@dataclass(frozen=True)
class DecodedPolicyInput:
    objects: tuple
    status: str
    payload_sha256: str | None
    view_id: str | None
    age_s: float | None


def decode_policy_input(
    message: ObjectMessage | None,
    *,
    now_ms: int,
    empty_premise: DevelopmentEmptyViewPremise | None = None,
) -> DecodedPolicyInput:
    """Distinguish no packet, untrusted empty, trusted synthetic empty, and objects.

    An empty packet is *not* assumed complete merely because its object list is
    empty. A matching premise permits a marker, whose age is later interpreted
    by the candidate planner. Any mismatched premise raises rather than silently
    blessing another packet.
    """
    if type(now_ms) is not int or now_ms < 0:
        raise ValueError("receiver time must be a nonnegative integer")
    if message is None:
        if empty_premise is not None:
            raise ValueError("an absent packet cannot have an empty-view premise")
        return DecodedPolicyInput((), "no_message", None, None, None)
    if now_ms < message.timestamp_ms:
        raise ValueError("packet timestamp is later than receiver time")
    digest = hashlib.sha256(message.encode()).hexdigest()
    age_s = (now_ms - message.timestamp_ms) / 1000.0
    if message.objects:
        if empty_premise is not None:
            raise ValueError("a nonempty packet cannot carry an empty-view premise")
        return DecodedPolicyInput(payload_objects(message), "objects", digest, None, age_s)
    if empty_premise is None:
        return DecodedPolicyInput((), "empty_unverified", digest, None, age_s)
    if (empty_premise.payload_sha256 != digest
            or empty_premise.sender_id != message.sender_id
            or empty_premise.timestamp_ms != message.timestamp_ms):
        raise ValueError("empty-view premise does not bind this exact packet")
    if not empty_premise.complete_for_conflict_region:
        return DecodedPolicyInput((), "empty_incomplete", digest, empty_premise.view_id, age_s)
    return DecodedPolicyInput(
        (VerifiedEmptyObservation(age_s),), "empty_complete_assumed",
        digest, empty_premise.view_id, age_s,
    )
