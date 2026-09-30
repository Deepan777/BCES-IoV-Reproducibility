"""Deterministic 32-bit bindings for sender, payload, policy, and query context."""

from __future__ import annotations

import hashlib
from typing import Any

from bces.utils.hashing import canonical_json_hash


def digest32(payload: bytes) -> int:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def sender_id_hash(sender_id: str) -> int:
    if not isinstance(sender_id, str) or not sender_id:
        raise ValueError("sender_id must be a non-empty string")
    return digest32(sender_id.encode("utf-8"))


def policy_hash32(policy_definition: Any) -> int:
    canonical_digest = canonical_json_hash(policy_definition)
    return int(canonical_digest[:8], 16)
