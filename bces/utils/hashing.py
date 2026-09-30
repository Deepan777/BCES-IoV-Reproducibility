"""Deterministic hashing helpers used by reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    """Hash JSON-compatible data using a stable canonical representation."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

