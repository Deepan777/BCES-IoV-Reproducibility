"""Integrity helpers for generated publication artifacts."""

from __future__ import annotations

from pathlib import Path

from bces.utils.hashing import sha256_file

PROHIBITED_CLAIM_PHRASES = (
    "world's first", "guaranteed safe", "certified safe", "formally safe",
    "provably safe", "universally valid", "real-road certified",
)


def prohibited_claims(text: str) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(phrase for phrase in PROHIBITED_CLAIM_PHRASES if phrase in lowered)


def artifact_hashes(root: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    root = Path(root)
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in set(exclude)
    }

