"""Frozen public normal codebooks for receiver-checkable expiry surfaces."""

from __future__ import annotations

import math
from typing import Final

from bces.utils.hashing import canonical_json_hash

from .drift import DRIFT_DIMENSION

NORMAL_CODEBOOK_ID_V1: Final[int] = 1
NORMAL_COUNT_V1: Final[int] = 16


def _build_v1() -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for axis in range(DRIFT_DIMENSION):
        positive = [0.0] * DRIFT_DIMENSION
        negative = [0.0] * DRIFT_DIMENSION
        positive[axis] = 1.0
        negative[axis] = -1.0
        rows.extend((tuple(positive), tuple(negative)))
    diagonal = tuple(1.0 / math.sqrt(DRIFT_DIMENSION) for _ in range(DRIFT_DIMENSION))
    rows.extend((diagonal, tuple(-value for value in diagonal)))
    return tuple(rows)


NORMAL_CODEBOOK_V1: Final[tuple[tuple[float, ...], ...]] = _build_v1()


def get_normal_codebook(codebook_id: int) -> tuple[tuple[float, ...], ...]:
    if codebook_id != NORMAL_CODEBOOK_ID_V1:
        raise ValueError(f"unsupported normal codebook id: {codebook_id}")
    return NORMAL_CODEBOOK_V1


def codebook_sha256(codebook_id: int = NORMAL_CODEBOOK_ID_V1) -> str:
    return canonical_json_hash(
        {"codebook_id": codebook_id, "normals": get_normal_codebook(codebook_id)}
    )
