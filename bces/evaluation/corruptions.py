"""Deterministic cooperative-track corruptions for robustness evaluation."""

from __future__ import annotations

import hashlib
import math

import torch


def _selected(reference_id: str, name: str, fraction: float) -> bool:
    digest = hashlib.sha256(f"{reference_id}:{name}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def apply_corruption(
    item: dict[str, torch.Tensor],
    reference_id: str,
    name: str,
    magnitude: float,
) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in item.items()}
    features = result["object_features"]
    mask = result["object_mask"].bool()
    if name == "position_error_m":
        features[mask, 0] += magnitude
    elif name == "velocity_error_mps":
        features[mask, 2] += magnitude
    elif name == "heading_error_deg":
        angle = math.radians(magnitude)
        vx, vy = features[mask, 2].clone(), features[mask, 3].clone()
        features[mask, 2] = math.cos(angle) * vx - math.sin(angle) * vy
        features[mask, 3] = math.sin(angle) * vx + math.cos(angle) * vy
    elif name == "confidence_drop":
        features[mask, 7] = (features[mask, 7] - magnitude).clamp_min(0.0)
    elif name == "missed_object_fraction":
        if mask.any() and _selected(reference_id, name, magnitude):
            distances = torch.linalg.vector_norm(features[:, :2], dim=1)
            distances = distances.masked_fill(~mask, torch.inf)
            index = int(distances.argmin())
            features[index] = 0
            result["object_mask"][index] = False
            result["object_count"] -= 1
    elif name == "spurious_object_fraction":
        if (~mask).any() and _selected(reference_id, name, magnitude):
            index = int(torch.where(~mask)[0][0])
            features[index] = torch.tensor(
                [10.0, 1.0, 2.0, 0.0, 4.5, 1.8, 0.0, 0.5, 2.0, 1.0],
                dtype=features.dtype,
            )
            result["object_mask"][index] = True
            result["object_count"] += 1
    elif name == "timestamp_error_ms":
        result["normalized_drift"][6] += (
            magnitude / 1000.0 / result["drift_scales"][6]
        )
    else:
        raise ValueError(f"unknown corruption: {name}")
    return result
