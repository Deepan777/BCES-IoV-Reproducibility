from __future__ import annotations

import torch

from bces.evaluation.corruptions import apply_corruption


def _item() -> dict[str, torch.Tensor]:
    features = torch.zeros(32, 10)
    features[0] = torch.tensor([1, 2, 3, 4, 5, 2, 0, 0.9, 1, 1])
    mask = torch.zeros(32, dtype=torch.bool)
    mask[0] = True
    return {
        "object_features": features,
        "object_mask": mask,
        "object_count": torch.tensor(1.0),
        "normalized_drift": torch.zeros(7),
        "drift_scales": torch.ones(7),
    }


def test_timestamp_and_position_corruptions_do_not_mutate_source() -> None:
    source = _item()
    position = apply_corruption(source, "r1", "position_error_m", 1.0)
    timestamp = apply_corruption(source, "r1", "timestamp_error_ms", 50.0)
    assert source["object_features"][0, 0] == 1.0
    assert position["object_features"][0, 0] == 2.0
    assert timestamp["normalized_drift"][6] == 0.05


def test_missing_and_spurious_object_counts_follow_masks() -> None:
    missing = apply_corruption(
        _item(), "selected", "missed_object_fraction", 1.0
    )
    spurious = apply_corruption(
        _item(), "selected", "spurious_object_fraction", 1.0
    )
    assert int(missing["object_count"]) == int(missing["object_mask"].sum()) == 0
    assert int(spurious["object_count"]) == int(spurious["object_mask"].sum()) == 2
