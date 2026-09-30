from __future__ import annotations

import json

import pytest

from bces.models.sumo_dataset import SumoPairDataset


def _row(split: str) -> dict:
    batch = {
        "object_features": [[0.0] * 10 for _ in range(32)],
        "object_mask": [False] * 32, "reference_state": [0.0] * 7,
        "drift_scales": [1.0] * 7, "reference_path": [[0.0] * 6 for _ in range(12)],
        "identifiers": [0, 1, 123, 1, 1, 0], "object_count": 0,
        "occlusion_proxy": 0.0, "estimated_delay_s": 0.0, "map_context_flags": 0,
    }
    return {
        "pair_id": f"pair:{split}", "split": split, "state_equivalent": True, "valid": True,
        "reference": {"frozen_input": {"batch": batch}},
        "cached": {"ticks": [{"normalized_drift": [0.1] * 7}]},
    }


def test_sumo_pair_dataset_uses_one_final_decision_per_pair(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_row("train"), separators=(",", ":")) + "\n", encoding="utf-8")
    dataset = SumoPairDataset(path, "train")
    assert len(dataset) == 1
    assert dataset.summary["unit_of_inference"] == "same_world_branch_pair"
    assert dataset[0]["normalized_drift"].tolist() == pytest.approx([0.1] * 7)


def test_sumo_pair_dataset_locks_test(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(_row("test"), separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="locked"):
        SumoPairDataset(path, "test")

