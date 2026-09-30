"""Scenario-level dataset for the compact same-world SUMO branch pairs."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SumoPairDataset(Dataset[dict[str, torch.Tensor]]):
    """Use one final-horizon validity decision per independent SUMO pair."""

    def __init__(self, path: Path, split: str, *, allow_test: bool = False) -> None:
        if split == "test" and not allow_test:
            raise PermissionError("final SUMO test partition is locked")
        if split not in {"train", "validation", "calibration", "test"}:
            raise ValueError("unknown split")
        rows = []
        marker = f'"split":"{split}"'
        with Path(path).open("rt", encoding="utf-8") as handle:
            for line in handle:
                if marker in line:
                    row = json.loads(line)
                    if not row["state_equivalent"]:
                        raise ValueError("non-equivalent SUMO branch pair")
                    rows.append(row)
        if not rows:
            raise ValueError(f"no SUMO pairs for {split}")
        batches = [row["reference"]["frozen_input"]["batch"] for row in rows]
        self.reference_tensors = {
            "object_features": torch.tensor([row["object_features"] for row in batches], dtype=torch.float32),
            "object_mask": torch.tensor([row["object_mask"] for row in batches], dtype=torch.bool),
            "reference_state": torch.tensor([row["reference_state"] for row in batches], dtype=torch.float32),
            "drift_scales": torch.tensor([row["drift_scales"] for row in batches], dtype=torch.float32),
            "reference_path": torch.tensor([row["reference_path"] for row in batches], dtype=torch.float32),
            "identifiers": torch.tensor([row["identifiers"] for row in batches], dtype=torch.float32),
            "object_count": torch.tensor([row["object_count"] for row in batches], dtype=torch.float32),
            "occlusion_proxy": torch.tensor([row["occlusion_proxy"] for row in batches], dtype=torch.float32),
            "estimated_delay_s": torch.tensor([row["estimated_delay_s"] for row in batches], dtype=torch.float32),
            "map_context_flags": torch.tensor([row["map_context_flags"] for row in batches], dtype=torch.float32),
        }
        self.normalized_drift = torch.tensor(
            [row["cached"]["ticks"][-1]["normalized_drift"] for row in rows], dtype=torch.float32
        )
        self.valid = torch.tensor([row["valid"] for row in rows], dtype=torch.float32)
        self.scenario_ids = tuple(str(row["pair_id"]) for row in rows)
        self.rows = rows
        self.summary = {
            "split": split, "pair_count": len(rows),
            "valid_count": int(self.valid.sum()), "invalid_count": int(len(rows) - self.valid.sum()),
            "unit_of_inference": "same_world_branch_pair", "other_split_records_deserialized": 0,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: values[index] for name, values in self.reference_tensors.items()}
        item.update({
            "normalized_drift": self.normalized_drift[index], "valid": self.valid[index],
            "boundary_targets": torch.zeros(16), "boundary_mask": torch.zeros(16, dtype=torch.bool),
        })
        return item

