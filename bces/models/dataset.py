"""Leakage-guarded loading of corrected Phase-4 SurfaceNet examples."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset


def _records_for_split(path: Path, split: str) -> Iterator[dict]:
    """Deserialize only matching compact JSONL records from a combined artifact."""
    marker = f'"split":"{split}"'
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if marker in line:
                yield json.loads(line)


class SurfaceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, artifact_root: Path, split: str, *, allow_test: bool = False) -> None:
        if split == "test" and not allow_test:
            raise PermissionError("final test partition is locked")
        if split not in {"train", "validation", "calibration", "test"}:
            raise ValueError("unknown split")
        root = Path(artifact_root)
        references = list(_records_for_split(root / "references.jsonl.gz", split))
        if not references:
            raise ValueError(f"no reference records for {split}")
        reference_index = {row["reference_id"]: index for index, row in enumerate(references)}
        if len(reference_index) != len(references):
            raise ValueError("duplicate reference IDs")
        targets = torch.zeros((len(references), 16), dtype=torch.float32)
        masks = torch.zeros((len(references), 16), dtype=torch.bool)
        for row in _records_for_split(root / "boundaries.jsonl.gz", split):
            if row["status"] != "transition":
                continue
            index = reference_index.get(row["reference_id"])
            if index is None:
                raise ValueError("boundary references an unknown input")
            direction = int(row["direction_index"])
            targets[index, direction] = float(row["valid_radius"])
            masks[index, direction] = True
        points = list(_records_for_split(root / "validity_points.jsonl.gz", split))
        if not points:
            raise ValueError(f"no validity points for {split}")
        self.split = split
        self.reference_ids = tuple(row["reference_id"] for row in references)
        self.reference_indices = torch.tensor(
            [reference_index[row["reference_id"]] for row in points], dtype=torch.long
        )
        self.normalized_drift = torch.tensor(
            [row["normalized_drift"] for row in points], dtype=torch.float32
        )
        self.valid = torch.tensor([row["valid"] for row in points], dtype=torch.float32)
        self.scenario_ids = tuple(str(row["scenario_id"]) for row in points)
        self.boundary_targets = targets
        self.boundary_mask = masks
        batches = [row["frozen_input"]["batch"] for row in references]
        self.reference_tensors = {
            "object_features": torch.tensor([row["object_features"] for row in batches], dtype=torch.float32),
            "object_mask": torch.tensor([row["object_mask"] for row in batches], dtype=torch.bool),
            "reference_state": torch.tensor([row["reference_state"] for row in batches], dtype=torch.float64).float(),
            "drift_scales": torch.tensor([row["drift_scales"] for row in batches], dtype=torch.float32),
            "reference_path": torch.tensor([row["reference_path"] for row in batches], dtype=torch.float32),
            "identifiers": torch.tensor([row["identifiers"] for row in batches], dtype=torch.float64).float(),
            "object_count": torch.tensor([row["object_count"] for row in batches], dtype=torch.float32),
            "occlusion_proxy": torch.tensor([row["occlusion_proxy"] for row in batches], dtype=torch.float32),
            "estimated_delay_s": torch.tensor([row["estimated_delay_s"] for row in batches], dtype=torch.float32),
            "map_context_flags": torch.tensor([row["map_context_flags"] for row in batches], dtype=torch.float32),
        }
        self.summary = {
            "split": split,
            "reference_count": len(references),
            "point_count": len(points),
            "valid_count": int(self.valid.sum().item()),
            "invalid_count": int(len(points) - self.valid.sum().item()),
            "measured_boundary_count": int(masks.sum().item()),
            "other_split_records_deserialized": 0,
        }

    def __len__(self) -> int:
        return len(self.reference_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        reference = self.reference_indices[index]
        item = {name: values[reference] for name, values in self.reference_tensors.items()}
        item.update(
            {
                "normalized_drift": self.normalized_drift[index],
                "valid": self.valid[index],
                "boundary_targets": self.boundary_targets[reference],
                "boundary_mask": self.boundary_mask[reference],
            }
        )
        return item
