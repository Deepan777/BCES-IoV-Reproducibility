"""Dataset-specific validation dispatch; schemas are never merged implicitly."""

from __future__ import annotations

from pathlib import Path

from .datasets import dataset_profile
from .manifest import CatalogEntry
from .tfd import ValidationResult, validate_downloaded_file as validate_tfd_file


def validate_dataset_file(
    path: Path, entry: CatalogEntry, *, dataset_id: str
) -> ValidationResult:
    adapter_id = dataset_profile(dataset_id).adapter_id
    if adapter_id in {"v2xtraj", "v2xseq_tfd"}:
        return validate_tfd_file(path, entry)
    # External/sensitivity adapters are deliberately fail-closed until a reviewed,
    # selective official file is allowlisted and its concrete schema is frozen.
    raise ValueError(
        f"adapter {adapter_id} is not activated without a reviewed schema fixture"
    )
