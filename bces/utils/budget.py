"""Workspace and CUDA resource-budget enforcement for BCES-IoV."""

from __future__ import annotations

import json
import os
import shutil
import stat
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class BudgetError(RuntimeError):
    """Base class for an enforced resource-budget failure."""


class BudgetExceeded(BudgetError):
    """Raised before a workspace, reserve, tranche, disk, or CUDA cap is crossed."""


class DatasetSymlinkError(BudgetError):
    """Raised when dataset content is redirected outside the data root."""


class BudgetWarning(UserWarning):
    """Warning emitted when the registered workspace warning level is reached."""


@dataclass(frozen=True)
class Budget:
    data_root: Path
    workspace_root: Path
    initial_data_review_bytes: int = 500_000_000
    max_download_tranche_bytes: int = 250_000_000
    required_output_reserve_bytes: int = 1_500_000_000
    warn_workspace_bytes: int = 8_500_000_000
    max_workspace_bytes: int = 10_000_000_000
    max_cuda_allocated_bytes: int = 5_500_000_000

    def __post_init__(self) -> None:
        integer_fields = (
            "initial_data_review_bytes",
            "max_download_tranche_bytes",
            "required_output_reserve_bytes",
            "warn_workspace_bytes",
            "max_workspace_bytes",
            "max_cuda_allocated_bytes",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.required_output_reserve_bytes >= self.max_workspace_bytes:
            raise ValueError("required output reserve must be below the workspace cap")
        if self.warn_workspace_bytes >= self.max_workspace_bytes:
            raise ValueError("warning threshold must be below the workspace cap")
        if self.max_download_tranche_bytes > self.max_workspace_bytes:
            raise ValueError("download tranche cannot exceed the workspace cap")
        object.__setattr__(self, "workspace_root", Path(self.workspace_root).resolve())
        object.__setattr__(self, "data_root", Path(self.data_root).resolve())
        if not _is_within(self.data_root, self.workspace_root):
            raise ValueError("data_root must be inside workspace_root")


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _is_link_or_reparse(entry: os.DirEntry[str], entry_stat: os.stat_result) -> bool:
    if entry.is_symlink():
        return True
    attributes = getattr(entry_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & reparse_flag)


def _scan_entries(root: Path) -> Iterator[tuple[Path, os.stat_result, bool]]:
    root = Path(root)
    if not root.exists() and not root.is_symlink():
        return
    stack = [root]
    while stack:
        current = stack.pop()
        if current.is_file() and not current.is_symlink():
            yield current, current.stat(), False
            continue
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError):
            continue
        for entry in entries:
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            is_link = _is_link_or_reparse(entry, entry_stat)
            path = Path(entry.path)
            yield path, entry_stat, is_link
            if not is_link and stat.S_ISDIR(entry_stat.st_mode):
                stack.append(path)


def tree_bytes(root: Path) -> int:
    """Count every regular file under *root* without following links/reparse points."""
    total = 0
    for _path, entry_stat, is_link in _scan_entries(Path(root)):
        if not is_link and stat.S_ISREG(entry_stat.st_mode):
            total += entry_stat.st_size
    return total


def validate_dataset_symlinks(budget: Budget) -> None:
    """Reject any dataset link or junction whose target escapes ``data_root``."""
    if not budget.data_root.exists():
        return
    for path, _entry_stat, is_link in _scan_entries(budget.data_root):
        if not is_link:
            continue
        try:
            target = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise DatasetSymlinkError(f"broken or unresolved dataset link: {path}") from exc
        if not _is_within(target, budget.data_root):
            raise DatasetSymlinkError(
                f"dataset link escapes data root: {path} -> {target}"
            )


def available_download_bytes(budget: Budget) -> int:
    """Return measured safe headroom while preserving the required output reserve."""
    validate_dataset_symlinks(budget)
    workspace_used = tree_bytes(budget.workspace_root)
    workspace_headroom = max(
        0,
        budget.max_workspace_bytes
        - budget.required_output_reserve_bytes
        - workspace_used,
    )
    disk_free = shutil.disk_usage(budget.workspace_root).free
    disk_headroom = max(0, disk_free - budget.required_output_reserve_bytes)
    return min(workspace_headroom, disk_headroom)


def require_headroom(
    budget: Budget,
    incoming_bytes: int = 0,
    preserve_reserve: bool = True,
    *,
    enforce_tranche: bool = True,
) -> None:
    """Fail before an incoming operation can cross any registered resource cap."""
    if not isinstance(incoming_bytes, int) or incoming_bytes < 0:
        raise ValueError("incoming_bytes must be a non-negative integer")
    if enforce_tranche and incoming_bytes > budget.max_download_tranche_bytes:
        raise BudgetExceeded(
            f"incoming bytes {incoming_bytes} exceed the "
            f"{budget.max_download_tranche_bytes}-byte tranche cap"
        )
    validate_dataset_symlinks(budget)
    current = tree_bytes(budget.workspace_root)
    projected = current + incoming_bytes
    if projected >= budget.max_workspace_bytes:
        raise BudgetExceeded(
            f"projected workspace {projected} reaches/exceeds cap "
            f"{budget.max_workspace_bytes}"
        )
    if preserve_reserve:
        safe_ceiling = budget.max_workspace_bytes - budget.required_output_reserve_bytes
        if projected > safe_ceiling:
            raise BudgetExceeded(
                f"projected workspace {projected} would violate the "
                f"{budget.required_output_reserve_bytes}-byte output reserve"
            )
    disk_free = shutil.disk_usage(budget.workspace_root).free
    required_disk = incoming_bytes + (
        budget.required_output_reserve_bytes if preserve_reserve else 0
    )
    if disk_free < required_disk:
        raise BudgetExceeded(
            f"disk free bytes {disk_free} are below required bytes {required_disk}"
        )
    if projected >= budget.warn_workspace_bytes:
        warnings.warn(
            f"workspace warning threshold reached: projected={projected}, "
            f"warning={budget.warn_workspace_bytes}",
            BudgetWarning,
            stacklevel=2,
        )


def assert_cuda_peak(budget: Budget, *, peak_allocated_bytes: int | None = None) -> None:
    """Raise if the measured PyTorch peak allocation exceeds the registered cap."""
    if peak_allocated_bytes is None:
        try:
            import torch
        except ImportError as exc:
            raise BudgetExceeded("PyTorch is unavailable for CUDA peak verification") from exc
        if not torch.cuda.is_available():
            raise BudgetExceeded("CUDA is unavailable for peak-memory verification")
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    if peak_allocated_bytes < 0:
        raise ValueError("peak_allocated_bytes cannot be negative")
    if peak_allocated_bytes > budget.max_cuda_allocated_bytes:
        raise BudgetExceeded(
            f"CUDA peak {peak_allocated_bytes} exceeds cap "
            f"{budget.max_cuda_allocated_bytes}"
        )


def build_budget_report(budget: Budget) -> dict[str, Any]:
    """Create a machine-readable report from measurements taken at call time."""
    validate_dataset_symlinks(budget)
    workspace_bytes = tree_bytes(budget.workspace_root)
    data_bytes = tree_bytes(budget.data_root)
    disk = shutil.disk_usage(budget.workspace_root)
    safe_headroom = available_download_bytes(budget)
    limits = asdict(budget)
    limits["workspace_root"] = str(budget.workspace_root)
    limits["data_root"] = str(budget.data_root)
    status = "warning" if workspace_bytes >= budget.warn_workspace_bytes else "pass"
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "measurements": {
            "workspace_bytes": workspace_bytes,
            "data_bytes": data_bytes,
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "safe_headroom_bytes": safe_headroom,
            "next_tranche_allowed_bytes": min(
                safe_headroom, budget.max_download_tranche_bytes
            ),
            "output_reserve_preserved": (
                workspace_bytes + budget.required_output_reserve_bytes
                <= budget.max_workspace_bytes
                and disk.free >= budget.required_output_reserve_bytes
            ),
            "initial_data_review_reached": (
                data_bytes >= budget.initial_data_review_bytes
            ),
        },
        "limits": limits,
        "dataset_external_links_rejected": True,
        "counting_policy": {
            "hidden_files": True,
            "partial_files": True,
            "cache_files": True,
            "temporary_files": True,
            "follow_links_or_reparse_points": False,
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_budget_report(path: Path, budget: Budget) -> dict[str, Any]:
    """Measure, atomically write, and return a budget report."""
    report = build_budget_report(budget)
    _atomic_write_json(Path(path), report)
    return report

