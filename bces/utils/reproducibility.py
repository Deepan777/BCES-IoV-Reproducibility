"""Run-manifest and environment-fingerprint support."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .hashing import sha256_file

RUN_MANIFEST_KEYS = (
    "git_commit",
    "git_dirty",
    "python_environment_hash",
    "python_lock_hash",
    "dataset_allowlist_hash",
    "dataset_manifest_hash",
    "split_hash",
    "configuration_hash",
    "policy_hash",
    "model_checkpoint_hash",
    "calibration_hash",
    "seed",
    "gpu",
    "cuda",
    "driver",
    "pytorch",
    "workspace_bytes_before",
    "workspace_bytes_after",
    "cuda_peak_allocated_bytes",
    "cuda_peak_reserved_bytes",
    "start_time_utc",
    "end_time_utc",
    "exit_code",
    "status",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def git_state(workspace_root: Path) -> dict[str, Any]:
    git = shutil.which("git")
    if git is None:
        return {"commit": None, "dirty": None, "available": False}
    root = str(Path(workspace_root).resolve())
    commit_proc = subprocess.run(
        [git, "-C", root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    status_proc = subprocess.run(
        [git, "-C", root, "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit_proc.stdout.strip() if commit_proc.returncode == 0 else None,
        "dirty": bool(status_proc.stdout.strip())
        if status_proc.returncode == 0
        else None,
        "available": status_proc.returncode == 0,
    }


def optional_file_hash(path: Path) -> str | None:
    path = Path(path)
    return sha256_file(path) if path.is_file() else None


def new_run_manifest(**values: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {key: None for key in RUN_MANIFEST_KEYS}
    unknown = set(values) - set(manifest)
    if unknown:
        raise ValueError(f"unknown run-manifest fields: {sorted(unknown)}")
    manifest.update(values)
    manifest["schema_version"] = 1
    return manifest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
