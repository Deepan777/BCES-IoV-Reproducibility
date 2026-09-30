#!/usr/bin/env python3
"""Generate the actual-machine Phase-0 environment and run-manifest reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bces.utils.budget import Budget, tree_bytes  # noqa: E402
from bces.utils.environment import collect_environment_report, print_json  # noqa: E402
from bces.utils.hashing import canonical_json_hash  # noqa: E402
from bces.utils.reproducibility import (  # noqa: E402
    git_state,
    new_run_manifest,
    optional_file_hash,
    utc_now,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "budget.yaml")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    workspace = args.workspace_root.resolve()
    with args.config.open("r", encoding="utf-8") as handle:
        budget_values = yaml.safe_load(handle) or {}
    budget = Budget(
        workspace_root=workspace,
        data_root=workspace / "data",
        **budget_values,
    )
    report_path = args.report or workspace / "outputs" / "phase0" / "environment_report.json"
    manifest_path = args.run_manifest or workspace / "outputs" / "phase0" / "run_manifest_environment.json"
    start_time = utc_now()
    workspace_before = tree_bytes(workspace)
    report = collect_environment_report(workspace, budget)
    report["timestamp_utc"] = utc_now()
    write_json_atomic(report_path, report)
    workspace_after = tree_bytes(workspace)
    git = git_state(workspace)
    manifest = new_run_manifest(
        git_commit=git["commit"],
        git_dirty=git["dirty"],
        python_environment_hash=report["python"]["environment_hash"],
        python_lock_hash=optional_file_hash(workspace / "uv.lock"),
        dataset_allowlist_hash=optional_file_hash(
            workspace / "configs" / "data" / "compact_tfd_allowlist.yaml"
        ),
        dataset_manifest_hash=None,
        split_hash=None,
        configuration_hash=canonical_json_hash(budget_values),
        policy_hash=None,
        model_checkpoint_hash=None,
        calibration_hash=None,
        seed=None,
        gpu=report["pytorch"].get("device_names"),
        cuda=report["pytorch"].get("cuda_build"),
        driver=(report["gpu"]["devices"][0]["driver_version"] if report["gpu"]["devices"] else None),
        pytorch=report["pytorch"].get("version"),
        workspace_bytes_before=workspace_before,
        workspace_bytes_after=workspace_after,
        cuda_peak_allocated_bytes=report["cuda_memory_probe"].get("peak_allocated_bytes"),
        cuda_peak_reserved_bytes=report["cuda_memory_probe"].get("peak_reserved_bytes"),
        start_time_utc=start_time,
        end_time_utc=utc_now(),
        exit_code=0 if report["phase0_environment_gate"] == "PASS" else 1,
        status=report["phase0_environment_gate"],
    )
    write_json_atomic(manifest_path, manifest)
    print_json(report)
    return int(report["phase0_environment_gate"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())

