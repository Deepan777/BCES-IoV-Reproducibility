#!/usr/bin/env python3
"""Generate and execute the deterministic Phase-2 smoke fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.smoke import generate_smoke_fixture
from bces.smoke.network_guard import deny_network_access
from bces.smoke.pipeline import (
    EXPECTED_COVERAGE,
    EXPECTED_UAR,
    SMOKE_POLICY_HASH,
    run_smoke_pipeline,
)
from bces.utils.budget import tree_bytes
from bces.utils.hashing import canonical_json_hash, sha256_file
from bces.utils.reproducibility import (
    git_state,
    new_run_manifest,
    optional_file_hash,
    utc_now,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-output",
        type=Path,
        default=ROOT / "data" / "smoke" / "fixture_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase2" / "smoke_report.json",
    )
    return parser.parse_args()


def _environment_fields() -> dict[str, Any]:
    report_path = ROOT / "outputs" / "phase0" / "environment_report.json"
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    devices = report.get("gpu", {}).get("devices", [])
    device = devices[0] if devices else {}
    pytorch = report.get("pytorch", {})
    return {
        "python_environment_hash": report.get("python", {}).get("environment_hash"),
        "gpu": device.get("name"),
        "cuda": pytorch.get("cuda_build"),
        "driver": device.get("driver_version"),
        "pytorch": pytorch.get("version"),
    }


def main() -> int:
    args = parse_args()
    start = utc_now()
    workspace_before = tree_bytes(ROOT)
    fixture = generate_smoke_fixture()
    write_json_atomic(args.fixture_output, fixture.to_dict())
    with deny_network_access() as network_state:
        first = run_smoke_pipeline(fixture)
        second = run_smoke_pipeline(fixture)
    replay_identical = first == second
    metrics = first["metrics"]
    passed = (
        replay_identical
        and network_state.attempts == 0
        and metrics["coverage"] == EXPECTED_COVERAGE
        and metrics["unsafe_accept_rate"] == EXPECTED_UAR
        and first["observed_surface_boundary_frame"]
        == fixture.known_surface_boundary_frame
    )
    git = git_state(ROOT)
    config_path = ROOT / "configs" / "smoke_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    workspace_after = tree_bytes(ROOT)
    manifest_values = {
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "python_environment_hash": optional_file_hash(
            ROOT / "outputs" / "phase0" / "environment_report.json"
        ),
        "python_lock_hash": optional_file_hash(ROOT / "requirements.lock"),
        "dataset_allowlist_hash": None,
        "dataset_manifest_hash": None,
        "split_hash": None,
        "configuration_hash": canonical_json_hash(config),
        "policy_hash": SMOKE_POLICY_HASH,
        "model_checkpoint_hash": None,
        "calibration_hash": None,
        "seed": 0,
        "workspace_bytes_before": workspace_before,
        "workspace_bytes_after": workspace_after,
        "cuda_peak_allocated_bytes": 0,
        "cuda_peak_reserved_bytes": 0,
        "start_time_utc": start,
        "end_time_utc": utc_now(),
        "exit_code": 0 if passed else 1,
        "status": "PASS" if passed else "FAIL",
        **_environment_fields(),
    }
    report = {
        "schema_version": 1,
        "phase": 2,
        "status": "PASS" if passed else "FAIL",
        "network_guard": {
            "enabled": True,
            "attempts": network_state.attempts,
        },
        "deterministic_replay_identical": replay_identical,
        "fixture_path": str(args.fixture_output.resolve()),
        "fixture_sha256": sha256_file(args.fixture_output),
        "result": first,
        "run_manifest": new_run_manifest(**manifest_values),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
