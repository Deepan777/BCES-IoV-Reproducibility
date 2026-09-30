from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bces.utils.hashing import canonical_json_hash, sha256_file


pytestmark = pytest.mark.phase0
ROOT = Path(__file__).resolve().parents[2]


def test_budget_cli_writes_passing_report(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    data = workspace / "data"
    data.mkdir(parents=True)
    config = tmp_path / "budget.yaml"
    config.write_text(
        "\n".join(
            [
                "initial_data_review_bytes: 100",
                "max_download_tranche_bytes: 50",
                "required_output_reserve_bytes: 100",
                "warn_workspace_bytes: 900",
                "max_workspace_bytes: 1000",
                "max_cuda_allocated_bytes: 100",
            ]
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "00_check_budget.py"),
            "--workspace-root",
            str(workspace),
            "--data-root",
            str(data),
            "--config",
            str(config),
            "--report",
            str(report),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert json.loads(report.read_text(encoding="utf-8"))["enforcement"]["status"] == "PASS"


def test_real_payload_is_absent_or_confined_to_receipt_backed_raw_root() -> None:
    raw_files = [
        path
        for path in (ROOT / "data" / "raw").rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    processed_files = [
        path
        for path in (ROOT / "data" / "processed").rglob("*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    assert processed_files == []
    receipt_paths = sorted(
        (ROOT / "data" / "manifests").glob("v2xtraj_download_receipt_v*.json")
    )
    expected = set()
    for receipt_path in receipt_paths:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        expected_hash = receipt.pop("receipt_sha256")
        assert canonical_json_hash(receipt) == expected_hash
        assert receipt["status"] == "complete"
        expected.update(ROOT / item["destination"] for item in receipt["records"])
    for manifest_path in sorted((ROOT / "data" / "manifests").glob("cooperscene_annotations*_v1.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "annotations_acquired_no_bces_outcome_computed"
        assert manifest["file_count"] == len(manifest["records"])
        for item in manifest["records"]:
            path = ROOT / item["relative_path"]
            assert path.is_relative_to(ROOT / "data" / "raw" / "cooperscene_annotations_v1")
            assert sha256_file(path) == item["sha256"]
            expected.add(path)
    for name, status, records_key in (
        ("urbaning_preflight_v1.json", "SCHEMA_PREFLIGHT_ONLY_NO_BCES_OUTCOME", "selected_records"),
        ("urbaning_labels_v1.json", "LABELS_ACQUIRED_NO_BCES_OUTCOME", "records"),
    ):
        manifest_path = ROOT / "data" / "manifests" / name
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == status
        for item in manifest[records_key]:
            path = ROOT / item["relative_path"]
            assert path.is_relative_to(ROOT / "data" / "raw" / name.removesuffix(".json"))
            assert sha256_file(path) == item["sha256"]
            expected.add(path)
    v2xreal_receipt_path = ROOT / "data" / "manifests" / "v2xreal_vips_assets_v1.json"
    if v2xreal_receipt_path.exists():
        receipt = json.loads(v2xreal_receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "ACQUIRED_VERIFIED_POSITION_PROXY_ONLY"
        assert receipt["file_count"] == len(receipt["records"])
        for item in receipt["records"]:
            path = ROOT / item["relative_path"]
            assert path.is_relative_to(ROOT / "data" / "raw" / "v2x_real_lidar64") or path.is_relative_to(
                ROOT / "data" / "raw" / "vips_v2xreal_assets"
            )
            assert path.stat().st_size == item["size_bytes"]
            assert sha256_file(path) == item["sha256"]
            expected.add(path)
    v2xpnp_receipt_path = ROOT / "data" / "manifests" / "v2xpnp_sample_assets_v1.json"
    if v2xpnp_receipt_path.exists():
        receipt = json.loads(v2xpnp_receipt_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "SAMPLE_ASSETS_ACQUIRED_NO_BCES_OUTCOME"
        assert receipt["file_count"] == len(receipt["records"])
        for item in receipt["records"]:
            path = ROOT / item["relative_path"]
            assert path.is_relative_to(ROOT / "data" / "raw" / "v2xpnp_sample_v1")
            assert path.stat().st_size == item["size_bytes"]
            assert sha256_file(path) == item["sha256"]
            expected.add(path)
    assert set(raw_files) == expected
    assert all(
        path.is_relative_to(ROOT / "data" / "raw" / "v2xtraj_primary")
        or path.is_relative_to(ROOT / "data" / "raw" / "cooperscene_annotations_v1")
        or path.is_relative_to(ROOT / "data" / "raw" / "urbaning_preflight_v1")
        or path.is_relative_to(ROOT / "data" / "raw" / "urbaning_labels_v1")
        or path.is_relative_to(ROOT / "data" / "raw" / "v2x_real_lidar64")
        or path.is_relative_to(ROOT / "data" / "raw" / "vips_v2xreal_assets")
        or path.is_relative_to(ROOT / "data" / "raw" / "v2xpnp_sample_v1")
        for path in raw_files
    )
