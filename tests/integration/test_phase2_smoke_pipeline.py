from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bces.data.smoke import generate_smoke_fixture
from bces.smoke.pipeline import EXPECTED_COVERAGE, EXPECTED_UAR, run_smoke_pipeline

pytestmark = pytest.mark.phase2
ROOT = Path(__file__).resolve().parents[2]


def test_end_to_end_smoke_pipeline_is_deterministic() -> None:
    fixture = generate_smoke_fixture()
    first = run_smoke_pipeline(fixture)
    second = run_smoke_pipeline(fixture)
    assert first == second
    assert first["metrics"]["coverage"] == EXPECTED_COVERAGE
    assert first["metrics"]["unsafe_accept_rate"] == EXPECTED_UAR
    assert first["metrics"]["accepted_count"] == 10
    assert first["metrics"]["accepted_oracle_invalid_count"] == 1
    assert first["metrics"]["refresh_count"] == 1
    assert first["metrics"]["invalid_count"] == 9
    assert first["observed_surface_boundary_frame"] == 11
    assert all(row["packet_bytes"] == 48 for row in first["rows"])


def test_smoke_cli_writes_fixture_and_report(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "02_smoke_pipeline.py"),
            "--fixture-output",
            str(fixture_path),
            "--output",
            str(report_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["network_guard"] == {"enabled": True, "attempts": 0}
    assert report["deterministic_replay_identical"] is True
    assert fixture["evidence_use"] == "software_verification_only"
