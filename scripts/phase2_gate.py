#!/usr/bin/env python3
"""Derive the Phase-2 exit decision from smoke, test, and budget evidence."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.smoke import generate_smoke_fixture
from bces.smoke.pipeline import (
    EXPECTED_COVERAGE,
    EXPECTED_UAR,
    SMOKE_REFRESH_GUARD_BAND,
    SMOKE_SURFACE_OFFSETS,
    SMOKE_THRESHOLDS,
)
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import sha256_file
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


class TestSummary(TypedDict):
    tests: int
    failures: int
    errors: int
    skipped: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-report",
        type=Path,
        default=ROOT / "outputs" / "phase2" / "smoke_test_report.xml",
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        default=ROOT / "outputs" / "phase2" / "smoke_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase2" / "phase2_gate_report.json",
    )
    return parser.parse_args()


def _test_summary(path: Path) -> TestSummary:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }  # type: ignore[return-value]


def _real_payload_files() -> list[str]:
    files: list[str] = []
    for directory in (ROOT / "data" / "raw", ROOT / "data" / "processed"):
        if not directory.exists():
            continue
        files.extend(
            str(path.relative_to(ROOT))
            for path in directory.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        )
    return sorted(files)


def main() -> int:
    args = parse_args()
    tests = _test_summary(args.test_report)
    smoke_report = json.loads(args.smoke_report.read_text(encoding="utf-8"))
    fixture_path = ROOT / "data" / "smoke" / "fixture_v1.json"
    fixture_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    generated_fixture = generate_smoke_fixture().to_dict()
    config = yaml.safe_load(
        (ROOT / "configs" / "smoke_v1.yaml").read_text(encoding="utf-8")
    )
    with (ROOT / "configs" / "budget.yaml").open("r", encoding="utf-8") as handle:
        budget_values = yaml.safe_load(handle) or {}
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data", **budget_values)
    budget_report = build_budget_report(budget)
    result = smoke_report["result"]
    metrics = result["metrics"]
    expected = config["expected_metrics"]
    objects = fixture_payload["cached_message"]["objects"]
    real_payload_files = _real_payload_files()
    artifact_limit = int(config["workspace_artifact_limit_bytes"])
    checks = [
        {
            "name": "phase2_tests_pass",
            "passed": tests["tests"] > 0
            and tests["failures"] == 0
            and tests["errors"] == 0
            and tests["skipped"] == 0,
        },
        {
            "name": "generated_fixture_exactly_20_frames",
            "passed": len(fixture_payload["frames"]) == 20
            and fixture_payload == generated_fixture,
        },
        {
            "name": "fixture_actor_and_behavior_coverage",
            "passed": sum(item["class_id"] == 1 for item in objects) == 2
            and sum(item["class_id"] == 2 for item in objects) == 1
            and result["behaviors"] == ["brake", "keep"]
            and result["occluded_track_ids"] == ["ped-hidden"],
        },
        {
            "name": "known_boundary_crossing",
            "passed": result["known_surface_boundary_frame"]
            == result["observed_surface_boundary_frame"]
            == config["fixture"]["known_surface_boundary_frame"],
        },
        {
            "name": "deterministic_expected_coverage",
            "passed": smoke_report["deterministic_replay_identical"] is True
            and metrics["coverage"] == expected["coverage"],
        },
        {
            "name": "deterministic_expected_uar",
            "passed": metrics["unsafe_accept_rate"]
            == expected["unsafe_accept_rate"],
        },
        {
            "name": "registered_config_matches_code",
            "passed": config["oracle_thresholds"] == asdict(SMOKE_THRESHOLDS)
            and config["surface"]["offsets"] == SMOKE_SURFACE_OFFSETS[0]
            and config["surface"]["refresh_guard_band"]
            == SMOKE_REFRESH_GUARD_BAND
            and expected["coverage"] == EXPECTED_COVERAGE
            and expected["unsafe_accept_rate"] == EXPECTED_UAR,
        },
        {
            "name": "complete_pipeline_and_exact_codec",
            "passed": len(result["rows"]) == 20
            and all(row["packet_bytes"] == 48 for row in result["rows"]),
        },
        {
            "name": "network_access_forbidden",
            "passed": smoke_report["network_guard"]
            == {"enabled": True, "attempts": 0},
        },
        {
            "name": "verification_only_not_evidence",
            "passed": result["evidence_use"] == "software_verification_only"
            and fixture_payload["evidence_use"] == "software_verification_only",
        },
        {
            "name": "fixture_hash_matches_report",
            "passed": sha256_file(fixture_path) == smoke_report["fixture_sha256"],
        },
        {
            "name": "phase2_artifacts_under_10mb",
            "passed": budget_report["measurements"]["workspace_bytes"]
            < artifact_limit
            and budget_report["measurements"]["data_bytes"] < artifact_limit,
        },
        {
            "name": "workspace_budget_and_reserve_pass",
            "passed": budget_report["status"] in ("pass", "warning")
            and budget_report["measurements"]["output_reserve_preserved"],
        },
        {"name": "no_real_data_payload", "passed": real_payload_files == []},
    ]
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    report = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "phase": 2,
        "status": status,
        "checks": checks,
        "tests": tests,
        "metrics": metrics,
        "fixture_sha256": sha256_file(fixture_path),
        "deterministic_sha256": result["deterministic_sha256"],
        "real_data_payload_files": real_payload_files,
        "budget": budget_report,
        "git": git_state(ROOT),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
