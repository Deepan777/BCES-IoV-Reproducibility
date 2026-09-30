#!/usr/bin/env python3
"""Derive the Phase-1 exit decision from test, protocol, data, and budget evidence."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TypedDict

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.geometry.codebooks import codebook_sha256
from bces.protocol.codec import decode_packet
from bces.utils.budget import Budget, build_budget_report
from bces.utils.hashing import canonical_json_hash
from bces.utils.reproducibility import (
    git_state,
    utc_now,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-report",
        type=Path,
        default=ROOT / "outputs" / "phase1" / "protocol_geometry_test_report.xml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase1" / "phase1_gate_report.json",
    )
    return parser.parse_args()


class TestSummary(TypedDict):
    tests: int
    failures: int
    errors: int
    skipped: int
    test_names: list[str]


def test_summary(path: Path) -> TestSummary:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    names = [
        case.attrib.get("name", "")
        for suite in suites
        for case in suite.findall("testcase")
    ]
    return {
        "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failures": sum(int(suite.attrib.get("failures", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "test_names": names,
    }


def real_payload_files() -> list[str]:
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
    tests = test_summary(args.test_report)
    golden = json.loads(
        (ROOT / "tests" / "protocol_vectors" / "v1_golden.json").read_text(
            encoding="utf-8"
        )
    )
    encoded = bytes.fromhex(golden["expected_hex"])
    decoded = decode_packet(encoded)
    with (ROOT / "configs" / "budget.yaml").open("r", encoding="utf-8") as handle:
        budget_values = yaml.safe_load(handle) or {}
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data", **budget_values)
    budget_report = build_budget_report(budget)
    payload_files = real_payload_files()
    test_names = set(tests["test_names"])
    test_counts = {
        key: tests[key] for key in ("tests", "failures", "errors", "skipped")
    }
    checks = [
        {
            "name": "phase1_tests_pass",
            "passed": tests["tests"] > 0
            and tests["failures"] == 0
            and tests["errors"] == 0
            and tests["skipped"] == 0,
        },
        {"name": "exact_48_byte_packet", "passed": len(encoded) == 48},
        {
            "name": "golden_vector_decodes",
            "passed": decoded.query_id == golden["packet"]["query_id"],
        },
        {
            "name": "shrinkage_property_executed",
            "passed": "test_conservative_shrinkage_never_increases_acceptance"
            in test_names,
        },
        {
            "name": "codebook_hash_frozen",
            "passed": codebook_sha256()
            == "2ba57e3d277ecda2ea947fe54e7c6c139b5a117c714a2f54aa56d4c7338106b1",
        },
        {
            "name": "workspace_budget_pass",
            "passed": budget_report["status"] in ("pass", "warning")
            and budget_report["measurements"]["output_reserve_preserved"],
        },
        {"name": "no_real_data_payload", "passed": payload_files == []},
    ]
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    git = git_state(ROOT)
    report = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "phase": 1,
        "status": status,
        "checks": checks,
        "tests": test_counts,
        "protocol": {
            "packet_bytes": len(encoded),
            "golden_vector_sha256": canonical_json_hash(golden),
            "codebook_sha256": codebook_sha256(),
        },
        "real_data_payload_files": payload_files,
        "budget": budget_report,
        "git": git,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
