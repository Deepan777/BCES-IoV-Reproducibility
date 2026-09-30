#!/usr/bin/env python3
"""Measure and enforce the complete BCES-IoV workspace budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from bces.utils.budget import (  # noqa: E402
    Budget,
    BudgetError,
    build_budget_report,
    require_headroom,
    write_budget_report,
)
from bces.utils.reproducibility import write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "budget.yaml")
    parser.add_argument("--incoming-bytes", type=int, default=0)
    parser.add_argument("--no-preserve-reserve", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_budget(args: argparse.Namespace) -> Budget:
    with args.config.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    workspace = args.workspace_root.resolve()
    data_root = (args.data_root or workspace / "data").resolve()
    return Budget(workspace_root=workspace, data_root=data_root, **values)


def main() -> int:
    args = parse_args()
    report_path = args.report or args.workspace_root / "outputs" / "phase0" / "budget_report.json"
    try:
        budget = load_budget(args)
        require_headroom(
            budget,
            args.incoming_bytes,
            preserve_reserve=not args.no_preserve_reserve,
        )
        report = write_budget_report(report_path, budget)
        report["enforcement"] = {"status": "PASS", "incoming_bytes": args.incoming_bytes}
        write_json_atomic(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (BudgetError, ValueError, OSError) as exc:
        failure = {
            "schema_version": 1,
            "status": "fail",
            "enforcement": {
                "status": "FAIL",
                "incoming_bytes": args.incoming_bytes,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
        write_json_atomic(report_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

