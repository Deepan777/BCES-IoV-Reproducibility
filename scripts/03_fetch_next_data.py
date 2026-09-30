#!/usr/bin/env python3
"""Execute exactly one reviewed Phase-3 allowlisted download tranche."""

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

from bces.data.download import fetch_next_data
from bces.data.manifest import load_catalog
from bces.data.workflow import (
    blocking_reasons,
    build_data_state,
    load_verified_manifest,
    write_data_state,
)
from bces.utils.budget import Budget, build_budget_report
from bces.utils.reproducibility import git_state, utc_now, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs" / "data" / "v2xtraj_allowlist.yaml",
    )
    parser.add_argument(
        "--budget", type=Path, default=ROOT / "configs" / "budget.yaml"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "v2xtraj_primary_manifest_v1.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "manifests" / "v2xtraj_primary_split_v1.json",
    )
    parser.add_argument(
        "--adequacy",
        type=Path,
        default=ROOT / "outputs" / "phase3" / "adequacy_report.json",
    )
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "data" / "raw" / "v2xtraj_primary"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase3" / "fetch_decision.json",
    )
    return parser.parse_args()


def _record_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for record in load_verified_manifest(path):
        item = record.__dict__.copy()
        item["source_role"] = record.source_role.value
        records.append(item)
    return records


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    budget_values = yaml.safe_load(args.budget.read_text(encoding="utf-8")) or {}
    budget = Budget(workspace_root=ROOT, data_root=ROOT / "data", **budget_values)
    blockers = blocking_reasons(args.catalog)
    current_state = build_data_state(
        catalog, raw_root=args.raw_root, blockers=blockers
    )
    records = _record_dicts(args.manifest)
    result = fetch_next_data(
        catalog,
        budget,
        records=records,
        split_assignments={
            key: value.value for key, value in current_state.split.by_intersection.items()
        },
        blockers=blockers,
        raw_root=args.raw_root,
    )
    downloaded: list[dict[str, Any]] = [
        {
            "entry_id": item.entry.entry_id,
            "source_file": item.entry.destination,
            "source_bytes": item.measured_bytes,
            "sha256": item.sha256,
        }
        for item in result.downloaded
    ]
    if downloaded:
        current_state = build_data_state(
            catalog, raw_root=args.raw_root, blockers=blockers
        )
        write_data_state(
            current_state,
            catalog=catalog,
            manifest_path=args.manifest,
            split_path=args.split,
            adequacy_path=args.adequacy,
        )
    report = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "decision": result.decision.value,
        "one_tranche_only": True,
        "downloaded": downloaded,
        "downloaded_bytes": sum(int(item["source_bytes"]) for item in downloaded),
        "adequacy_before": result.adequacy_before.to_dict(),
        "adequacy_after": result.adequacy_after.to_dict(),
        "catalog_sha256": catalog.sha256,
        "budget": build_budget_report(budget),
        "git": git_state(ROOT),
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
