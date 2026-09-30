#!/usr/bin/env python3
"""Build isolated per-dataset Phase-3 state and the primary-only aggregate gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.manifest import load_catalog
from bces.data.multidataset import aggregate_adequacy, load_multidataset_plan
from bces.data.workflow import (
    blocking_reasons,
    build_data_state,
    load_verified_intersection_capacities,
    write_data_state,
)
from bces.utils.reproducibility import write_json_atomic
from bces.utils.hashing import canonical_json_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs" / "data" / "multi_dataset_plan.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase3" / "multidataset_adequacy_report.json",
    )
    args = parser.parse_args()
    plan = load_multidataset_plan(args.plan)
    reports = {}
    artifacts = {}
    primary_capacities = load_verified_intersection_capacities(
        ROOT / "data" / "manifests" / "v2xtraj_intersection_inventory_v1.json"
    )
    for item in plan.datasets:
        catalog_path = ROOT / item.catalog
        catalog = load_catalog(catalog_path)
        if catalog.dataset_id != item.dataset_id or catalog.evidence_role != item.evidence_role.value:
            raise ValueError(f"catalog does not match plan: {item.dataset_id}")
        state = build_data_state(
            catalog,
            raw_root=ROOT / "data" / "raw" / item.dataset_id,
            blockers=blocking_reasons(catalog_path),
            split_capacities=(
                primary_capacities if item.dataset_id == "v2xtraj_primary" else None
            ),
        )
        manifest = ROOT / "data" / "manifests" / f"{item.dataset_id}_manifest_v1.json"
        split = ROOT / "data" / "manifests" / f"{item.dataset_id}_split_v1.json"
        adequacy = ROOT / "outputs" / "phase3" / f"{item.dataset_id}_adequacy_report.json"
        artifacts[item.dataset_id] = write_data_state(
            state,
            catalog=catalog,
            manifest_path=manifest,
            split_path=split,
            adequacy_path=adequacy,
        )
        reports[item.dataset_id] = state.adequacy
    aggregate = aggregate_adequacy(plan, reports)
    aggregate["artifacts"] = artifacts
    aggregate.pop("report_sha256", None)
    aggregate["report_sha256"] = canonical_json_hash(aggregate)
    write_json_atomic(args.output, aggregate)
    # Preserve the conventional primary adequacy path for downstream Phase-3 tools.
    write_json_atomic(
        ROOT / "outputs" / "phase3" / "adequacy_report.json",
        reports[plan.primary_dataset_id].to_dict(),
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
