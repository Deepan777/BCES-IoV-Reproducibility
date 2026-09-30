#!/usr/bin/env python3
"""Build deterministic Phase-3 provenance, split, and adequacy artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.manifest import load_catalog
from bces.data.workflow import (
    blocking_reasons,
    build_data_state,
    load_verified_intersection_capacities,
    write_data_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs" / "data" / "v2xtraj_allowlist.yaml",
    )
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw" / "v2xtraj_primary")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    state = build_data_state(
        catalog,
        raw_root=args.raw_root,
        blockers=blocking_reasons(args.catalog),
        split_capacities=(
            load_verified_intersection_capacities(
                ROOT / "data" / "manifests" / "v2xtraj_intersection_inventory_v1.json"
            )
            if catalog.dataset_id == "v2xtraj_primary"
            else None
        ),
    )
    report = write_data_state(
        state,
        catalog=catalog,
        manifest_path=args.manifest,
        split_path=args.split,
        adequacy_path=args.adequacy,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
