#!/usr/bin/env python3
"""Replay the disjoint cohort with the unchanged 576-trace raw verifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.utils.hashing import sha256_file

REGISTRATION = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_REGISTRATION.json"
VERIFIER = ROOT / "scripts/132_verify_diverse_empty_labels_dev.py"
NEW_PLAN = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_PLAN.md"
OLD_PLAN = ROOT / "docs/STUDY_B_DIVERSE_EMPTY_LABELS_DEV_V1_PLAN.md"
NEW_DIR = ROOT / "outputs/study_b/diverse_empty_labels_development_v2"


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if registration["new_seeds"] != list(range(8702000, 8702064)):
        raise RuntimeError("disjoint cohort registration changed")
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("frozen source changed: " + name)
    protocol = json.loads((NEW_DIR / "protocol.json").read_text(encoding="utf-8"))
    if protocol["seeds"] != registration["new_seeds"]:
        raise RuntimeError("new cohort seed identity changed")
    if sha256_file(NEW_PLAN) != protocol["plan_sha256"]:
        raise RuntimeError("new cohort plan changed")
    spec = importlib.util.spec_from_file_location("unchanged_diverse_empty_raw_verifier", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DIR = NEW_DIR
    original_sha256 = module.sha256_file

    def sha256_with_new_plan(path):
        path = Path(path)
        if path == OLD_PLAN:
            path = NEW_PLAN
        return original_sha256(path)

    module.sha256_file = sha256_with_new_plan
    module.main()


if __name__ == "__main__":
    main()
