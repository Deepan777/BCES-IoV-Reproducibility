#!/usr/bin/env python3
"""Run unchanged diverse-label generator on preregistered disjoint dev seeds."""

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
GENERATOR = ROOT / "scripts/131_generate_diverse_empty_labels_dev.py"
OUTPUT = ROOT / "outputs/study_b/diverse_empty_labels_development_v2"
PLAN = ROOT / "docs/STUDY_B_EMPTY_TEACHER_TRANSFER_DEV_V1_PLAN.md"
SEEDS = tuple(range(8702000, 8702064))


def main():
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if registration["status"] != "FROZEN_DISJOINT_DEVELOPMENT_TRANSFER_V1":
        raise RuntimeError("wrong transfer registration")
    if registration["new_seeds"] != list(SEEDS) or registration["new_output"] != str(OUTPUT.relative_to(ROOT)).replace("\\", "/"):
        raise RuntimeError("new cohort identity changed")
    for name, digest in registration["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError("frozen source changed: " + name)
    if registration["old_training_seeds"] != list(range(8701000, 8701064)):
        raise RuntimeError("training cohort identity changed")
    if OUTPUT.exists() and (OUTPUT / "report.json").exists():
        raise RuntimeError("new cohort completed; preserve artifacts")
    spec = importlib.util.spec_from_file_location("unchanged_diverse_empty_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SEEDS = SEEDS
    module.OUTPUT = OUTPUT
    module.PLAN = PLAN
    if module.STRATA != ("absent", "present_near", "present_control"):
        raise RuntimeError("generator stratum contract changed")
    if module.CONTINUATIONS != (-1.0, 0.0, 1.0) or module.AGES != (0.0, 0.2, 0.4, 0.8, 1.2, 2.0):
        raise RuntimeError("generator continuation/age contract changed")
    module.main()


if __name__ == "__main__":
    main()
