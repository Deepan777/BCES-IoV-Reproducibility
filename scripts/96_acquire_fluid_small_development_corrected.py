#!/usr/bin/env python3
"""Correct only the figshare source filename in the unrun download contract.

The first acquisition attempt failed its metadata check before creating
the output directory or downloading any data. Keep that failed script and
plan unchanged; the source file ID, size, digest, and tranche are identical.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / "scripts/95_acquire_fluid_small_development.py"
ORIGINAL_SHA256 = "59a4ee983a5bec2e42969bac453321eaab5ff0a4fea5f4db5699a1414b9a4444"


def main():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise RuntimeError("failed original acquisition script changed")
    spec = importlib.util.spec_from_file_location("fluid_acquisition_original", ORIGINAL)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load original acquisition script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.FILES = (
        (61924402, "README.md", 22447, "e0ef4265daa22f774c1e3c8be642bc2e"),
        (61923805, "Conflict.zip", 548688, "014d6d2731ff211b9b6f36afda66423d"),
        (64179079, "derived_data.zip", 30932517, "fc2870464b3509130ce606c4f43a492b"),
    )
    module.main()
    output = ROOT / "data/external/fluid_small_development_v1"
    manifest = output / "manifest.json"
    receipt = {"status": "CORRECTED_PRE_DOWNLOAD_FILENAME_CONTRACT",
               "original_script_sha256": ORIGINAL_SHA256,
               "corrected_wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "correction": "official file 64179079 is named derived_data.zip; local disambiguated filename remains derived_data_64179079.zip",
               "no_data_acquired_by_failed_original_attempt": True}
    (output / "wrapper_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
