#!/usr/bin/env python3
"""Acquire the same registered central windows for vehicle agents 2 and 3."""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cooperscene_acquisition_v1", HERE / "37_acquire_cooperscene_annotations.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen annotation acquisition procedure")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.AGENTS = (2, 3)
module.MANIFEST = module.ROOT / "data/manifests/cooperscene_annotations_extra_agents_v1.json"

if __name__ == "__main__":
    module.main()
