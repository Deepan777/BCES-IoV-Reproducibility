#!/usr/bin/env python3
"""Acquire all official UrbanIng label JSONs only, with provider hash receipts."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.reproducibility import write_json_atomic

BASE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/A9LPY7"
CATALOG = f"{BASE}/api/datasets/:persistentId/?persistentId={DOI}"
DEST = ROOT / "data/raw/urbaning_labels_v1"
MANIFEST = ROOT / "data/manifests/urbaning_labels_v1.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


def main() -> None:
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    response = session.get(CATALOG, timeout=60)
    response.raise_for_status()
    version = response.json()["data"]["latestVersion"]
    if version.get("license", {}).get("name") != "CC BY-NC-ND 4.0":
        raise RuntimeError("provider license changed; review terms")
    entries = sorted((x for x in version["files"] if x.get("directoryLabel") == "labels" and x["label"].endswith(".json")), key=lambda x: x["label"])
    if len(entries) != 34:
        raise RuntimeError(f"expected exactly 34 label files, found {len(entries)}")
    if sum(x["dataFile"]["filesize"] for x in entries) > 150_000_000:
        raise RuntimeError("label tranche exceeds 150 MB acquisition cap")
    records = []
    for i, item in enumerate(entries, 1):
        data = item["dataFile"]
        target = DEST / "labels" / item["label"]
        if target.exists():
            raw = target.read_bytes()
        else:
            url = f"{BASE}/api/access/datafile/{data['id']}"
            response = session.get(url, timeout=180)
            response.raise_for_status()
            raw = response.content
        if len(raw) != data["filesize"] or hashlib.md5(raw).hexdigest() != data["md5"]:
            raise RuntimeError(f"provider size/MD5 mismatch: {item['label']}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        records.append({"original_filename": item["label"], "directory_label": "labels", "datafile_id": data["id"], "bytes": len(raw), "provider_md5": data["md5"], "sha256": hashlib.sha256(raw).hexdigest(), "relative_path": str(target.relative_to(ROOT)).replace("\\", "/")})
        print(f"{i}/34 verified {item['label']} ({len(raw)} bytes)", flush=True)
    write_json_atomic(MANIFEST, {"status": "LABELS_ACQUIRED_NO_BCES_OUTCOME", "provider_dataset_url": f"{BASE}/dataset.xhtml?persistentId={DOI}", "catalog_url": CATALOG, "license": version["license"], "file_count": len(records), "total_bytes": sum(x["bytes"] for x in records), "records": records})


if __name__ == "__main__":
    main()
