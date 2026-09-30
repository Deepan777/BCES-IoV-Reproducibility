#!/usr/bin/env python3
"""Acquire only official UrbanIng labels/map schema samples, no sensor archive."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.reproducibility import write_json_atomic

DOI = "doi:10.7910/DVN/A9LPY7"
BASE = "https://dataverse.harvard.edu"
CATALOG_URL = f"{BASE}/api/datasets/:persistentId/?persistentId={DOI}"
SELECT = ("crossings_lanelet2map.osm", "labels_av_track_ids.json",
          "20241126_0008_crossing1_01.json")
DEST = ROOT / "data/raw/urbaning_preflight_v1"
MANIFEST = ROOT / "data/manifests/urbaning_preflight_v1.json"


def main():
    if MANIFEST.exists():
        raise FileExistsError(MANIFEST)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"})
    response = session.get(CATALOG_URL, timeout=60)
    response.raise_for_status()
    catalog = response.json()
    if catalog.get("status") != "OK":
        raise RuntimeError("official Dataverse catalog unavailable")
    version = catalog["data"]["latestVersion"]
    if version.get("license", {}).get("name") != "CC BY-NC-ND 4.0":
        raise RuntimeError("dataset license changed; review before acquisition")
    entries = {f["label"]: f for f in version["files"]}
    if any(name not in entries for name in SELECT):
        raise RuntimeError("selected official files absent")
    if sum(entries[name]["dataFile"]["filesize"] for name in SELECT) > 5_000_000:
        raise RuntimeError("preflight exceeds small-data cap")
    records = []
    for name in SELECT:
        item = entries[name]
        data_file = item["dataFile"]
        target = DEST / item.get("directoryLabel", "") / name
        if target.exists():
            raw = target.read_bytes()
        else:
            source_url = f"{BASE}/api/access/datafile/{data_file['id']}"
            result = session.get(source_url, timeout=120)
            result.raise_for_status()
            raw = result.content
        if len(raw) != data_file["filesize"] or hashlib.md5(raw).hexdigest() != data_file["md5"]:
            raise RuntimeError(f"official size/MD5 mismatch: {name}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        records.append({"original_filename": name, "directory_label": item.get("directoryLabel", ""),
                        "datafile_id": data_file["id"], "source_url": f"{BASE}/api/access/datafile/{data_file['id']}",
                        "bytes": len(raw), "provider_md5": data_file["md5"],
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "relative_path": str(target.relative_to(ROOT)).replace("\\", "/")})
        print(f"verified {name}: {len(raw)} bytes", flush=True)
    write_json_atomic(MANIFEST, {"status": "SCHEMA_PREFLIGHT_ONLY_NO_BCES_OUTCOME",
        "provider_dataset_url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/A9LPY7",
        "catalog_url": CATALOG_URL, "license": version["license"],
        "selected_records": records, "total_bytes": sum(r["bytes"] for r in records)})


if __name__ == "__main__":
    main()
