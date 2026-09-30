#!/usr/bin/env python3
"""Acquire only the public V2XPnP sample trajectory, README and vector map.

This step does not unpickle data or run BCES. It records byte hashes and ZIP CRC.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bces.utils.reproducibility import write_json_atomic

RAW = ROOT / "data/raw/v2xpnp_sample_v1"
RECEIPT = ROOT / "data/manifests/v2xpnp_sample_assets_v1.json"
SOURCES = {
    "README.txt": {
        "url": "https://drive.google.com/uc?export=download&id=1G2qL24OFequijcuB7UYo_SApFOc0nPVB",
        "size": 625,
    },
    "trajectory_database_sample.pkl": {
        "url": "https://drive.google.com/uc?export=download&id=19mRyCJmUnwBQgyMi2KR2ORhWKN2WwiLn",
        "size": 4_258_463,
    },
    "map.zip": {
        "url": "https://ucla.app.box.com/index.php?rm=box_download_shared_file"
               "&shared_name=eapz852kkjzov95gxoxl6p613u63j14s&file_id=f_1820521730296",
        "size": 4_386_892,
        "publisher_sha1": "7b1c467178b4663c840f1c82c2b66888c3dc59bb",
    },
}


def acquire(name: str, spec: dict) -> dict:
    destination = RAW / name
    if destination.exists():
        raise FileExistsError(destination)
    size = spec["size"]
    with requests.get(spec["url"], headers={"Range": f"bytes=0-{size-1}"},
                      stream=True, timeout=90) as response:
        expected_range = f"bytes 0-{size-1}/{size}"
        if response.status_code != 206 or response.headers.get("content-range", "").lower() != expected_range:
            raise RuntimeError(f"unexpected response for {name}: {response.status_code} "
                               f"{response.headers.get('content-range')}")
        data = response.content
    if len(data) != size:
        raise RuntimeError(f"length mismatch for {name}")
    sha1 = hashlib.sha1(data).hexdigest()
    if "publisher_sha1" in spec and sha1 != spec["publisher_sha1"]:
        raise RuntimeError(f"publisher digest mismatch for {name}")
    members = None
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            paths = [info.filename for info in infos]
            if len(paths) != len(set(paths)) or any(
                PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or "\\" in path or ":" in path for path in paths
            ):
                raise RuntimeError("unsafe or duplicate ZIP member")
            if archive.testzip() is not None:
                raise RuntimeError("ZIP CRC failure")
            members = [{"name": info.filename, "bytes": info.file_size} for info in infos]
    RAW.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {
        "relative_path": str(destination.relative_to(ROOT)).replace("\\", "/"),
        "source_url": spec["url"],
        "size_bytes": size,
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha1": sha1,
        "zip_members_crc_verified": members,
    }


def main() -> None:
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    records = [acquire(name, spec) for name, spec in SOURCES.items()]
    receipt = {
        "status": "SAMPLE_ASSETS_ACQUIRED_NO_BCES_OUTCOME",
        "publisher": "UCLA V2XPnP Sequential Dataset 1.0 public links",
        "sample_folder": "https://drive.google.com/drive/folders/1ZjVW-OKu-afIoiqfQJgFYwHOWzWE8_e8",
        "map_share": "https://ucla.box.com/s/eapz852kkjzov95gxoxl6p613u63j14s",
        "file_count": len(records),
        "records": records,
        "warning": "Untrusted pickle; inspect opcode globals before any load. Sample scenes are development only.",
    }
    write_json_atomic(RECEIPT, receipt)
    print(json.dumps({"status": receipt["status"],
                      "files": [{k: r[k] for k in ("relative_path", "size_bytes", "sha256")}
                                for r in records]}, indent=2))


if __name__ == "__main__":
    main()
