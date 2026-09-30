#!/usr/bin/env python3
"""Extract explicitly named V2X-Traj ZIP members through bounded ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bces.data.remote_zip import RemoteZipMember, download_member
from bces.utils.hashing import canonical_json_hash


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", action="append", required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "manifests" / "v2xtraj_archive_index_v1.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data" / "tmp" / "v2xtraj_review",
    )
    parser.add_argument("--max-compressed-bytes", type=int, default=10_000_000)
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    expected = index.pop("index_sha256")
    if canonical_json_hash(index) != expected:
        raise ValueError("remote archive index hash does not verify")
    by_name = {item["name"]: RemoteZipMember(**item) for item in index["members"]}
    selected = []
    for name in args.member:
        if name not in by_name:
            raise ValueError(f"member is absent from the verified index: {name}")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "v2x-traj":
            raise ValueError("member path is not contained")
        selected.append(by_name[name])
    compressed_total = sum(item.compressed_bytes for item in selected)
    if compressed_total > args.max_compressed_bytes:
        raise ValueError("selected members exceed the approved extraction tranche")
    args.output_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    results = []
    for member in selected:
        destination = (args.output_root / Path(*PurePosixPath(member.name).parts[1:])).resolve()
        destination.relative_to(args.output_root.resolve())
        if destination.exists():
            raise FileExistsError(f"immutable destination already exists: {destination}")
        payload = download_member(session, index["official_url"], member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        with part.open("xb") as handle:
            handle.write(payload)
        os.replace(part, destination)
        results.append(
            {
                "member": member.name,
                "destination": destination.relative_to(ROOT).as_posix(),
                "compressed_bytes": member.compressed_bytes,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    print(json.dumps({"downloaded": results, "compressed_bytes": compressed_total}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
