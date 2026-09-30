"""Read-only integrity and schema audit of the official UCLA LiDAR-64 test ZIP.

This script does not extract archive members or evaluate BCES outcomes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path, PurePosixPath
import sys
import zipfile


EXPECTED_SIZE = 5_772_980_847
EXPECTED_SHA1 = "6a13f3e59d1abcd87b0f09757699e35e5c464b36"
DEFAULT_PATH = Path("data/raw/v2x_real_lidar64/test.zip")


def digests(path: Path) -> tuple[str, str]:
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha1.update(chunk)
            sha256.update(chunk)
    return sha1.hexdigest(), sha256.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--full-crc", action="store_true")
    args = parser.parse_args()
    path = args.archive
    if not path.is_file():
        raise SystemExit(f"Missing archive: {path}")
    size = path.stat().st_size
    print(f"archive={path.resolve()}")
    print(f"bytes={size}")
    if size != EXPECTED_SIZE:
        raise SystemExit(f"Size mismatch: expected {EXPECTED_SIZE}")
    sha1, sha256 = digests(path)
    print(f"sha1={sha1}")
    print(f"sha256={sha256}")
    if sha1 != EXPECTED_SHA1:
        raise SystemExit(f"Publisher SHA-1 mismatch: expected {EXPECTED_SHA1}")

    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [entry.filename for entry in members]
        dupes = [name for name, n in Counter(names).items() if n > 1]
        unsafe = [
            name
            for name in names
            if PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or "\\" in name
            or ":" in name
        ]
        if dupes or unsafe:
            raise SystemExit(f"Unsafe ZIP member paths: duplicates={dupes[:5]}, unsafe={unsafe[:5]}")
        suffixes = Counter(Path(name).suffix.lower() for name in names if not name.endswith("/"))
        top = Counter(PurePosixPath(name).parts[0] for name in names if name)
        second = defaultdict(set)
        for name in names:
            parts = PurePosixPath(name).parts
            if len(parts) >= 2:
                second[parts[0]].add(parts[1])
        print(f"members={len(members)}")
        print(f"uncompressed_bytes={sum(entry.file_size for entry in members)}")
        print(f"top_level={dict(top.most_common(15))}")
        print(f"extensions={dict(suffixes.most_common(15))}")
        for key in sorted(second):
            values = sorted(second[key])
            print(f"second_level[{key}]={len(values)} sample={values[:8]}")
        print(f"first_members={names[:12]}")
        if args.full_crc:
            failed = archive.testzip()
            if failed is not None:
                raise SystemExit(f"CRC failure: {failed}")
            print("all_member_crc=pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
