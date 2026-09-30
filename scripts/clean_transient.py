#!/usr/bin/env python3
"""Print a narrow transient-deletion plan and require CONFIRM=1 to execute it."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_TARGETS = (
    ROOT / ".pytest_cache",
    ROOT / ".mypy_cache",
    ROOT / ".ruff_cache",
    ROOT / ".hypothesis",
    ROOT / ".phase0_docx_review",
    ROOT / "data" / "tmp",
)


def within_workspace(path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def main() -> int:
    plan = []
    targets = (*STATIC_TARGETS, *ROOT.rglob("__pycache__"))
    for target in sorted(set(targets)):
        if not within_workspace(target):
            raise RuntimeError(f"refusing target outside workspace: {target}")
        if target.exists():
            plan.append(str(target.resolve()))
    print(
        json.dumps(
            {"deletion_plan": plan, "confirmed": os.getenv("CONFIRM") == "1"}, indent=2
        )
    )
    if os.getenv("CONFIRM") != "1":
        return 2
    for target_text in plan:
        target = Path(target_text)
        if target.name == "tmp":
            for child in target.iterdir():
                if child.name == ".gitkeep":
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        elif target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
