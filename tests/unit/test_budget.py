from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from bces.utils.budget import (
    Budget,
    BudgetExceeded,
    BudgetWarning,
    DatasetSymlinkError,
    assert_cuda_peak,
    available_download_bytes,
    require_headroom,
    tree_bytes,
    validate_dataset_symlinks,
    write_budget_report,
)


pytestmark = pytest.mark.phase0


def small_budget(root: Path, **overrides: int) -> Budget:
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    values = {
        "initial_data_review_bytes": 10,
        "max_download_tranche_bytes": 10,
        "required_output_reserve_bytes": 5,
        "warn_workspace_bytes": 15,
        "max_workspace_bytes": 20,
        "max_cuda_allocated_bytes": 10,
    }
    values.update(overrides)
    return Budget(workspace_root=root, data_root=data, **values)


def test_recursive_accounting_includes_hidden_cache_part_and_temp(tmp_path: Path) -> None:
    (tmp_path / ".hidden").write_bytes(b"123")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "entry.bin").write_bytes(b"4567")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "transfer.part").write_bytes(b"89")
    assert tree_bytes(tmp_path) == 9


def test_external_dataset_symlink_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    budget = small_budget(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload.bin").write_bytes(b"external")
    link = budget.data_root / "external-link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except OSError:
        proc = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    with pytest.raises(DatasetSymlinkError):
        validate_dataset_symlinks(budget)


def test_warning_threshold_is_reported_without_crossing_safe_ceiling(tmp_path: Path) -> None:
    budget = small_budget(
        tmp_path,
        warn_workspace_bytes=5,
        max_workspace_bytes=20,
        required_output_reserve_bytes=5,
    )
    (tmp_path / "payload.bin").write_bytes(b"12345")
    with pytest.warns(BudgetWarning):
        require_headroom(budget, preserve_reserve=True)


def test_reserve_violation_fails_before_workspace_cap(tmp_path: Path) -> None:
    budget = small_budget(tmp_path)
    (tmp_path / "payload.bin").write_bytes(b"12345678901234")
    with pytest.raises(BudgetExceeded, match="output reserve"):
        require_headroom(budget, incoming_bytes=2)


def test_workspace_cap_is_exclusive(tmp_path: Path) -> None:
    budget = small_budget(tmp_path)
    (tmp_path / "payload.bin").write_bytes(b"1234567890123456789")
    with pytest.raises(BudgetExceeded, match="reaches/exceeds"):
        require_headroom(
            budget,
            incoming_bytes=1,
            preserve_reserve=False,
        )


def test_download_tranche_limit(tmp_path: Path) -> None:
    budget = small_budget(tmp_path)
    with pytest.raises(BudgetExceeded, match="tranche cap"):
        require_headroom(budget, incoming_bytes=11)


def test_negative_incoming_is_rejected(tmp_path: Path) -> None:
    budget = small_budget(tmp_path)
    with pytest.raises(ValueError):
        require_headroom(budget, incoming_bytes=-1)


@given(st.integers(min_value=0, max_value=40))
def test_arbitrary_incoming_never_authorizes_a_cap_crossing(incoming: int) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        budget = small_budget(root)
        if incoming > 10:
            with pytest.raises(BudgetExceeded):
                require_headroom(budget, incoming_bytes=incoming)
        else:
            require_headroom(budget, incoming_bytes=incoming)


def test_available_download_bytes_preserves_reserve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    budget = small_budget(tmp_path)
    monkeypatch.setattr(
        "bces.utils.budget.shutil.disk_usage",
        lambda _path: os.statvfs(tmp_path) if hasattr(os, "statvfs") else type("D", (), {"free": 100})(),
    )
    headroom = available_download_bytes(budget)
    assert 0 <= headroom <= budget.max_workspace_bytes - budget.required_output_reserve_bytes


def test_budget_report_is_machine_readable(tmp_path: Path) -> None:
    budget = small_budget(tmp_path)
    report_path = tmp_path / "report.json"
    report = write_budget_report(report_path, budget)
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["measurements"]["workspace_bytes"] == report["measurements"]["workspace_bytes"]
    assert loaded["counting_policy"]["partial_files"] is True


def test_cuda_peak_guard_uses_measured_bytes(tmp_path: Path) -> None:
    budget = small_budget(tmp_path)
    assert_cuda_peak(budget, peak_allocated_bytes=10)
    with pytest.raises(BudgetExceeded):
        assert_cuda_peak(budget, peak_allocated_bytes=11)
