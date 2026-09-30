"""Actual-machine environment probing for the Phase-0 hardware gate."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .budget import Budget, build_budget_report
from .hashing import canonical_json_hash, sha256_file


RELEVANT_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "PyYAML",
    "pydantic",
    "jsonschema",
    "requests",
    "httpx",
    "pytest",
    "hypothesis",
    "matplotlib",
    "seaborn",
)


def parse_smi_csv(text: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {row!r}")
        index, name, memory_mib, driver, compute_capability = (
            value.strip() for value in row
        )
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(memory_mib),
                "memory_total_bytes": int(memory_mib) * 1024 * 1024,
                "driver_version": driver,
                "compute_capability": compute_capability,
            }
        )
    return devices


def parse_driver_cuda_version(text: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    return match.group(1) if match else None


def _command_output(command: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def _windows_registry_value(key_path: str, value_name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _kind = winreg.QueryValueEx(key, value_name)
        return str(value)
    except OSError:
        return None


def _memory_snapshot() -> dict[str, int | None]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return {"total_bytes": int(memory.total), "available_bytes": int(memory.available)}
    except ImportError:
        return {"total_bytes": None, "available_bytes": None}


def _package_versions() -> tuple[dict[str, str | None], str]:
    versions: dict[str, str | None] = {}
    for name in RELEVANT_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    all_versions = sorted(
        (dist.metadata.get("Name", "unknown").lower(), dist.version)
        for dist in importlib.metadata.distributions()
    )
    fingerprint = canonical_json_hash(
        {"python_executable": sys.executable, "distributions": all_versions}
    )
    return versions, fingerprint


def _cuda_probe(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {
            "attempted": False,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "error": "torch.cuda.is_available() is false",
        }
    try:
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        first = torch_module.empty(
            (1024, 1024), dtype=torch_module.float16, device="cuda"
        )
        second = first.square()
        torch_module.cuda.synchronize()
        result = {
            "attempted": True,
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved()),
            "error": None,
        }
        del first, second
        torch_module.cuda.empty_cache()
        return result
    except Exception as exc:  # pragma: no cover - hardware-specific error path
        return {
            "attempted": True,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_environment_report(workspace_root: Path, budget: Budget) -> dict[str, Any]:
    workspace_root = Path(workspace_root).resolve()
    registry_cpu_name = _windows_registry_value(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString"
    )
    cpu_name = registry_cpu_name or platform.processor()
    os_product = _windows_registry_value(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "ProductName"
    )
    os_display_version = _windows_registry_value(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "DisplayVersion"
    )
    os_build = _windows_registry_value(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "CurrentBuild"
    )
    memory = _memory_snapshot()
    disk = shutil.disk_usage(workspace_root)
    package_versions, environment_hash = _package_versions()

    smi_path = shutil.which("nvidia-smi")
    smi_devices: list[dict[str, Any]] = []
    smi_full = ""
    smi_error: str | None = None
    if smi_path:
        query = [
            smi_path,
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
        code, output, error = _command_output(query)
        if code == 0:
            try:
                smi_devices = parse_smi_csv(output)
            except ValueError as exc:
                smi_error = str(exc)
        else:
            smi_error = error.strip() or f"nvidia-smi query exit code {code}"
        _full_code, smi_full, _full_error = _command_output([smi_path])
    else:
        smi_error = "nvidia-smi not found"

    torch_info: dict[str, Any]
    cuda_probe: dict[str, Any]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
        device_memory = [
            int(torch.cuda.get_device_properties(i).total_memory)
            for i in range(device_count)
        ]
        torch_path = Path(torch.__file__).resolve()
        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "path": str(torch_path),
            "import_file_sha256": sha256_file(torch_path),
            "cuda_available": cuda_available,
            "device_count": device_count,
            "device_names": device_names,
            "device_total_memory_bytes": device_memory,
        }
        cuda_probe = _cuda_probe(torch)
    except Exception as exc:  # pragma: no cover - exercised only on broken hosts
        torch_info = {
            "installed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "device_total_memory_bytes": [],
        }
        cuda_probe = {
            "attempted": False,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "error": "PyTorch import/probe failed",
        }

    budget_report = build_budget_report(budget)
    peak_allocated = cuda_probe.get("peak_allocated_bytes")
    checks = [
        {
            "name": "python_3_11",
            "passed": sys.version_info[:2] == (3, 11),
            "detail": platform.python_version(),
        },
        {
            "name": "amd_ryzen_7_cpu",
            "passed": bool(cpu_name and "Ryzen 7" in cpu_name),
            "detail": cpu_name,
        },
        {
            "name": "minimum_16gb_class_ram",
            "passed": bool(memory["total_bytes"] and memory["total_bytes"] >= 15_000_000_000),
            "detail": memory["total_bytes"],
        },
        {
            "name": "single_nvidia_smi_gpu",
            "passed": len(smi_devices) == 1,
            "detail": smi_devices,
        },
        {
            "name": "rtx_3060_visible",
            "passed": (
                len(smi_devices) == 1
                and "RTX 3060" in str(smi_devices[0].get("name"))
                and torch_info.get("device_count") == 1
                and "RTX 3060" in str(torch_info.get("device_names", [""])[0])
            ),
            "detail": {
                "nvidia_smi": smi_devices,
                "torch": torch_info.get("device_names"),
            },
        },
        {
            "name": "torch_cuda_available",
            "passed": torch_info.get("cuda_available") is True,
            "detail": torch_info.get("cuda_available"),
        },
        {
            "name": "cuda_peak_probe_under_cap",
            "passed": (
                isinstance(peak_allocated, int)
                and peak_allocated <= budget.max_cuda_allocated_bytes
                and cuda_probe.get("error") is None
            ),
            "detail": cuda_probe,
        },
        {
            "name": "workspace_budget_and_reserve",
            "passed": (
                budget_report["measurements"]["workspace_bytes"]
                < budget.max_workspace_bytes
                and budget_report["measurements"]["output_reserve_preserved"]
            ),
            "detail": budget_report["measurements"],
        },
    ]
    missing_future_packages = [
        name for name, version in package_versions.items() if version is None
    ]
    warnings_list: list[str] = []
    if memory["available_bytes"] is not None and memory["available_bytes"] < 2_000_000_000:
        warnings_list.append(
            "Available RAM was below 2 GB at probe time; close unrelated processes before memory-intensive phases."
        )
    if missing_future_packages:
        warnings_list.append(
            "Some later-phase packages are not installed in the reused system environment: "
            + ", ".join(missing_future_packages)
        )

    return {
        "schema_version": 1,
        "platform": {
            "os": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "windows_product_name": os_product,
            "windows_display_version": os_display_version,
            "windows_build": os_build,
            "cpu_model": cpu_name,
            "logical_cpu_count": os.cpu_count(),
            "ram_total_bytes": memory["total_bytes"],
            "ram_available_bytes": memory["available_bytes"],
        },
        "gpu": {
            "nvidia_smi_path": smi_path,
            "nvidia_smi_error": smi_error,
            "devices": smi_devices,
            "driver_supported_cuda_version": parse_driver_cuda_version(smi_full),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "environment_kind": "reused_preinstalled" if sys.prefix == sys.base_prefix else "virtual_environment",
            "environment_hash": environment_hash,
        },
        "pytorch": torch_info,
        "cuda_memory_probe": cuda_probe,
        "packages": package_versions,
        "workspace": {
            "path": str(workspace_root),
            "size_bytes": budget_report["measurements"]["workspace_bytes"],
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "budget": budget_report,
        "checks": checks,
        "warnings": warnings_list,
        "phase0_environment_gate": "PASS" if all(item["passed"] for item in checks) else "FAIL",
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
