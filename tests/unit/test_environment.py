from __future__ import annotations

import pytest

from bces.utils.environment import parse_driver_cuda_version, parse_smi_csv


pytestmark = pytest.mark.phase0


def test_parse_nvidia_smi_csv() -> None:
    devices = parse_smi_csv("0, NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 596.08, 8.6\n")
    assert devices == [
        {
            "index": 0,
            "name": "NVIDIA GeForce RTX 3060 Laptop GPU",
            "memory_total_mib": 6144,
            "memory_total_bytes": 6144 * 1024 * 1024,
            "driver_version": "596.08",
            "compute_capability": "8.6",
        }
    ]


def test_parse_driver_cuda_version() -> None:
    assert parse_driver_cuda_version("Driver Version: 596.08 CUDA Version: 13.2") == "13.2"
    assert parse_driver_cuda_version("no cuda field") is None


def test_unexpected_smi_shape_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_smi_csv("NVIDIA RTX 3060, 6144")

