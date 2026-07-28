"""Shared pytest fixtures for the engine test suite."""

import copy

import pytest

from config.npu_config import load_config


@pytest.fixture
def engine_config():
    """Return a deep-copied, Wave-1-calibrated NPU config.

    Settings:
      - array: 64x64
      - frequency: 1000 MHz
      - weight precision: INT4
      - memory: LPDDR5-6400 @ 51.2 GB/s
    """
    cfg = copy.deepcopy(load_config())
    cfg["mxu"]["array_height"] = 64
    cfg["mxu"]["array_width"] = 64
    cfg["mxu"]["frequency_mhz"] = 1000
    cfg["mxu"]["weight_precision_bits"] = 4
    cfg["memory"]["type"] = "LPDDR5-6400"
    cfg["memory"]["bandwidth_gbps"] = 51.2
    cfg["memory"]["bandwidth_bytes_per_cycle"] = 51.2
    return cfg
