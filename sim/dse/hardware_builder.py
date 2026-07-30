"""Materialize a normalized hardware config dict from axis values."""

from __future__ import annotations

import copy
from typing import Any

_MEMORY_TYPE_LABEL = {
    "lpddr5": "LPDDR5-6400",
    "lpddr5x": "LPDDR5X-8533",
    "hbm2e": "HBM2e",
    "hbm3": "HBM3",
    "on_chip_3d_dram": "OnChip3D",
}

_MEMORY_EFFICIENCY = {
    "lpddr5": 0.85,
    "lpddr5x": 0.85,
    "hbm2e": 0.85,
    "hbm3": 0.85,
    "on_chip_3d_dram": 1.0,
}


def build_hardware_config(base_config: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any]:
    """Merge axis values into the base hardware config."""
    cfg = copy.deepcopy(base_config)
    mem_type = combo["memory_type"]

    mac = cfg.setdefault("mac_engine", {})
    mac["type"] = combo["engine"]
    mac["array_height"] = combo["array_height"]
    mac["array_width"] = combo["array_width"]
    mac["frequency_mhz"] = combo["frequency_mhz"]
    mac["weight_precision_bits"] = combo["weight_precision_bits"]
    mac["activation_precision_bits"] = combo["activation_precision_bits"]
    mac["dataflow"] = "output_stationary" if combo["engine"] == "os_systolic" else "weight_stationary"
    mac["double_buffer"] = True
    mac["ops_per_mac"] = 2

    mem = cfg.setdefault("memory", {})
    mem["type"] = _MEMORY_TYPE_LABEL.get(mem_type, mem_type)
    mem["bandwidth_gbps"] = combo["bandwidth_gbps"]
    mem["dram_efficiency"] = _MEMORY_EFFICIENCY.get(mem_type, 0.85)
    mem["dram_width_bits"] = combo["dram_width_bits"]
    mem.setdefault("tRC_cycles", 48)
    mem.setdefault("tRAS_cycles", 42)
    mem.setdefault("refresh_overhead_percent", 3.0)

    sram = cfg.setdefault("sram", {})
    sram["l2_shared_kb"] = combo["sram_l2_kb"]
    sram.setdefault("l1_per_core_kb", 512)
    sram.setdefault("banks", 16)
    sram.setdefault("read_width_bits", 256)
    sram.setdefault("write_width_bits", 256)

    onchip = cfg.setdefault("on_chip_memory", {})
    onchip["capacity_gb"] = combo["on_chip_capacity_gb"]
    onchip["bandwidth_gbps"] = combo["on_chip_bandwidth_gbps"]

    opts = cfg.setdefault("optimizations", {})
    opts["weight_cache"] = combo["weight_cache"]
    opts.setdefault("dma_bw_multiplier", 1.0)

    # Propagate process_node from DSE axis to area_model config.
    # Backward compatibility: combos without process_node fall back to
    # base_config["area_model"]["process_node"] (default 7).
    if "process_node" in combo:
        am = cfg.setdefault("area_model", {})
        am["process_node"] = combo["process_node"]

    cfg.setdefault("version", "2")
    return cfg


__all__ = ["build_hardware_config"]
