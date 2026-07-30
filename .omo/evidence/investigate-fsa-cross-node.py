#!/usr/bin/env python3
"""Targeted cross-node comparison: FSA vs block engine.

Fixed configuration:
  - Array: 128 x 128
  - Frequency: 1000 MHz
  - Weight precision: INT4
  - L2 SRAM: 2048 KB
  - External memory: LPDDR5 51.2 GB/s (lpddr5_3b scenario)
  - Process nodes: 7 / 12 / 22 / 28 nm

Uses the actual AreaModel/PowerModel and the same throughput model as the
cross-node DSE (design_space_explorer.evaluate_config), so results are
directly comparable to Todo 14 evidence.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

# Make the simulator modules importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = REPO_ROOT / "sim"
sys.path.insert(0, str(SIM_DIR))

import design_space_explorer as dse
import yaml
from engine.mac_engine import create_engine
from engine.ppa_model import AreaModel, PowerModel

BASE_CONFIG_PATH = SIM_DIR / "config" / "design_space.yaml"
NODES = (7.0, 12.0, 22.0, 28.0)
ENGINES = ("block", "fsa")


def build_config(node_nm: float, engine_type: str) -> dict:
    """Return a deep copy of the design-space base with the fixed comparison config."""
    cfg = copy.deepcopy(yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8")))

    cfg.setdefault("area_model", {})["process_node"] = float(node_nm)

    mac = cfg.setdefault("mac_engine", {})
    mac["type"] = engine_type
    mac["array_height"] = 128
    mac["array_width"] = 128
    mac["frequency_mhz"] = 1000
    mac["weight_precision_bits"] = 4
    mac["activation_precision_bits"] = 8

    cfg.setdefault("sram", {})["l2_shared_kb"] = 2048

    mem = cfg.setdefault("memory", {})
    mem["type"] = "LPDDR5-6400"
    mem["bandwidth_gbps"] = 51.2
    mem["dram_efficiency"] = 0.85

    cfg.setdefault("optimizations", {})["weight_cache"] = False
    cfg["optimizations"]["dma_bw_multiplier"] = 1.0

    return cfg


def layer_cycle_breakdown(cfg: dict) -> dict:
    """Aggregate compute/dma/utilization across the standard Qwen2.5-3B decode layer."""
    engine = create_engine(cfg)
    trace = dse._LLM_TRACE

    total_compute = 0
    total_dma = 0
    total_cycles = 0
    total_macs = 0

    ffn_down_compute = 0
    ffn_down_dma = 0
    ffn_down_cycles = 0
    ffn_down_util = 0.0

    for M, K, N, _layer, name in trace:
        r = engine.estimate(M, K, N)
        total_compute += r.compute_cycles
        total_dma += r.dma_cycles
        total_cycles += r.total_cycles
        total_macs += r.mac_count

        if name == "FFN_down":
            ffn_down_compute = r.compute_cycles
            ffn_down_dma = r.dma_cycles
            ffn_down_cycles = r.total_cycles
            ffn_down_util = r.utilization

    peak = engine.peak_macs_per_cycle
    weighted_util = total_macs / (peak * total_cycles) if total_cycles > 0 else 0.0

    return {
        "layer_compute_cycles": total_compute,
        "layer_dma_cycles": total_dma,
        "layer_total_cycles": total_cycles,
        "layer_utilization": weighted_util,
        "ffn_down_compute_cycles": ffn_down_compute,
        "ffn_down_dma_cycles": ffn_down_dma,
        "ffn_down_total_cycles": ffn_down_cycles,
        "ffn_down_utilization": ffn_down_util,
    }


def evaluate(node_nm: float, engine_type: str) -> dict:
    """Run one (node, engine) evaluation."""
    cfg = build_config(node_nm, engine_type)

    area_model = AreaModel(cfg)
    power_model = PowerModel(cfg)
    ppa = dse.evaluate_config(cfg, area_model, power_model)
    cycles = layer_cycle_breakdown(cfg)

    return {
        "node_nm": node_nm,
        "engine_type": engine_type,
        "tok_per_s": ppa.tok_s,
        "area_mm2": ppa.area_mm2,
        "power_w": ppa.power_w,
        "efficiency_tok_per_watt": ppa.efficiency_tok_per_watt,
        "efficiency_tok_per_mm2": ppa.efficiency_tok_per_mm2,
        **cycles,
    }


def main() -> int:
    results: list[dict] = []
    for node in NODES:
        for engine in ENGINES:
            results.append(evaluate(node, engine))

    # Compute per-node ratios (block / fsa).
    ratios: list[dict] = []
    by_key = {(r["node_nm"], r["engine_type"]): r for r in results}
    for node in NODES:
        block = by_key[(node, "block")]
        fsa = by_key[(node, "fsa")]
        ratios.append(
            {
                "node_nm": node,
                "block_area_mm2": block["area_mm2"],
                "fsa_area_mm2": fsa["area_mm2"],
                "area_ratio_block_over_fsa": round(block["area_mm2"] / fsa["area_mm2"], 3),
                "block_tok_per_s": block["tok_per_s"],
                "fsa_tok_per_s": fsa["tok_per_s"],
                "tok_ratio_block_over_fsa": round(block["tok_per_s"] / fsa["tok_per_s"], 3),
            }
        )

    output = {
        "schema_version": "1",
        "description": "FSA vs block cross-node single-config comparison (lpddr5_3b)",
        "fixed_config": {
            "array_height": 128,
            "array_width": 128,
            "frequency_mhz": 1000,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
            "l2_shared_kb": 2048,
            "memory_type": "LPDDR5-6400",
            "bandwidth_gbps": 51.2,
            "weight_cache": False,
        },
        "results": results,
        "ratios": ratios,
    }

    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Print comparison table.
    print("\n=== FSA vs block cross-node comparison (lpddr5_3b) ===\n")
    header = (
        f"{'Node':>6s} | {'Engine':>8s} | {'tok/s':>7s} | "
        f"{'area_mm2':>9s} | {'power_w':>8s} | {'compute_c':>10s} | "
        f"{'dma_c':>10s} | {'util':>6s}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['node_nm']:>6.0f} | {r['engine_type']:>8s} | {r['tok_per_s']:>7.1f} | "
            f"{r['area_mm2']:>9.1f} | {r['power_w']:>8.1f} | "
            f"{r['layer_compute_cycles']:>10d} | {r['layer_dma_cycles']:>10d} | "
            f"{r['layer_utilization']:>6.3f}"
        )

    print("\n=== Ratios (block / FSA) ===\n")
    print(f"{'Node':>6s} | {'area_ratio':>11s} | {'tok_ratio':>10s}")
    print("-" * 35)
    for r in ratios:
        print(
            f"{r['node_nm']:>6.0f} | {r['area_ratio_block_over_fsa']:>11.3f} | "
            f"{r['tok_ratio_block_over_fsa']:>10.3f}"
        )

    print(f"\nEvidence saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
