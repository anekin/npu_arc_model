#!/usr/bin/env python3
"""Generate CV PPA comparison for MobileNetV3-Small across engine types."""

import json
import os
import sys
from copy import deepcopy

from cv.cv_sim import simulate_cv
from engine.ppa_model import AreaModel, PowerModel


BASE_CFG = {
    "mac_engine": {
        "type": None,
        "array_height": 64,
        "array_width": 64,
        "weight_precision_bits": 4,
        "activation_precision_bits": 8,
        "frequency_mhz": 1000,
        "sparsity": 0.0,
    },
    "memory": {
        "bandwidth_bytes_per_cycle": 51.2,
        "dram_efficiency": 0.85,
        "dram_width_bits": 64,
    },
    "sram": {"l1_per_core_kb": 512, "l2_shared_kb": 2048},
    "dma": {"channels": 2},
    "area_model": {},
    "optimizations": {"weight_cache": False},
}

ENGINES = ["systolic", "block", "gmma", "tensor_core", "input_stationary"]


def depthwise_util(layers):
    vals = [layer["mxu_util_pct"] for layer in layers if layer["type"] == "depthwise_conv"]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trace_path = os.path.join(repo_root, "results", "cv", "mobilenetv3_small", "trace.json")
    out_path = os.path.join(repo_root, "results", "cv", "mobilenetv3_small", "cv_int4_comparison.json")
    note_path = os.path.join(repo_root, ".omo", "notepads", "systolic-cv-fix", "learnings.md")

    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    results = []
    for engine in ENGINES:
        cfg = deepcopy(BASE_CFG)
        cfg["mac_engine"]["type"] = engine

        sim = simulate_cv(trace, cfg)
        area_model = AreaModel(cfg)
        power_model = PowerModel(cfg)

        area = area_model.estimate(cfg, engine)["total_mm2"]
        power = power_model.estimate(area_model, cfg, engine)
        total_cycles = sim["total_cycles"]
        fps = 1e9 / total_cycles if total_cycles > 0 else 0.0

        results.append({
            "engine_type": engine,
            "array": "64x64",
            "fps": round(fps, 2),
            "area_mm2": area,
            "power_w": power,
            "sram_spill_mb": round(sim["sram_spill_mb"], 2),
            "depthwise_util_pct": round(depthwise_util(sim["layers"]), 2),
            "label": f"{engine} 64x64 INT4/INT8",
            "total_cycles": total_cycles,
        })

    # Best engine under area constraint <= 30 mm2 by FPS
    eligible = [r for r in results if r["area_mm2"] <= 30.0]
    best = max(eligible, key=lambda r: r["fps"]) if eligible else max(results, key=lambda r: r["fps"])

    output = {
        "config": {
            "array": "64x64",
            "weight_bits": 4,
            "activation_bits": 8,
            "frequency_mhz": 1000,
            "bandwidth_bytes_per_cycle": 51.2,
            "dram_efficiency": 0.85,
        },
        "engines": results,
        "summary": {
            "best_engine": best["engine_type"],
            "best_fps": best["fps"],
            "best_area_mm2": best["area_mm2"],
            "constraint": "area_mm2 <= 30",
            "note": f"{best['engine_type']} delivers highest FPS among engines meeting the area constraint.",
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"Wrote comparison to {out_path}")

    # Append summary to learnings.md
    summary_lines = [
        "",
        "## CV PPA Comparison — MobileNetV3-Small 64x64 INT4/INT8",
        "",
        "| Engine | FPS | Area (mm²) | Power (W) | SRAM spill (MB) | Depthwise util (%) | Total cycles |",
        "|--------|-----|-----------|-----------|-----------------|-------------------|--------------|",
    ]
    for r in results:
        summary_lines.append(
            f"| {r['engine_type']:16} | {r['fps']:6.2f} | {r['area_mm2']:8.1f} | "
            f"{r['power_w']:8.1f} | {r['sram_spill_mb']:14.2f} | "
            f"{r['depthwise_util_pct']:16.2f} | {r['total_cycles']:12d} |"
        )
    summary_lines.extend([
        "",
        f"**Best engine under ≤30 mm²:** {best['engine_type']} at {best['fps']:.2f} FPS "
        f"({best['area_mm2']:.1f} mm², {best['power_w']:.1f} W).",
        "",
    ])

    with open(note_path, "a", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Appended summary to {note_path}")

    # Print verification checks
    block = next(r for r in results if r["engine_type"] == "block")
    systolic = next(r for r in results if r["engine_type"] == "systolic")
    print(f"\nVerification:")
    print(f"  Block 64x64 FPS: {block['fps']:.2f} (highest under 30mm²: "
          f"{block == best and block['area_mm2'] <= 30.0})")
    print(f"  Systolic 64x64 FPS: {systolic['fps']:.2f} (>50: {systolic['fps'] > 50})")


if __name__ == "__main__":
    main()
