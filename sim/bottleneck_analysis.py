#!/usr/bin/env python3
"""NPU 性能瓶颈逐层分析 — 定位最大开销来源"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.mxu import MXUModel
import yaml

# Load config
config = yaml.safe_load(open("config/npu_config.yaml"))
mxu = MXUModel(config)

HIDDEN = 2560
INTERMEDIATE = 9728
NUM_LAYERS = 28

matmuls = [
    ("Q_proj",    1, HIDDEN, 4096),
    ("K_proj",    1, HIDDEN, 256),
    ("V_proj",    1, HIDDEN, 256),
    ("O_proj",    1, 4096, HIDDEN),
    ("FFN_gate",  1, HIDDEN, INTERMEDIATE),
    ("FFN_up",    1, HIDDEN, INTERMEDIATE),
    ("FFN_down",  1, INTERMEDIATE, HIDDEN),
]

print("=" * 75)
print("  NPU Decode (M=1) 逐层瓶颈分析 — v2 tiling-aware")
print("  128×128 @ 1GHz, 51.2 GB/s DRAM, INT4 weight + INT8 act")
print("=" * 75)

total_per_layer = 0
per_matmuls = []

for name, M, K, N in matmuls:
    result = mxu.estimate(M, K, N)
    tiles = result.num_tiles
    us = result.total_cycles / 1000
    ideal_cycles = M * K * N / mxu.macs_per_cycle
    eff_pct = ideal_cycles / result.total_cycles * 100

    per_matmuls.append((name, result.total_cycles, tiles, us, eff_pct))
    total_per_layer += result.total_cycles

print(f"\n{'Op':<12} {'Cycles':>10} {'Tiles':>8} {'μs':>10} {'Eff%':>8} {'%Layer':>8}")
print("-" * 65)
for name, cycles, tiles, us, eff in per_matmuls:
    pct = cycles / total_per_layer * 100
    print(f"{name:<12} {cycles:>10,} {tiles:>8} {us:>10.1f} {eff:>7.1f}% {pct:>7.1f}%")

print("-" * 65)
total_us = total_per_layer / 1000
print(f"{'Layer total':<12} {total_per_layer:>10,} {'':>8} {total_us:>10.1f}")

total_28 = total_per_layer * NUM_LAYERS
total_28_us = total_28 / 1000
tok_s = 1e6 / total_28_us
print(f"{'28 layers':<12} {total_28:>10,} {'':>8} {total_28_us:>10.1f}")
print(f"\n  → {tok_s:.0f} tok/s (M=1 decode)")

# Bandwidth analysis
total_weight_bytes = sum(r.weight_bytes for r in [mxu.estimate(M, K, N) for _, M, K, N in matmuls])
total_weight_gb = total_weight_bytes * NUM_LAYERS / 1e9
bw_needed = total_weight_gb / (total_28_us / 1e6)
bw_available = 51.2 * 0.85

print(f"\n--- Bandwidth ---")
print(f"  Weights/token:  {total_weight_gb:.2f} GB")
print(f"  BW needed:      {bw_needed:.1f} GB/s")
print(f"  BW available:   {bw_available:.1f} GB/s (85% eff)")
print(f"  Headroom:       {bw_available - bw_needed:.1f} GB/s")

# Bottleneck breakdown
ffn_cycles = sum(c for name, c, _, _, _ in per_matmuls if "FFN" in name)
attn_cycles = sum(c for name, c, _, _, _ in per_matmuls if "FFN" not in name)
print(f"\n--- Bottleneck Breakdown ---")
print(f"  Attention (Q/K/V/O):  {attn_cycles/total_per_layer*100:.0f}% of layer")
print(f"  FFN (gate/up/down):   {ffn_cycles/total_per_layer*100:.0f}% of layer")
print(f"  Per-tile overhead:    {mxu.H + mxu.W} fill + {1 + mxu.H} drain = {mxu.H + mxu.W + 1 + mxu.H} cycles")
print(f"  Useful MACs per tile: {mxu.H} (M=1)")
print(f"  Overhead/MAC ratio:   {(mxu.H + mxu.W + 1 + mxu.H) / mxu.H:.1f}x")

# Optimization paths
print(f"\n--- Optimization Paths ---")
print(f"  Path A: Wider array (fewer tiles)")
for h, w, area in [(128, 256, 42), (256, 256, 108), (128, 384, 60)]:
    # Quick scan: change config and re-estimate
    mxu2 = MXUModel({**config, "mxu": {**config["mxu"], "array_height": h, "array_width": w}})
    t = sum(mxu2.estimate(M, K, N).total_cycles for _, M, K, N in matmuls)
    ts = 1e6 / (t * NUM_LAYERS / 1000)
    print(f"    {h}×{w} ({area}mm²): {ts:.0f} tok/s, {area/ts:.1f} mm²/tok")

print(f"\n  Path B: Continuous batching (M≥2)")
for M in [2, 4, 8]:
    t = sum(mxu.estimate(M, K, N).total_cycles for _, _, K, N in matmuls)
    ts = M * 1e6 / (t * NUM_LAYERS / 1000)
    print(f"    M={M}: {ts:.0f} tok/s (same 27mm²)")

print(f"\n  Path C: Reduce FFN intermediate (9728→6912)")
mxu3 = MXUModel(config)
t = 0
for name, _, K, N in matmuls:
    if "FFN" in name and N == 9728:
        N = 6912  # ~70% of original
    elif "FFN" in name and K == 9728:
        K = 6912
    t += mxu3.estimate(1, K, N).total_cycles
ts_c = 1e6 / (t * NUM_LAYERS / 1000)
print(f"    Intermed 9728→6912: {ts_c:.0f} tok/s + model quality tradeoff")
