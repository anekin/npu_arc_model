#!/usr/bin/env python3
"""权重缓存建模 — FFN gate/up 共享输入激活，PE 双 weight reg 避免重复 pipeline"""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.mxu import MXUModel
import yaml

config = yaml.safe_load(open("config/npu_config.yaml"))
mxu = MXUModel(config)

H = mxu.H  # 128
W = mxu.W
eff_bw = mxu.eff_bw  # 43.5 bytes/cycle
HIDDEN = 2560
INTER = 9728
NUM_LAYERS = 28


def estimate_weight_cache(M, K, N):
    """
    Estimate gate+up combined with weight caching.
    
    Hardware: each PE has dual weight registers (reg_w0, reg_w1).
    Loads W_gate and W_up for a (k,n) tile together, computes gate,
    switches reg, computes up — no pipeline drain/fill between.
    
    Returns: total cycles, breakdown dict
    """
    K_tiles = math.ceil(K / H)
    N_tiles = math.ceil(N / W)
    
    # Per dual-tile: load 2×8KB weights + 1×128B activation
    dual_weight_bytes = 2 * math.ceil(H * W * mxu.w_bits / 8)  # 16KB
    dual_act_bytes = math.ceil(M * H * mxu.a_bits / 8)         # 128B
    dual_dma = (dual_weight_bytes + dual_act_bytes) / eff_bw
    
    # Compute per dual-tile: gate(129) + switch(1) + up(129) = 259 cycles
    # (pipeline already filled, so just M=1 pass × 2 matmuls)
    per_matm_compute = M + W  # M rows through W columns = drain for one matmul
    dual_compute = 2 * per_matm_compute + 1  # +1 for weight reg switch
    
    # Pipeline overhead: fill once per K-tile, drain once per K-tile
    fill = H + W   # 256
    drain = M + H  # 129
    
    # Per K-tile with double-buffering:
    # - Fill: 256
    # - First dual-tile: DMA + compute = cold start
    # - Remaining: max(DMA, compute) overlapped
    # - Drain: 129
    
    bottleneck = max(dual_dma, dual_compute)
    first_cold = dual_dma + dual_compute
    
    if N_tiles >= 2:
        per_Ktile = fill + first_cold + (N_tiles - 1) * bottleneck + drain
    else:
        per_Ktile = fill + first_cold + drain
    
    total = int(K_tiles * per_Ktile)
    
    return {
        "total_cycles": total,
        "total_us": total / 1000,
        "K_tiles": K_tiles,
        "N_tiles": N_tiles,
        "dual_tiles": K_tiles * N_tiles,
        "per_dual_dma": round(dual_dma, 1),
        "per_dual_compute": dual_compute,
        "bottleneck": "DMA" if dual_dma > dual_compute else "compute",
        "per_Ktile": per_Ktile,
        "dual_weight_kb": dual_weight_bytes / 1024,
    }


# ── Current (no caching) ──
print("=" * 65)
print("  权重缓存性能评估 — FFN gate+up 合并建模")
print("=" * 65)

gate_current = mxu.estimate(1, HIDDEN, INTER)
up_current = mxu.estimate(1, HIDDEN, INTER)
down_current = mxu.estimate(1, INTER, HIDDEN)

ffn_current = gate_current.total_cycles + up_current.total_cycles + down_current.total_cycles
gate_up_current = gate_current.total_cycles + up_current.total_cycles

print(f"\n  Current (no cache):")
print(f"    Gate:       {gate_current.total_cycles:>10,} cycles ({gate_current.total_cycles/1000:.0f} μs)")
print(f"    Up:         {up_current.total_cycles:>10,} cycles ({up_current.total_cycles/1000:.0f} μs)")
print(f"    Gate+Up:    {gate_up_current:>10,} cycles ({gate_up_current/1000:.0f} μs)")
print(f"    Down:       {down_current.total_cycles:>10,} cycles ({down_current.total_cycles/1000:.0f} μs)")
print(f"    FFN total:  {ffn_current:>10,} cycles ({ffn_current/1000:.0f} μs)")

# ── With weight cache ──
cached = estimate_weight_cache(1, HIDDEN, INTER)

ffn_cached = cached["total_cycles"] + down_current.total_cycles

print(f"\n  With weight cache (PE dual reg):")
print(f"    Gate+Up:    {cached['total_cycles']:>10,} cycles ({cached['total_us']:.0f} μs)")
print(f"    Per tile:   DMA={cached['per_dual_dma']:.0f}, compute={cached['per_dual_compute']}")
print(f"    Bottleneck: {cached['bottleneck']}")
print(f"    Dual tiles: {cached['dual_tiles']} ({cached['K_tiles']}K × {cached['N_tiles']}N)")
print(f"    Down:       {down_current.total_cycles:>10,} cycles ({down_current.total_cycles/1000:.0f} μs)")
print(f"    FFN total:  {ffn_cached:>10,} cycles ({ffn_cached/1000:.0f} μs)")

# ── Full layer comparison ──
# Build one layer cost
# Without cache, from bottleneck_analysis:
# Q:246K, K:15.6K, V:15.6K, O:246K, Gate:585K, Up:585K, Down:585K = 2278K

q_cycles = mxu.estimate(1, HIDDEN, 4096).total_cycles
k_cycles = mxu.estimate(1, HIDDEN, 256).total_cycles
v_cycles = mxu.estimate(1, HIDDEN, 256).total_cycles
o_cycles = mxu.estimate(1, 4096, HIDDEN).total_cycles

attn = q_cycles + k_cycles + v_cycles + o_cycles
layer_current = attn + ffn_current
layer_cached = attn + ffn_cached

saving_per_layer = layer_current - layer_cached
saving_pct = saving_per_layer / layer_current * 100

print(f"\n  Per-layer comparison:")
print(f"    Attention (Q/K/V/O): {attn:>10,} cycles ({attn/1000:.0f} μs)")
print(f"    FFN current:         {ffn_current:>10,} cycles")
print(f"    FFN cached:          {ffn_cached:>10,} cycles")
print(f"    Saving:              {saving_per_layer:>10,} cycles ({saving_pct:.0f}%)")

total_current = layer_current * NUM_LAYERS
total_cached = layer_cached * NUM_LAYERS
tok_current = 1e6 / (total_current / 1000)
tok_cached = 1e6 / (total_cached / 1000)

print(f"\n  Full model (28 layers):")
print(f"    Current:  {total_current/1000:>10.0f} μs → {tok_current:.0f} tok/s")
print(f"    Cached:   {total_cached/1000:>10.0f} μs → {tok_cached:.0f} tok/s")
print(f"    Speedup:  {tok_cached/tok_current:.2f}×")
print(f"    +{tok_cached - tok_current:.0f} tok/s")

# ── HW cost ──
print(f"\n  Hardware cost:")
print(f"    PE weight reg: 4-bit → 8-bit (+4 bits per PE)")
print(f"    PE area:       ~+15% (1 extra FF + MUX per PE)")
print(f"    Array area:    8mm² → ~9.2mm²")
print(f"    Total chip:    27mm² → ~28mm²")

# ── Combined with batching ──
print(f"\n  Combined with batching (M=2):")
# Quick batch estimate
batch_current = sum(mxu.estimate(2, K, N).total_cycles for _, K, N in [
    ("Q", HIDDEN, 4096), ("K", HIDDEN, 256), ("V", HIDDEN, 256),
    ("O", 4096, HIDDEN),
    ("Gate", HIDDEN, INTER), ("Up", HIDDEN, INTER), ("Down", INTER, HIDDEN)
])
tok_batch_current = 2 * 1e6 / (batch_current * NUM_LAYERS / 1000)

# M=2 with cache: gate+up share, down separate
# For M=2, the gate+up combined estimate:
cached_m2 = estimate_weight_cache(2, HIDDEN, INTER)
attn_m2 = mxu.estimate(2, HIDDEN, 4096).total_cycles + \
          mxu.estimate(2, HIDDEN, 256).total_cycles + \
          mxu.estimate(2, HIDDEN, 256).total_cycles + \
          mxu.estimate(2, 4096, HIDDEN).total_cycles
down_m2 = mxu.estimate(2, INTER, HIDDEN).total_cycles
layer_m2 = attn_m2 + cached_m2["total_cycles"] + down_m2
tok_batch_cached = 2 * 1e6 / (layer_m2 * NUM_LAYERS / 1000)

print(f"    Batching only:     ~{tok_batch_current:.0f} tok/s")
print(f"    Batching + cache:  ~{tok_batch_cached:.0f} tok/s")
