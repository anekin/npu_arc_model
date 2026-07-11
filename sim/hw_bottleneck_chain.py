#!/usr/bin/env python3
"""纯硬件瓶颈链分析 — 逐层拆解每 token 耗时构成"""

import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models.mxu import MXUModel
import yaml

config = yaml.safe_load(open("config/npu_config.yaml"))
H, W = 128, 128
HIDDEN, INTER = 2560, 9728
NUM_LAYERS = 28
eff_bw = 51.2 * 0.85

mxu = MXUModel(config)

print("=" * 70)
print("  纯硬件瓶颈链分析 — 128×128, INT4, 51.2GB/s DRAM")
print("=" * 70)

# ── 1. 原始瓶颈（无任何优化）──
print("\n  ═══ Level 0: 原始基线 ═══")
gate = mxu.estimate(1, HIDDEN, INTER)
up = mxu.estimate(1, HIDDEN, INTER)
down = mxu.estimate(1, INTER, HIDDEN)

print(f"  FFN per matmul: {gate.total_cycles/1000:.1f}μs, tiles={gate.num_tiles}")
print(f"  Per-tile breakdown: fill={H+W}c drain={1+H}c = {H+W+1+H}c total")
print(f"  瓶颈: pipeline fill/drain — 385c overhead vs 128 MAC useful work")
print(f"  效率: {128/(H+W+1+H)*100:.1f}%")

# ── 2. 权重缓存（PE 双 weight reg）──
print(f"\n  ═══ Level 1: +权重缓存 ═══")

K_tiles = math.ceil(HIDDEN/H)
N_tiles = math.ceil(INTER/W)

dual_dma = (math.ceil(H*W*4/8) * 2 + math.ceil(H*8/8)) / eff_bw
dual_compute = 2*(1+W) + 1  # 259

fill = H+W
drain = 1+H

first_cold = dual_dma + dual_compute
bottleneck0 = max(dual_dma, dual_compute)
per_K = fill + first_cold + (N_tiles-1)*bottleneck0 + drain
gate_up_cached = K_tiles * per_K

print(f"  Gate+Up: {585391*2/1000:.0f}μs → {gate_up_cached/1000:.0f}μs (节省 {585391*2 - gate_up_cached:.0f} cycles)")
print(f"  新瓶颈: DMA={dual_dma:.0f}c > compute={dual_compute}c")
print(f"  每 dual-tile: DMA {dual_dma:.0f}c 等待权重 + {dual_compute}c 计算")
print(f"  瓶颈占比: DMA {(dual_dma-dual_compute)/dual_dma*100:.0f}% > compute")

# ── 3. DMA 改进（128-bit / 4ch）──
print(f"\n  ═══ Level 2: +DMA 翻倍 ═══")

dual_dma2 = dual_dma / 2
dual_compute2 = dual_compute  # 259 unchanged
bottleneck2 = max(dual_dma2, dual_compute2)
per_K2 = fill + (dual_dma2 + dual_compute2) + (N_tiles-1)*bottleneck2 + drain
gate_up_l2 = K_tiles * per_K2

print(f"  Gate+Up: {gate_up_cached/1000:.0f}μs → {gate_up_l2/1000:.0f}μs (节省 {gate_up_cached - gate_up_l2:.0f} cycles)")
print(f"  瓶颈翻转: DMA={dual_dma2:.0f}c < compute={dual_compute2}c → compute-bound")
print(f"  新瓶颈: pipeline drain = 2×{1+W} = {2*(1+W)} cycles per dual-tile")
print(f"  瓶颈占比: drain {2*(1+W)}c / {bottleneck2}c = {2*(1+W)/bottleneck2*100:.0f}%")

# ── 4. 拆解 compute 瓶颈 ──
print(f"\n  ═══ Level 3: compute 瓶颈内部 ═══")
print(f"  Per dual-tile compute = 2×(M + W) + 1 = 2×({1}+{W}) + 1 = {dual_compute2}")
print(f"    ├─ Gate drain:  M+H = {1}+{H} = {1+H} cycles  ← 结果从阵列底部流出")
print(f"    ├─ 权重切换:     1 cycle")
print(f"    └─ Up drain:    M+H = {1}+{H} = {1+H} cycles")

# Can we reduce H? 
print(f"\n  M=1 drain = M+H. 减少 H 能减少 drain。但会增加 K_tiles。")
for h_test in [128, 96, 64]:
    kt = math.ceil(HIDDEN / h_test)
    nt = N_tiles
    n_tiles_total = kt * nt
    drain_test = 1 + h_test
    compute_test = 2 * drain_test + 1
    dma_per = (math.ceil(h_test*W*4/8)*2 + math.ceil(h_test*8/8)) / eff_bw / 2  # with DMA fix
    bottleneck_test = max(dma_per, compute_test)
    fill_test = h_test + W
    per_K_test = fill_test + (dma_per + compute_test) + (nt-1)*bottleneck_test + drain_test
    total_test = kt * per_K_test
    tok = 1e6 / ((total_test + down.total_cycles + 
                   sum(mxu.estimate(1, k, n).total_cycles for k,n in 
                       [(HIDDEN,4096),(HIDDEN,256),(HIDDEN,256),(4096,HIDDEN)])) 
                  * NUM_LAYERS / 1000)
    print(f"    H={h_test}: tiles={kt}×{nt}={n_tiles_total}, drain={drain_test}c, "
          f"compute={compute_test}c, gate+up={total_test/1000:.0f}μs → {tok:.0f} tok/s")

# ── 5. 物理极限 ──
print(f"\n  ═══ Level 4: 物理极限 ═══")
ideal_drain = 0  # if drain were zero
ideal_compute = 1  # just weight switch
ideal_dual = ideal_compute
# But still need K_tiles×N_tiles tiles
ideal_per_K = fill + (0 + 1) + (N_tiles-1)*1 + 0  # ideal: no DMA wait, instant drain
ideal_total = K_tiles * ideal_per_K
print(f"  Gate+Up theoretical minimum: {ideal_total/1000:.0f}μs "
      f"(tiles={K_tiles*N_tiles}, fill={fill}c × {K_tiles} K-tiles)")
# What if array were monolithic (1 tile = entire matmul)?
ideal_mono = fill + (2*(1+INTER))  # fill once, drain gate + drain up
print(f"  Monolithic阵列 min: {ideal_mono/1000:.0f}μs (1 tile, fill once, drain twice)")

# ── 6. 总结 ──
print(f"\n  ═══════════════════════════════════════════")
print(f"  硬件瓶颈链总结")
print(f"  ═══════════════════════════════════════════")
print(f"")
print(f"  瓶颈            来源              改进方案          剩余")  
print(f"  ─────────────────────────────────────────────────────")
print(f"  1. pipeline fill/drain  tile开销    权重缓存(PE双reg)  → DMA等待")
print(f"  2. DMA 等待          双倍权重加载   +128bit/DMA通道    → drain等待")
print(f"  3. pipeline drain    M+H=129c/tile  物理极限(结构性的)  → 无法硬件消除")
print(f"")
print(f"  结论: 纯硬件路径走到头是 ~24 tok/s。剩下的 drain 是 systolic")
print(f"  array 在 M=1 下的物理极限 — 信号必须走 128 列才能出结果。")
print(f"  突破需要: output-stationary 数据流 或 不同 microarchitecture。")
