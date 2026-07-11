#!/usr/bin/env python3
"""DMA 瓶颈分析与改进方案评估 — 在权重缓存基础上叠加"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from models.mxu import MXUModel
import yaml

config = yaml.safe_load(open("config/npu_config.yaml"))
H, W = 128, 128
HIDDEN, INTER = 2560, 9728
NUM_LAYERS = 28

def calc(bw_factor=1.0, weight_bits=4, sparsity=1.0):
    """
    bw_factor: 1.0 = 51.2 GB/s, 2.0 = 102.4 GB/s
    weight_bits: 4 (INT4), 2 (INT2), 1.58 (ternary)
    sparsity: 1.0 (dense), 0.5 (2:4 structured)
    """
    eff_bw = 51.2 * 0.85 * bw_factor
    data_per_tile = math.ceil(H * W * weight_bits / 8) * sparsity * 2  # 2 matrices
    act_per_tile = math.ceil(1 * H * 8 / 8)  # M=1, INT8 act
    dma = (data_per_tile + act_per_tile) / eff_bw
    compute = 2 * (1 + W) + 1  # 259
    bottleneck = max(dma, compute)
    
    K_tiles = math.ceil(HIDDEN / H)
    N_tiles = math.ceil(INTER / W)
    fill = H + W
    drain = 1 + H
    first_cold = dma + compute
    if N_tiles >= 2:
        per_K = fill + first_cold + (N_tiles - 1) * bottleneck + drain
    else:
        per_K = fill + first_cold + drain
    total = int(K_tiles * per_K)
    
    return {
        "dma_per_tile": round(dma),
        "compute_per_tile": compute,
        "bottleneck": round(bottleneck),
        "bottleneck_is": "DMA" if dma > compute else "compute",
        "gate_up_total": total,
        "gate_up_us": total / 1000,
    }

# Baseline: with weight cache, no DMA improvement
base = calc()
print("=" * 65)
print("  DMA 改进方案评估（权重缓存基础上叠加）")
print("=" * 65)

configs = [
    ("Baseline (cache only)",      1.0, 4, 1.0, ""),
    ("128-bit LPDDR5 (102GB/s)",   2.0, 4, 1.0, "双倍 pin，功耗 +40%"),
    ("4 DMA channels",             2.0, 4, 1.0, "等效双倍带宽，DMA 引擎面积 +50%"),
    ("INT2 权重",                  1.0, 2, 1.0, "需重训练，精度损失 ~2%"),
    ("2:4 结构化稀疏",             1.0, 4, 0.5, "需稀疏训练，精度损失 <1%"),
    ("INT2 + 128-bit",             2.0, 2, 1.0, "双重叠加"),
    ("INT2 + 2:4 稀疏",            1.0, 2, 0.5, "激进压缩"),
]

print(f"\n{'方案':<30} {'DMA':>6} {'Comp':>6} {'瓶颈':>8} {'Gate+Up':>10} {'改善':>8}")
print("-" * 75)

base_total = base["gate_up_total"]

for name, bw, bits, sp, note in configs:
    r = calc(bw, bits, sp)
    improvement = (base_total - r["gate_up_total"]) / base_total * 100
    flag = "✅" if r["bottleneck_is"] == "compute" else "⚠️ DMA"
    print(f"{name:<30} {r['dma_per_tile']:>4}c {r['compute_per_tile']:>4}c {flag:>8} {r['gate_up_total']:>8,}c  {improvement:>+5.0f}%")
    if note:
        print(f"  {'':30} {note}")

# Full model projection
print(f"\n  Full model projection:")
print(f"  {'方案':<30} {'FFN/层':>10} {'全模型':>10} {'tok/s':>8} {'累计':>8}")
print(f"  {'-'*65}")

down = MXUModel(config).estimate(1, INTER, HIDDEN).total_cycles
attn = sum(MXUModel(config).estimate(1, *dims).total_cycles 
           for dims in [(HIDDEN, 4096), (HIDDEN, 256), (HIDDEN, 256), (4096, HIDDEN)])

prev_tok = 16  # baseline no cache
for name, bw, bits, sp, _ in configs[:6]:  # skip last two combos for brevity
    r = calc(bw, bits, sp)
    ffn = r["gate_up_total"] + down
    layer = attn + ffn
    tok = 1e6 / (layer * NUM_LAYERS / 1000)
    cumulative = f"+{tok - 16:.0f}" if tok > 16 else ""
    print(f"  {name:<30} {ffn/1000:>8.0f}μs {layer*28/1000:>8.0f}μs {tok:>6.0f}  {cumulative:>8}")
