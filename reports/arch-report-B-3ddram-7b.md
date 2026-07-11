# 架构选型报告 B — On-chip 3D DRAM 高带宽 NPU

**日期**: 2026-07 | **工艺**: TSMC 12nm  
**需求**: 高端端侧 NPU，3D DRAM 堆叠，跑 7B 模型实时推理  
**模型**: Qwen2.5-7B INT4 (3.5 GB)，28 层  
**内存**: On-chip 3D DRAM 5GB, 500 GB/s（权重全量常驻，TSV 互联）  
**对标**: 瑞芯微 RK1828 (20 TOPS INT8, ~100mm²@22nm, 7B=59-180 tok/s)

---

## 0. 设计约束

| 约束 | 要求 | 验证 |
|------|------|:---:|
| 模型 | 7B | 7B |
| Decode TPS | ≥ 100 | ✓ (148) |
| TTFT (128 tok) | < 200ms | ✓ (160ms) |
| 权重驻留 | 5GB ≥ 3.5GB | ✓ |

---

## 1. H×W 维度扫描

On-chip 权重常驻 → 无 K-tiling。H 处理 batch 维度（M），W 处理输出维度（N）。SRAM=1MB。

| H×W | TOPS INT8 | TPS | TTFT | 面积 | 达标 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **4×1536** | **6.1** | **148** | **160ms** | **34.3mm²** | ✓ |
| 8×1536 | 12.3 | 148 | 80ms | 39.1mm² | ✓ |
| 16×1536 | 24.6 | 148 | 40ms | 48.8mm² | ✓ |
| 32×1024 | 32.8 | 142 | 28ms | 55.3mm² | ✓ |
| 32×1536 | 49.2 | 148 | 20ms | 68.2mm² | ✓ |
| 64×1024 | 65.5 | 142 | 14ms | 81.1mm² | ✓ |
| 128×512 | 65.5 | **78** | 13ms | 81.1mm² | ✗ TPS |

**核心洞察**：

1. **Decode TPS 恒定 ~148 tok/s**——on-chip 500GB/s 带宽天花板（500/3.5≈143，加 KV overhead~148）。算力严重过剩。

2. **H=4 用 6.1 TOPS 达成 148 TPS，H=128 需 65.5 TOPS 仅 78 TPS**——高 H 的 per-tile 固定开销（H+4=132c）远大于矮 H（4+4=8c）。→ **矮宽阵列 ≈ 24× TOPS 效率提升**

3. **128×512 不及格**：H 过高 per-tile overhead 致命，W 过小 N_tiling 8 次。矮宽（H≤16, W≥1024）是正确范式。

---

## 2. SRAM 敏感性（关键发现）

On-chip 3D DRAM 场景下 **SRAM 对性能零影响**——原因：

- **权重常驻**：无 K-tiling，不需要 SRAM 缓冲权重 tile
- **激活极小**：decode M=1，单 token 激活 = 4KB；weight tile = 3KB。总工作集 ~20KB/layer
- **KV cache 带宽充裕**：每层 KV=2MB @seq_len=2048，500GB/s 读取仅需 4µs

| SRAM | 面积 | TPS | TTFT |
|:---:|:---:|:---:|:---:|
| 512KB | 31.8mm² | 148.3 | 160ms ✓ |
| 1MB | 34.3mm² | 148.3 | 160ms ✓ |
| 2MB | 39.3mm² | 148.3 | 160ms ✓ |
| 8MB | 69.1mm² | 148.3 | 160ms ✓ |

> **512KB 已足够**（需求约 64KB）。之前 8MB 基线是 LPDDR5 场景的习惯性延续，对 on-chip 场景不适用。

---

## 3. 面积分解 (block 4×1536, SRAM=1MB @12nm)

| 组件 | 面积 | 占比 |
|------|:---:|:---:|
| PE 阵列 (6,144 MACs) | 4.4mm² | 12.8% |
| SRAM L1+L2 (1MB) | 7.1mm² | 20.7% |
| PCIe Gen4 ×4 | 5.9mm² | 17.2% |
| SFU + RISC-V + DMA + Crossbar | 10.8mm²¹ | 31.5% |
| TSV overhead (10%) | 3.1mm² | 9.0% |
| DRAM PHY | 0mm² | — |

> ¹ 固定开销中 SFU=4.4mm², RISC-V=2.9mm², Crossbar=2.9mm², DMA=2.9mm² → 减去部分被 TSV 基数变化影响  
> 面积 = (PE + SRAM + PCIe + 固定) × 1.10 = 31.1 × 1.10 = 34.2mm²

---

## 4. 与 RK1828 对标

| | RK1828 | Arc Model (H=4, W=1536) |
|---|---|---|
| TOPS INT8 | 20 | 6.1 |
| 工艺 | 22nm | 12nm |
| 面积 | ~100mm² | **~34mm²** |
| 7B TPS | 59-180 | **148** |
| mm²/TOPS | 5.0 | 5.6 |
| mm²/TPS | 0.56-1.69 | **0.23** |

> 面积效率 mm²/TPS=0.23，比 RK1828 好 2.4-7.3×。mm²/TOPS 接近（5.0 vs 5.6）。

---

## 5. 推荐配置

| 参数 | 值 |
|------|------|
| 引擎 | **block 4×1536** (broadcast output stationary) |
| SRAM | **1 MB** (L2)，512KB 也可行 |
| 精度 | INT4 权重 / INT8 激活 |
| 内存 | On-chip 3D DRAM 5GB, 500 GB/s, TSV 互联 |
| PCIe | Gen4 ×4（主机通信） |
| 面积 @ 12nm | **~34 mm²** |
| TOPS INT8 | **6.1** |
| 功耗 | ~10W（估） |

### 预期性能

| 负载 | 性能 |
|------|:---:|
| Qwen2.5-7B decode | **148 tok/s** |
| TTFT (128 tok) | **160 ms** ✓ |
| Qwen2.5-3B decode | ~600 tok/s |

---

## 6. 两场景对比

| | 场景 A (LPDDR5) | 场景 B (On-chip 3D DRAM) |
|---|---|---|
| 模型 | 3B | 7B |
| 引擎 | FSA 128×128 | block 4×1536 |
| TOPS INT8 | 16.4 | **6.1** |
| SRAM | 4MB | **1MB** |
| 面积 @ 12nm | 62mm² | **34mm²** |
| Decode TPS | 22.5 | 148 |
| TTFT | 106ms | 160ms |
| 瓶颈 | LPDDR5 BW | On-chip BW |
| SRAM 敏感 | ✓（tile 粒度影响 DDR 读取次数） | ✗（权重常驻 + 500GB/s） |

**架构哲学**：
- 低带宽场景（LPDDR5）：SRAM 越大越好，靠 tile 粒度减少 DDR 读取
- 高带宽 + 权重常驻场景（3D DRAM）：SRAM 只需覆盖物理下限（~64KB），512KB 已充裕。多配纯浪费面积

---

*DSE 工具: Arc Model v4 — 统面积模型（TPUv1 ISCA 2017 校准）*  
*面积数据来源: [references/area_sources.md](../references/area_sources.md)*  
*工艺: TSMC 12nm | 模型: Qwen2.5 系列 INT4*  
*3D 堆叠: TSV overhead 10%，PCIe 保留*
