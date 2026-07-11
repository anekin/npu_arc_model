# NPU 面积模型 — 数据来源与推导

> 所有面积数据以 **TSMC 7nm 为参考节点**，跨工艺缩放用 `(target_nm / 7nm)²` 平方律。
> 修改此文档时同步更新 `sim/engine/ppa_model.py` 和 `sim/config/design_space.yaml`。

---

## 1. 主校准点：TPUv1 systolic array

**来源：** Google TPUv1 ISCA 2017 — Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit"

| 参数 | 值 | 来源 |
|------|:---:|------|
| 工艺 | TSMC 28nm | §2.1 |
| 总面积 | ≤331mm²（Haswell 一半） | §2.1, Fig.1 |
| MXU 规格 | 256×256 INT8 systolic | §2.2 |
| MXU 时钟 | 700 MHz | §2.2 |
| 峰值算力 | 92 TOPS INT8 | §2.2 |
| MXU 占 die 比 | ~24% (die photo) | Fig.1 估算 |

**推导：**
- MXU 面积 @28nm ≈ 331 × 24% ≈ **79.4mm²**
- MAC 数 = 256 × 256 = 65,536
- 单 MAC @28nm ≈ 79.4 / 65536 = **1,212 µm²/MAC**
- 单 MAC @7nm ≈ 1,212 × (7/28)² = **75.7 µm²/MAC**
- 128×128 systolic array @7nm ≈ 16,384 × 75.7 = **1.24 mm²**
- **代码取 2.0 mm²**（含绕线损失、clock tree、电源/地网格，约 1.6× 实际物理比）

**配置项：** `systolic_pe_area_mm2: 2.0`

---

## 2. 交叉校验：Eyeriss v1

**来源：** Chen et al., "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs", JSSC 2016

| 参数 | 值 |
|------|:---:|
| 工艺 | TSMC 65nm |
| 总面积 | 12.25 mm² |
| PE 阵列 | 12×14 = 168 PEs |
| 单 PE | ~73,000 µm² @65nm |

**注：** Eyeriss PE 含独立 Scratchpad SRAM (0.5KB) + RLC 解压，面积偏大。
→ **不作为校准基准**，仅作量级交叉确认。折算 PE 逻辑部分（减 SRAM）与 TPUv1 在同一量级。

---

## 3. 相对比例推导（engine 间比值）

TPUv1 只提供 **systolic** 基准。其他 engine 类型的 PE 面积通过**架构差异推理 + 公开论文定性描述**推导：

| Engine | 相对 systolic | 理由 | 来源 |
|--------|:---:|------|------|
| **block** (output stationary + broadcast) | 2.0× | 每 MAC 需本地 accumulator（非 systolic pass-through），broadcast 网络比 systolic pipeline 更宽 | 架构推理；与 NVIDIA 博客 "Deep Dive into Tensor Cores" 对 broadcast 结构的描述一致 |
| **FSA** (CMP + Split) | 1.1× | 只在 systolic 基础上增加 CMP 比较器和 Split 控制逻辑，不增加 MAC 单元 | 架构推理 |
| **tensor_core** | 2.0× | NVIDIA V100 SM 架构分析：TC = 4×4×4 MAC 矩阵，与 block-style output stationary 结构类似 | NVIDIA V100 whitepaper, 2017 |
| **WMMA** | ~1.5× block | warp-level matmul 需额外的 warp scheduler 和 shared memory 接口 | NVIDIA Turing whitepaper, 2018 |
| **GMMA** | ~1.75× block | async copy (TMA) + 更大的 register file | NVIDIA Hopper whitepaper, 2022 |

**配置项（@7nm, 128×128 baseline）：**

| 引擎 | 面积 (mm²) | 每 MAC (µm²) |
|------|:---:|:---:|
| systolic | 2.0 | 122 |
| block / OS / IS / TC | 4.0 | 244 |
| FSA | 2.2 | 134 |
| WMMA | 6.0 | 366 |
| GMMA | 7.0 | 427 |

---

## 4. 非 PE 面积数据来源

| 组件 | 7nm 值 | 来源 |
|------|:---:|------|
| SRAM L1 | 0.002 mm²/KB | TSMC 7nm SRAM macro 公开数据（HD bitcell: 0.027µm²/bit），含外围电路 1.5× overhead |
| SRAM L2 | 0.0015 mm²/KB | 同上，HPC bitcell |
| DRAM PHY (DDR4/LPDDR5 64-bit) | 5.0 mm² | Cadence/SNPS DDR PHY IP 公开数据，12nm 折算 |
| PCIe Gen4 ×4 | 2.0 mm² | SNPS PCIe PHY IP 公开数据 |
| RISC-V 微控制器 | 1.0 mm² | 业界 RV32IMC 微控制器典型值 |
| SFU (Softmax/LayerNorm/GELU) | 1.5 mm² | 估算（256-wide ALU pipeline + LUT） |
| Crossbar (4×4 256-bit) | 1.0 mm² | 业界 crossbar IP 估算 |
| DMA (2ch + desc) | 1.0 mm² + 0.5/ch | AXI DMA IP 公开数据 |

---

## 5. 第三方产品基准（交叉校验用）

| 产品 | 工艺 | 面积 | TOPS | mm²/TOPS | 来源 |
|------|:---:|:---:|:---:|:---:|------|
| **RK1828** (block engine) | 22nm | ~100mm² | 20 INT8 | 5.0 | 产品规格书 / 行业分析 |
| Apple M4 ANE | 3nm | ~12mm²† | 38 FP16 | 0.32 | M4 die shot 分析 (Chipwise/TechInsights) |
| Google TPUv1 (systolic) | 28nm | 331mm² | 92 INT8 | 3.6 | ISCA 2017 |
| DaDianNao (学术 systolic) | 28nm | 67.7mm² | — | — | MICRO 2014 |

> † M4 总面积 ~166mm²，ANE 占 5-8% (die shot 估算)

**折算到 12nm 交叉校验：**

| 产品 | 面积 | TOPS | mm²/TOPS @12nm | PE 密度 µ²/MAC |
|------|:---:|:---:|:---:|:---:|
| TPUv1 | 60.7mm² | 92 | 0.66 | 223 |
| RK1828 | 29.8mm² | 20 | 1.49 | ~1,488 |
| Arc Model block 32×1536 (LPDDR5) | **~103mm²** | 49 | 2.11 | ~717 |
| Arc Model block 4×1536 (on-chip) | **~69mm²** | 6.1 | 11.3 | ~717 |

> Arc on-chip 面积含 PCIe (5.9mm²) + TSV 10% overhead (6.3mm²)，不含 DRAM PHY。

---

## 6. TSV 面积开销（3D 堆叠）

**来源**: 行业经验法则（HBM2/3 设计实践），非单篇论文。

| 参数 | 值 | 说明 |
|------|:---:|------|
| TSV overhead | 10% of total die | 含 keep-out zone + SerDes + 冗余修复 |
| 适用场景 | on_chip_memory capacity > 0 | 3D DRAM 堆叠时自动计入 |
| 配置项 | `tsv_overhead_pct: 0.10` | 可调参数 |

TSV（Through-Silicon Via）物理尺寸：~5µm 直径 + ~10µm keep-out zone ≈ 78.5µm²/TSV。500 GB/s 约需 2000 TSV（~2Gbps/TSV），裸 TSV 面积 ≈ 0.16mm²——但实际开销来自：
- TSV 阵列排布（不可与逻辑混合，需预留完整区域）
- SerDes 电路（每通道）
- 冗余 TSV（良率修复）

这些叠加导致 ~10% 规则值。对无 3D 堆叠的设计此项为 0。

---

## 7. 模型局限性

1. **PE 面积是 MAC 单元净面积 × 绕线/clock tree/grid 系数**。绕线开销随阵列增大而增大（O(N) 信号线），当前用固定 baseline → scale 是简化处理。
2. **SRAM 面积按 KB 线性叠加**，实际宏观 SRAM 效率随总容量增大而提高。
3. **工艺缩放平方律 `(node/7)²`** 在 12nm → 3nm 区间合理，对更老工艺（28nm, 65nm）是近似。
4. **Block/WMMA/GMMA 的相对比值**基于架构推理而非 die-shot 反推，量级可信但精确比无硬数据。

---

*最后更新: 2026-07*
