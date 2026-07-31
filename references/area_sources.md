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

**跨节点缩放公式（PE 逻辑面积）：**

逻辑面积以 TSMC 7nm 为锚点，缩放因子由 `sim/engine/ppa_model.py` 中 `_node_scale_factor(node)` 定义：

```
scale(7nm)  = 1.0
scale(12nm) = 2.70   # TSMC 12FFC 密度比（非几何平方律 (12/7)²）
scale(22nm) = (22/7)² ≈ 9.88
scale(28nm) = (28/7)² = 16.0
```

因此 128×128 systolic PE 面积为：

| 节点 | 缩放因子 | 面积 (mm²) |
|:----:|:--------:|:----------:|
| 7nm  | 1.0      | 2.0        |
| 12nm | 2.70     | 5.4        |
| 22nm | 9.88     | 19.76      |
| 28nm | 16.0     | 32.0       |

> SRAM 面积不随 PE 逻辑一起按几何律缩放，而是按 `contracts/bitcell.py` 中 `BitcellTable` 的逐节点 bitcell 面积查表计算，再乘以 peripheral overhead。详见 §4。

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
| **WMMA** | ~1.13× block | 128×128 INT4 warp-level MMA 阵列，面积从 H100 SM die 分析推导（见 §7） | NVIDIA H100/Hopper whitepaper (2022); die-shot 分析 (Locuza, Semianalysis) |
| **GMMA** | ~1.375× block | WMMA PE + TMA 描述符引擎 + 共享内存控制逻辑；面积从 H100 SM die 分析推导（见 §8） | NVIDIA H100/Hopper whitepaper (2022); die-shot 分析 (Locuza, Semianalysis) |

**配置项（@7nm, 128×128 baseline）：**

| 引擎 | 面积 (mm²) | 每 MAC (µm²) |
|------|:---:|:---:|
| systolic | 2.0 | 122 |
| block / OS / IS / TC | 4.0 | 244 |
| FSA | 2.2 | 134 |
| WMMA | 4.5 | 275 |
| GMMA | 5.5 | 336 |

---

## 4. 非 PE 面积数据来源

| 组件 | 7nm 值 | 来源 |
|------|:---:|------|
| SRAM L1 | 0.002 mm²/KB | TSMC 7nm SRAM macro 公开数据（HD bitcell: 0.027µm²/bit），含外围电路 1.5× overhead。**2026-07 P0 改进：使用 `contracts/bitcell.py` `BitcellTable` + `sram_area_mm2()` 替代固定常量，支持按节点查表 + 可配 overhead。`l1_per_kb` / `l2_per_kb` 保留向后兼容。** |
| SRAM L2 | 0.0015 mm²/KB | 同上，HPC bitcell，overhead 1.3×。**同上，新代码建议使用 `sram_area_mm2(overhead=1.3)`** |
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

## 7. WMMA PE 面积推导（H100 SM die 参考）

> Todo 5 of `.omo/plans/wmma-gmma-pe-recalibration.md`：将 WMMA PE 基线从 T0 的 "1.5× block"（6.0 mm²）校准到基于 H100 SM die 分析的物理推导值。

**来源：**
- 主引用：NVIDIA H100/Hopper Tensor Core 白皮书 — https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
- Die-shot 交叉参考：Locuza die annotation、Semianalysis H100 架构深度分析（定性量级确认）

**推导步骤：**

| 步骤 | 值 | 依据 |
|------|:---:|------|
| H100 SM 面积 @4nm | ~6–8 mm² | die-shot 分析（Locuza, Semianalysis） |
| 每 SM 张量核心数 | 4 | H100 whitepaper：每 SM 4 个 TC 分区 |
| 每 TC 面积 @4nm（扣除 CUDA core/L1 后分摊） | ~1.0–1.5 mm² | (SM 面积 − CUDA core − L1) / 4 |
| 4nm → 7nm 密度缩放 | ~1.5× | 4nm 到 7nm 密度比（die 面积/逻辑密度），对应 TC @7nm ≈ 1.5–2.5 mm² |
| **我们的 PE 不是 TC** — 128×128 INT4 MAC 阵列 | ~16× TC 的每周期 MAC 数 | H100 TC 每周期 2048 FP16 MACs（128×16 矩阵）；128×128 = 16384 MACs |
| 面积因子（含布线/广播复杂性） | ~4–6× | 16× MAC 数按亚线性面积增长折算 |
| **WMMA PE @7nm 保守估计** | **3.5–5.0 mm²** | 1.5–2.5 mm² × (4–6) / ~2.7 折算因子，取中值 |
| **校准后推荐值** | **4.5 mm²** | = 2.25× systolic (2.0) = **1.125× block (4.0)**；替代旧 T0 值 1.5× block = 6.0 mm² |

> 注：H100 TC 的 2048 MACs/周期 是"每个周期每 TC"的稠密 FP16 吞吐；128×128 INT4 阵列的 16384 MACs 是其 ~16×。MAC 阵列面积随 MAC 数近似亚线性增长（MAC 单元本身规则、布线/广播/累加网络是主要开销），故取 4–6× 而非 16×。该推导为**公开代理量级**（T1），非硅实测。

**配置项：** `wmma_pe_area_mm2: 4.5`（@7nm 基线，跨节点由 `ppa_model.py` `_node_scale_factor` 缩放）

---

## 8. GMMA PE 面积推导（H100 SM die 参考）

> Todo 6 of `.omo/plans/wmma-gmma-pe-recalibration.md`：将 GMMA PE 基线从 T0 的 "1.75× block"（7.0 mm²）校准到基于 H100 SM die 分析的物理推导值。

**来源：**
- 主引用：NVIDIA H100/Hopper Tensor Core 白皮书 — https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
- Die-shot 交叉参考：Locuza die annotation、Semianalysis H100 架构深度分析（定性量级确认）

**推导步骤：**

| 步骤 | 值 | 依据 |
|------|:---:|------|
| WMMA PE @7nm（§7 推导） | **4.5 mm²** | 128×128 INT4 阵列，H100 SM die 分析推导（Todo 5） |
| TMA 描述符引擎 @7nm | **~1.0 mm²** | H100 TMA 是每 SM 一个集中式异步 DMA/descriptor 引擎，远小于 CUDA core 面积；die 面积推导的修正值（旧硬编码 `TMA_AREA_MM2 = 2.0` 偏大，现仅为文档常量，见 `gmma_engine.py`） |
| 共享内存控制逻辑 @7nm | ~0.3–0.5 mm² | 与 WMMA 共享 L1/SMEM 接口，额外仅仲裁/双缓冲控制 |
| **GMMA PE @7nm 保守估计** | **5.0–6.5 mm²** | 4.5 + 1.0 + 0.3~0.5，叠加布线/时钟树余量 |
| **校准后推荐值** | **5.5 mm²** | = 1.375× block (4.0) = 2.75× systolic (2.0)；替代旧 T0 值 1.75× block = 7.0 mm² |

> 注：GMMA 必须比 WMMA 大（TMA 溢价），但远小于旧的 1.75× block 拍脑袋值——TMA 描述符引擎是集中式小面积单元，不是第二个 MAC 阵列。该推导为**公开代理量级**（T1），非硅实测。

**配置项：** `gmma_pe_area_mm2: 5.5`（@7nm 基线，跨节点由 `ppa_model.py` `_node_scale_factor` 缩放）

---

## 9. 模型局限性

1. **PE 面积是 MAC 单元净面积 × 绕线/clock tree/grid 系数**。绕线开销随阵列增大而增大（O(N) 信号线），当前用固定 baseline → scale 是简化处理。
2. ~~**SRAM 面积按 KB 线性叠加**，实际宏观 SRAM 效率随总容量增大而提高。~~ **✅ 已修复 (P0 改进)** — 现使用 `sim/contracts/bitcell.py` `BitcellTable` + `sram_area_mm2()` 按节点查 bitcell 面积并应用可配 peripheral overhead。参考外部校准（TPUv1/RK1828）偏差 <30%。详见 [`sim/contracts/bitcell.py`](../sim/contracts/bitcell.py) 和 [`docs/model-trust-and-release.md`](../docs/model-trust-and-release.md)。
3. **工艺缩放平方律 `(node/7)²`** 在 12nm → 3nm 区间合理，对更老工艺（28nm, 65nm）是近似。
4. **Block 的相对比值**基于架构推理而非 die-shot 反推，量级可信但精确比无硬数据。**WMMA 已于 2026-07 从 H100 SM die 分析推导**（§7，T1 公开代理），**GMMA 亦于同月推导**（§8，T1 公开代理），两者均不再是纯架构推理。
5. **Bitcell 查表仅限 TSMC 节点** — `BitcellTable` 仅收录已发布 TSMC HD bitcell 数据（7nm, 12FFC, 22nm, 28nm）。三星/Intel 节点不在表中；未来如需跨厂查表需扩展 `source_uri` + `provenance` 元数据。处于两个已知节点之间的工艺（如 16nm）暂不支持插值。
6. **Peripheral overhead 为线性近似** — 实际 SRAM macro 效率随容量增大而 sub-linear（banking / 地址解码分摊），当前固定 overhead 参数（L1=1.5×, L2=1.3×）在中大容量（≥1 MiB）下偏保守。未来可引入容量依赖的 overhead 函数。
7. **Bitcell 数据为 HD（高密度）变体** — TSMC 同时提供 HP（高性能, bitcell 更大）和 UHD（超高密度, bitcell 更小）变体，当前表仅收录 HD。若设计中 SRAM 频率是关键约束，应改用 HP bitcell 面积。

---

*最后更新: 2026-07-31*
