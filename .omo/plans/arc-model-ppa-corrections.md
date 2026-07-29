# Arc Model PPA 参数修正 — 基于 ULW-Research 20260729

> 本文件是 `arc-model-scenario-driven-dse-development.md` 的参数修正附录。
> 调研数据来源: `.omo/ulw-research/20260729-pparams/SYNTHESIS.md`
> 以下修正需在 Wave 0 的 Todo 1-6 中落实。

---

## 修正 1: DRAM 有效带宽

### 当前状态
- `README.md`: 声称 75%
- `npu_config.yaml` / `design_space.yaml`: 默认 `dram_efficiency: 0.85`
- `mac_engine.py`: `self.eff_bw = bw_raw * dram_efficiency * bw_multiplier`
- `dram.py`: 内部计算 ~90.4%，但 **代码未使用**（dead code）

### 修正后
- **删除 README 的 75% 声明**（自身 breakdown 矛盾，与 JEDEC/实测均不符）
- **保留 `dram_efficiency = 0.85`** — 对 LLM decode 的 sequential 权重流是保守估计
- **加注释**说明: "per-bank refresh 仅 3.6% (tRFCpb=140ns / tREFI=3900ns @ 16Gb LPDDR5)；额外损耗来自控制器调度/命令总线/bank 冲突；85% 是 sequential decode 的保守值"
- **删除或标记 `dram.py` 为 dead code**：引擎不走 `DRAMModel.effective_bandwidth_bytes_per_cycle()` 路径

### 影响范围
- `sim/config/npu_config.yaml:87` — 加 provenance 注释
- `sim/config/design_space.yaml:23` — 同上
- `sim/engine/mac_engine.py:57` — 注释说明 0.85 的来源
- `sim/models/dram.py` — 标记 unused 或删除
- `README.md:56` — 删除 75%，替换为 85%
- `sim/results/engine_eval_v3.md` — 标注 75% 声明不准确
- `docs/NPU_Engines_Architecture_Guide.md` — 多处引用 85% 的保持，引用 75% 的修正
- 所有 report 文件 — 如果引用了 75%，修正为 85%

### 不必改的
- `scenarios.yaml` 中的 `dram_efficiency: 0.85` 和 `effective_bw_gbps: 43.5` — 这些值本身是正确的

---

## 修正 2: TSMC 12nm 面积缩放因子

### 当前状态
- `ppa_model.py:37`: `node = float(am.get("process_node_nm", 7.0))`，然后 `(node/7)**2` 缩放
- `area_sources.md`: 所有面积以 7nm 参考，用 `(target_nm/7nm)²` 缩放
- README: "TSMC 12nm（面积 = 7nm 基线 × 2.94）"
- 架构报告中所有 12nm 面积都用 2.94× 计算

### 为什么 2.94× 是错的
12FFC 不是真正的 12nm 几何——它是 16FFC 的光学缩小版：
- SRAM bitcell 和 16FFC 完全相同 (0.074 µm²)，**没有缩小**
- 12FFC vs 16FFC 面积仅缩小 0.86×（TSMC 官方数据）
- 7nm 密度 = 91.2 MTr/mm², 12nm 密度 = 33.8 MTr/mm²
- **正确比值 = 91.2/33.8 = 2.70×**，不是 2.94×

### 修正后
- `ppa_model.py`: 改为查表或精确密度比 2.70× for 12nm
- `area_sources.md` §5: 所有 12nm 交叉校验面积用 2.70× 重算
- README: "面积 = 7nm 基线 × 2.70"
- `area_sources.md`: 补充 12FFC 的 SRAM 数据 — bitcell 未缩小

### 影响范围
- `sim/engine/ppa_model.py:37-40` — 核心修正点
- `references/area_sources.md` — §1-§5 以及 §7
- `README.md:59` — 缩放因子
- `reports/arch-report-A-lpddr5-3b.md` — 面积分解表
- `reports/arch-report-B-3ddram-7b.md` — 面积分解表
- `reports/arch-dse-three-scenarios.md` — 所有面积数字
- `docs/tiny-npu-analysis/tiny-NPU vs CaduceusCore PPA对比.md` — 面积表
- `docs/NPU硬件详细架构设计v0.1.md` — 面积部分

---

## 修正 3: PE 面积模型的 trust level

### 当前状态
- `area_sources.md` 中 block/systolic=2.0×, WMMA=1.5×block, GMMA=1.75×block 均为 "架构推理"
- 所有值在 config 中作为精确乘法因子使用，无置信度标注

### 修正后
- **保留 2.0× 比值**（Gemmini DAC 2021 显示 systolic=1.79× vector，支撑量级合理性）
- **降级为 T1 假设**：`参考文献: Genc et al., DAC 2021 (systolic 1.79× vector)；架构推理；无 die-shot 直接验证`
- WMMA(1.5×)、GMMA(1.75×)：无任何 published proxy，降为 **T0 假设**
- 在 `area_sources.md` 中为每个比值添加 trust level 和 source 列

### Trust level 定义（对齐 Phase 3b claim ledger）
| Level | 含义 | 可用于 |
|:---:|------|--------|
| T0 | engineering assumption, 无 published source | exploratory sensitivity only |
| T1 | published proxy or architectural reasoning | feasibility/bound within range |
| T2 | reproduced from verified source + held-out validation | relative decision with residual interval |
| T3 | signed-off reference RTL/silicon | numeric prediction with interval |

### 影响范围
- `references/area_sources.md` — 新增 trust level 列
- `sim/engine/ppa_model.py` — 可标记 T0 参数（不改数值）
- Plan Todo 17（calibration gate）— 输入数据

---

## 修正 4: 新增的能耗锚点

### 当前状态
- PPA 模型无 pJ/MAC 基线（只有面积模型 + 简单功耗密度估算）
- `ppa_model.py:141`: `# 12nm: ~0.5 W/mm² for logic, ~0.1 W/mm² for SRAM (active)` — 无来源

### 修正后
- 新增能耗锚点引用（不一定要改代码，但需在 Todo 17 中可追溯）:

| Source | Node | pJ/MAC (core) | TOPS/W |
|--------|------|:---:|:---:|
| NVIDIA MCM (JSSC 2020) | 16nm | 0.11 | 9.5 |
| DepFiN (JSSC 2023) | 12nm | 0.05-0.20 | 5-20 |
| Metis AIPU (ISSCC 2024) | 12nm | 0.067 | 15 |
| DiP-WS 64×64 | 22nm | — | 9.5 |

- `ppa_model.py` 功耗密度注释加来源引用
- `references/area_sources.md` 新增能耗章节

---

## 修正 5: 参数溯源与 trust gate 输入

以下参数对照表用于 Todo 17 的 calibration gate:

| 参数 | 当前来源 | 修正后来源 | 目标 Trust Level |
|------|----------|-----------|:---:|
| systolic PE area @7nm | TPUv1 ISCA 2017 die-shot | 保持；新增 Simba/DiP 交叉验证 | T2 |
| block/systolic = 2.0× | 架构推理 | 保持；降为 T1 (= Gemmini proxy) | T2 (需 synthesis) |
| GMMA pipeline_scale = 0.05 | uncalibrated | 保持 T0；标注来源为 H100 架构假设 | T1 |
| TensorCore descriptor = 5 | uncalibrated | 保持 T0；标注来源为 TC fragment model | T1 |
| pJ/MAC @ 12nm | 无 | 0.10-0.20 (DepFiN 实测) | T1 |
| TSV overhead = 10% | 行业经验法则 | 保持；标注为 rule-of-thumb | T1 (需 published proxy) |
| DRAM PHY @ 12nm = 14.7mm² | Cadence/SNPS IP 公开数据 | 保持 | T1 |
| 功耗密度 0.5/0.1 W/mm² | 代码注释，无来源 | 加 DepFiN 引用 | T1 |

---

## 对 Plan Todos 的影响

| Todo | 影响 |
|------|------|
| **Todo 1** (baseline freeze) | 无直接影响。但 baseline snapshot 应包含当前（修正前）的 2.94× 面积值作为历史参考 |
| **Todo 2** (schema v2) | 需要在 schema 中加 `provenance` 字段：source, trust_level, calibration_range |
| **Todo 3** (物理 oracle) | 红色矩阵中，面积/能耗对比应使用修正后的 2.70× 而非 2.94× |
| **Todo 4** (engine registry) | 无直接影响 |
| **Todo 5** (engine formula fix) | 无直接影响，但 `dram_efficiency` 的文档要修正 |
| **Todo 6** (frequency/bandwidth units) | **高影响**：`bandwidth_bytes_per_cycle` 的换算必须用正确的频率关系，同时确保 `dram_efficiency` 作为独立乘法因子而非隐式单位转换 |
| **Todo 11** (3D DRAM backend) | **高影响**：PPA 计算必须用 2.70× 而非 2.94× |
| **Todo 17** (calibration gate) | **最高影响**：修正 5 的 trust level 表直接输入 |

---

*最后更新: 2026-07-29 | 基于 .omo/ulw-research/20260729-pparams/SYNTHESIS.md*
