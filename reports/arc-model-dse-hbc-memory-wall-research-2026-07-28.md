# Arc Model DSE 改进方向：从 Qualcomm HBC 近内存计算看内存墙问题

> **来源**: Qualcomm HBC（High Bandwidth Compute）架构分析 × CaduceusCore Arc Model DSE 现状
> **日期**: 2026-07-28
> **状态**: 调研报告，建议纳入 DSE v3 规划

---

## 摘要

Qualcomm 在 2026 年 6 月 Investor Day 公布的 HBC（High Bandwidth Compute）近内存计算架构，通过将计算 die 3D 堆叠在 LPDDR DRAM 下方、以 TSV 直连，实现了 133 TB/s 有效内存带宽（较 LPDDR5X 提升 18 倍），从根本上缓解 AI 推理的内存墙问题。本报告分析 HBC 架构对 CaduceusCore Arc Model DSE 的启发，指出当前 DSE 在能耗模型、内存拓扑、封装成本和热约束四个维度的缺失，提出可立即执行的三个改进方向和两个中期规划。

---

## 1. HBC 架构概述

### 1.1 核心设计

```
传统方案（LPDDR/HBM）：           HBC 方案：
  SoC ← 厘米级走线 → DRAM           DRAM 堆栈
  （基板或中介层）                    ↕ TSV（硅通孔，微米级垂直互连）
                                   计算 die（HBC 加速器）
                                    ↕ 标准有机基板（低成本）
                                   主 SoC
```

**关键区别：不是 PIM（内存内计算），是近内存加速器。** 计算 die 在 DRAM 下方，数据移动距离从厘米级压缩到微米级。

### 1.2 关键数据

| 指标 | HBC Gen1（AI250） | HBC Gen2（AI300） |
|---|---|---|
| 每卡有效内存带宽 | 133 TB/s | 比 AI200 高 54× |
| 相对 LPDDR5X 提升 | **18×** | **54×** |
| 机架级带宽 | 7.4 PB/s | 3× AI250 |
| 封装方式 | 标准有机基板 | 标准有机基板 |

### 1.3 核心论据

> "在现代 AI 加速器中，数据移动的能耗远高于计算本身。一次 DRAM 读取的能耗（pJ/bit）往往是简单乘加运算的数十倍到数百倍。"

LLM decode 阶段属于**低算术强度工作负载**——每 token 需要访问全部权重，但只做极少计算。内存在这里不是辅助瓶颈，是主瓶颈。

---

## 2. CaduceusCore Arc Model DSE 现状

### 2.1 当前 DSE 搜索空间

```
引擎类型(7) × 阵列尺寸(4) × DRAM带宽(4) × SRAM尺寸(5) × 频率(3) × 模型(5) × Batch(2)
```

**DRAM 建模：** 扁平带宽数字（25.6 / 51.2 / 102.4 / 204.8 GB/s），基于 LPDDR5 变位宽方案。统一 SRAM 效率模型（权重预取 buffer + KV cache tile buffer）。

### 2.2 当前评估指标

| 已有 | 缺失 |
|---|---|
| ✅ tok/s（M=1 / M=2） | ❌ pJ/bit（数据移动能耗） |
| ✅ area（mm²） | ❌ energy-per-token（每 token 总能耗） |
| ✅ power（W） | ❌ compute/total-energy ratio（有效计算占比） |
| ✅ TPS/mm² | ❌ 封装成本分层（硅/中介层/基板） |
| ✅ TTFT | ❌ 热路径差异化约束 |
| ✅ DRAM 带宽利用率 | ❌ 数据移动距离/层次模型 |

### 2.3 当前 DSE 的核心结论

从 v2 DSE 报告（2026-06-29）和架构复盘（2026-06-19）：

1. **DRAM 带宽是主瓶颈**——所有引擎在相同带宽下收敛到相同 TPS，利用率稳定在 68%
2. **引擎差异体现在面积，不是吞吐**——在 DRAM 墙下计算单元差异被掩盖
3. **FSA 64×256 在 32mm²/11.5W 下达 23 tok/s**，为当前最优配置
4. **Pipeline fill+drain (385c/tile) 和 DRAM 是联合瓶颈**——只优化一个，另一个立刻成新墙

**关键问题：当前 DSE 无法解释「为什么 DRAM 利用率卡在 68%」，因为缺少能耗维度的数据移动成本模型。**

---

## 3. HBC 对 Arc Model DSE 的四维启发

### 3.1 内存物理拓扑 → 新增搜索维度

当前 DSE 的 DRAM 只有「带宽」一个变量。HBC 揭示：**同带宽不同物理拓扑，面积/功耗/成本差一个数量级。**

| 方案 | 带宽 | 物理距离 | 位宽需求 | 功耗 | 封装成本 | 当前 DSE |
|---|---|---|---|---|---|---|
| LPDDR5-32b | 25.6 GB/s | 厘米级（基板走线） | 32-bit | 高（长线驱动） | 低 | ✅ 已扫 |
| LPDDR5-64b | 51.2 GB/s | 厘米级 | 64-bit | 高 | 低 | ✅ 已扫 |
| LPDDR5-256b | 204.8 GB/s | 厘米级 | 256-bit | 极高 | 中 | ✅ 已扫 |
| **HBC 3D 堆叠** | **133 TB/s** | **微米级（TSV）** | 低（垂直） | **极低** | **中（有机基板）** | ❌ 缺失 |
| HBM3 | 819 GB/s | 毫米级（中介层） | 1024-bit | 中 | **极高（硅中介层）** | ❌ 缺失 |

**建议：** DSE v3 增加 `memory_topology` 维度，枚举三类方案（LPDDR 传统封装 / 3D 堆叠近内存 / HBM 中介层），每类带入不同的带宽-功耗-成本三元组。

### 3.2 数据移动能耗 → DSE 最关键的缺失指标

**HBC 的核心论据用数字翻译到 CaduceusCore 当前配置：**

```
LPDDR5-64b (51.2 GB/s) 当前配置:
  DRAM 读取能耗 ≈ 7 pJ/bit
  每 token 权重读取 ≈ 1.5 GB（Qwen2.5-3B INT4）
  DRAM 能耗 ≈ 7 × 10⁻¹² × 1.5 × 8 × 10⁹ = 84 mJ/token（仅数据搬运）
  MXU 计算能耗 ≈ 0.5 pJ/MAC × 几十万 MAC ≈ 忽略不计

  → 有效计算占比可能 < 5%
  → 大部分电费花在「搬数据」上

HBC 等效配置 (133 TB/s, TSV 0.5 pJ/bit):
  TSV 能耗 ≈ 0.5 × 10⁻¹² × 1.5 × 8 × 10⁹ = 6 mJ/token

  → 数据移动能耗降低 ~14 倍
  → 有效计算占比大幅提升
```

**这解释了 DSE 的 68% 带宽利用率天花板：不是带宽数字不够，是获取这些带宽的物理能耗限制了实际可达吞吐。**

**建议：** 在 DRAM 模型上加 `pJ/bit` 系数（LPDDR5 ≈ 7, HBC TSV ≈ 0.5），输出 `energy_per_token` 和 `compute_energy_ratio` 两个新指标。这会改变 DSE 的选型结论——从「吞吐最高」变成「能效最优」。

### 3.3 封装成本 → 面积模型需要分层

当前 DSE 面积模型是平面的一维数字（mm²）。HBC 的成本分三层：

| 成本层 | 内容 | 当前 DSE | HBC 做法 |
|---|---|---|---|
| 硅面积 | 计算 die + SRAM | ✅ 已建模 | ✅ 同 |
| 中介层/TSV | 垂直互连 | ❌ 缺失 | 标准有机基板（低成本） |
| 封装基板 | 物理承载 | ❌ 缺失 | 有机基板，非硅中介层 |

**HBC 的竞争力恰好来自第二层——用有机基板替代硅中介层，获得近内存的带宽优势同时控制成本。** 硅中介层（HBM 所需）成本极高，是 HBM 难以大规模替代 LPDDR 的主要原因。

**建议：** DSE v3 面积模型分层：`total_cost = silicon_cost + interconnect_cost + substrate_cost`。interconnect_cost 按物理拓扑差异化（TSV vs 基板走线 vs 硅中介层）。

### 3.4 热约束 → 功耗上限不是均匀的

HBC 明确指出：计算 die 在 DRAM 下方是热管理的挑战。

```
传统 LPDDR5:  计算 die 上方直接散热 → 12W 功耗上限可行
3D 堆叠 HBC:  计算 die → 散热需穿透 DRAM 堆栈
              → 热路径长 → 有效功耗上限可能降至 6-8W
```

当前 DSE 假设 ≤12W 均匀散热上限。**如果 DSE 加入 3D 堆叠选项，必须对「计算在 DRAM 下方」的配置施加更严格的功耗约束。**

**建议：** PPAC 模型增加 `thermal_derating_factor`，按内存拓扑取不同值（传统 = 1.0, 3D 堆叠 = 0.5~0.7）。

---

## 4. 可立即执行的三项改进

以下三项不需要改动引擎模型结构，可在 1-2 天内完成。

### 4.1 加 `pJ/bit` 能耗模型 + `energy_per_token` 指标

**难度：⭐ 一天**

```python
# DRAM 模型当前
dram_read_bytes = weight_per_token + kv_cache_per_token

# 新增
PJ_PER_BIT = {
    "lpddr5": 7.0,      # 长基板走线
    "hbm3": 3.5,         # 硅中介层
    "hbc_tsv": 0.5,      # TSV 垂直直连
}
dram_energy = dram_read_bytes * 8 * PJ_PER_BIT[topology]  # J/token
compute_energy = total_macs * PJ_PER_MAC                   # J/token
energy_per_token = dram_energy + compute_energy
compute_ratio = compute_energy / energy_per_token
```

**输出新增两列到 DSE 报告中：** `energy_per_token` 和 `compute_energy_ratio`。

### 4.2 把「带宽功耗比」加入 Pareto 前沿

**难度：⭐ 半天**

当前选型是二维 Pareto（TPS vs area）。改为三维：

```
选型 = max(TPS) subject to:
  area ≤ 40mm²
  power ≤ 12W
  energy_per_token ≤ X  ← 新增约束
```

这会筛选掉「用高带宽强推吞吐但能量浪费严重」的配置。预期 LPDDR5-256b 虽然 TPS 高，但 energy_per_token 恶化严重，三维 Pareto 可能筛掉它。

### 4.3 增加「SRAM-DDR 带宽比」作为架构参数

**难度：⭐⭐ 一天**

当前 DSE 扫 SRAM 尺寸只看缓存命中率。HBC 的启示是：**SRAM 和 DDR 之间的带宽比决定了什么计算该放哪侧。**

```
当前: SRAM BW / DDR BW ≈ 2048 GB/s / 51.2 GB/s = 40:1
      数据每 tile 必须往返 DDR，SRAM 只做缓冲

HBC 思路: 把低算术强度操作推到内存侧
      内存侧计算单元 / 远端 MXU 之间带宽比 → 决定分区策略
```

**DSE 新增对比场景：**

| 场景 | SFU 位置 | DDR 往返次数 |
|---|---|---|
| A（当前） | MXU 同侧 | weight→DDR→SRAM→MXU→SFU→DDR |
| B（HBC 启发） | 内存侧 | weight→本地 SFU（不经过 DDR），只回写结果 |

预期：场景 B 在 decode（M=1）时 DDR 往返减少 30-50%，energy_per_token 显著改善。

---

## 5. 中期规划（DSE v3 方向）

### 5.1 内存拓扑作为一级搜索维度

**难度：⭐⭐⭐ 一周+**

```python
MEMORY_TOPOLOGIES = {
    "lpddr5_traditional": {
        "pJ_per_bit": 7.0, "max_bw_gbps": 204.8,
        "thermal_derating": 1.0, "interconnect_cost_factor": 1.0,
    },
    "hbc_3d_stack": {
        "pJ_per_bit": 0.5, "max_bw_gbps": 133000,
        "thermal_derating": 0.6, "interconnect_cost_factor": 1.5,
    },
    "hbm3_interposer": {
        "pJ_per_bit": 3.5, "max_bw_gbps": 819,
        "thermal_derating": 0.9, "interconnect_cost_factor": 3.0,
    },
}
```

每类拓扑带入不同的带宽上限、pJ/bit、热降额、封装成本乘数。

### 5.2 封装成本分层模型

**难度：⭐⭐ 几天**

```python
total_cost = (
    silicon_area * cost_per_mm2 +
    dram_capacity * cost_per_gb +
    interconnect_cost(topology) +
    substrate_cost(topology)
)
```

需要从公开数据估算 `interconnect_cost` 和 `substrate_cost` 的合理范围。初步可参照：HBM 硅中介层 ≈ 硅成本的 2-3×，有机基板 ≈ 硅成本的 20-30%。

---

## 6. 优先级建议

| 优先级 | 改进项 | 理由 |
|---|---|---|
| **P0** | `pJ/bit` + `energy_per_token` | 改动最小（DRAM 模型加一个系数），但改变选型逻辑——从吞吐导向转向能效导向 |
| **P0** | 三维 Pareto（加 energy 约束） | 筛选掉「高吞吐高浪费」配置 |
| **P1** | SRAM-DDR 带宽比 / SFU 近内存 | 架构级改动，需要模型验证 |
| **P2** | 内存拓扑搜索维度 | 依赖 P0/P1 完成后的能耗模型 |
| **P2** | 封装成本分层 | 需要外部数据，可在拓扑维度之后做 |

---

## 7. 与 Func Model / SoC 的关系

Arc Model DSE 的改进**不影响当前 Func Model 和 SoC RTL**：

- Func Model 是选定配置后的 bit-exact 实现，不涉及架构搜索
- RTL 基于 Func Model 的接口设计，不受 DSE 搜索空间变化影响
- DSE 选型变更后，Func Model 和 RTL 只需更新参数（阵列尺寸/DRAM 配置），不需要逻辑改动

**唯一例外：** 如果 DSE 选了「SFU 近内存放置」，需要在 SoC 层面修改 SFU 的挂载位置（从 MXU 总线移到 DRAM 侧总线）。这属于架构级变更，需要 RTL Phase 规划。

---

## 8. 参考文献

- Qualcomm HBC 架构公布，Investor Day 2026.06
- 高通 Dragonfly AI250/AI300 平台白皮书
- CaduceusCore Arc Model DSE v2 报告（2026-06-29）
- CaduceusCore 架构复盘（2026-06-19）
- CaduceusCore Arc vs Func Model 定位文档
- Google TPU Ironwood 软硬件协同设计博客（2025.11）
- Hexagon-MLIR: Qualcomm NPU 编译器栈（arXiv 2602.19762）
- D.E. Shaw Anton 层次化验证方法（ICCD 2008）

---

*本报告由 Hermes Agent 根据 Qualcomm HBC 公开资料和 CaduceusCore 内部 DSE 数据综合分析生成。*
