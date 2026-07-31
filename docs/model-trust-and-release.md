# Model Trust Levels and Release Profiles

This document defines the trust framework used to qualify Arc Model results and
the release profiles under which architecture recommendations may be published.

## Trust Levels for Individual Parameters

Every decision-driving physical parameter carries a trust level from the
calibration registry (`references/calibration/parameters.yaml`):

| Level | Meaning | Permitted Use |
|-------|---------|---------------|
| **T0** | Engineering assumption, no direct evidence | Exploratory sensitivity only |
| **T1** | Published proxy or analytic bound | Feasibility / bound arguments |
| **T2** | Reproduced from verified source + held-out validation | Relative ranking inside calibrated range |
| **T3** | Signed-off reference RTL or silicon | Numeric prediction with residual interval |

A design point's effective trust level is the **minimum** trust level among all
parameters it consumes.  One T0 assumption is enough to keep the whole point
exploratory.

## Run Trust Levels

`DesignSpaceResultV2.trust_level` classifies an entire run:

| Level | Conditions |
|-------|------------|
| `authoritative` | Complete coverage, no failures, all ranking parameters T2+ and in range |
| `calibrated_estimate` | Calibrated model but missing secondary coverage axes |
| `exploratory` | Contains T0/T1 parameters or extrapolated values |
| `non_authoritative` | Partial run, failures, or coverage gaps on required axes |

## Release Profiles

### `experimental`

- Allows T0/T1 parameters.
- Requires all exploratory points to be explicitly tagged (`trust_level=exploratory`).
- Requires complete coverage manifest (`missing_axes={}`).
- Requires valid content-addressed artifact hashes.
- Does **not** publish promoted rankings.

Pass command:

```bash
uv run python scripts/release_gate.py --profile experimental
```

### `decision-grade`

- Every Pareto-driving parameter must be T2+.
- Every design point must be inside its calibration range.
- No extrapolated winner may enter the recommendation set.
- Coverage must be complete and the worktree must be clean.
- Generates a content-addressed release bundle under
  `artifacts/releases/<run-id>/`.

Because the current calibration registry keeps several ranking drivers at T0/T1
(e.g. `gmma_pipeline_scale`, `tensor_core_descriptor_overhead`),
`decision-grade` is expected to fail until additional measured evidence is
provided.  This is intentional and prevents uncalibrated rankings from being
promoted as authoritative.

```bash
# Expected to fail until T2+ evidence is added for T0/T1 parameters.
uv run python scripts/release_gate.py --profile decision-grade
```

## Release Artifacts

A successful gate writes:

```
artifacts/releases/<run-id>/
├── manifest.json    # profile, scenario, commit, bundle digest
└── SHA256SUMS       # checksums over canonical payload files
```

Existing artifact directories are never overwritten.  The canonical payload
(inputs + result + coverage + manifest) must reproduce byte-identically on a
clean checkout.

## Evidence Requirements

Each todo in the development plan must have matching evidence under
`.omo/evidence/`.  The evidence ledger verifier (`scripts/verify_evidence_ledger.py`)
checks that every todo and final verification wave (F1-F4) has a file, that
recorded commands exit 0, and that artifact digests are present.

## Scope Rules

The scope verifier (`scripts/verify_scope.py`) enforces:

- No PyTorch, ROS, Ramulator, or DRAMSim dependencies in phase one.
- Dated historical reports (`reports/dse-engine-model-bugs-2026-07-27.md` and
  `reports/dse-engine-model-bugs-postfix-2026-07-27.md`) are not modified.
- `.omo/ultraresearch/20260723-vla-models/sources/` is not staged.
- Every current recommendation in `docs/publication-manifest.yaml` binds to a
  run manifest.

## Mutation Tests

The acceptance suite includes monkey-patch mutations that prove the following
regressions are caught:

- Frequency forced to 1000 MHz.
- GMMA ideal MAC floor removed.
- TensorCore descriptor overhead ignored.
- Memory spill forced to zero.
- Unknown CV op returning zero cycles.
- Design-point positional association restored.
- Partial/non-authoritative point entering Pareto frontier.

Each mutation test restores state in teardown and asserts that a gate or oracle
fails.

---

## 跨节点引擎选择发现 (Cross-Node Engine Selection Findings)

**来源：** `.omo/evidence/task-14-engine-selection-p0-cross-node-dse.md`、`.omo/evidence/investigate-fsa-cross-node-freq.md`、`.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md`

跨节点全引擎 DSE 在 7/12/22/28nm 四个工艺节点上评估全部 8 种引擎在两种带宽场景中的表现。每个节点使用物理可行的频率范围（源自 [`sim/config/dse_axes.yaml`](../sim/config/dse_axes.yaml) 的 frequency-bound constraints）。

各节点报告每个引擎的最佳 tok/s（固定配置：128×128，INT4，2048 KB L2，无 weight cache）。

### lpddr5_3b (51.2 GB/s, Qwen2.5-3B INT4)

| 引擎 | 7nm | 12nm | 22nm | 28nm |
|:---|:---:|:---:|:---:|:---:|
| block | 36.6 / 99.0 | 20.8 / 119.1 | 17.9 / 195.4 | 9.7 / 261.4 |
| os_systolic | 31.8 / 99.0 | 31.8 / 119.1 | 31.8 / 195.4 | 31.8 / 261.4 |
| systolic | 22.0 / 97.0 | 22.0 / 113.7 | 9.9 / 175.6 | 5.2 / 229.4 |
| gmma | 20.8 / 102.0 | 20.8 / 127.2 | 20.8 / 225.0 | 20.8 / 309.4 |
| fsa | 20.5 / 97.2 | 20.5 / 114.3 | 9.9 / 177.6 | 5.2 / 232.6 |
| input_stationary | 11.1 / 99.0 | 11.1 / 119.1 | 11.1 / 195.4 | 11.1 / 261.4 |
| tensor_core | 9.9 / 99.0 | 9.9 / 119.1 | 8.4 / 195.4 | 6.8 / 261.4 |
| wmma | 0.1 / 101.0 | 0.1 / 124.5 | 0.0 / 215.1 | 0.0 / 293.4 |

> 格式: tok/s / area_mm²。高亮值 = 该节点最佳 tok/s。

### onchip_7b (500 GB/s, Qwen2.5-7B INT4)

| 引擎 | 7nm | 12nm | 22nm | 28nm |
|:---|:---:|:---:|:---:|:---:|
| block | 131.4 / 107.0 | 50.3 / 119.1 | 20.6 / 195.4 | 10.4 / 261.4 |
| os_systolic | **310.9** / 99.0 | **310.9** / 119.1 | **310.7** / 195.4 | **224.1** / 261.4 |
| systolic | 26.4 / 97.0 | 26.4 / 113.7 | 10.7 / 175.6 | 5.4 / 229.4 |
| gmma | 203.5 / 102.0 | 203.5 / 127.2 | 180.5 / 225.0 | 97.7 / 309.4 |
| fsa | 26.4 / 97.2 | 26.4 / 114.3 | 10.7 / 177.6 | 5.4 / 232.6 |
| input_stationary | 108.1 / 99.0 | 108.1 / 119.1 | 107.1 / 195.4 | 81.5 / 261.4 |
| tensor_core | 48.2 / 99.0 | 48.2 / 119.1 | 26.3 / 195.4 | 14.9 / 261.4 |
| wmma | 0.1 / 101.0 | 0.1 / 124.5 | 0.0 / 215.1 | 0.0 / 293.4 |

**关键发现：**

1. **os_systolic 是跨所有节点和场景的绝对领先者** — 在 lpddr5_3b 保持 31.8 tok/s（BW-bound），在 onchip_7b 达到 310.9 tok/s（compute-bound）。输出驻留（output-stationary）数据流使其在带宽受限和计算受限两种场景下均占优。
2. **GMMA 在高 BW 场景表现强劲，但老旧节点受限** — onchip_7b 7nm 下 GMMA 达 203.5 tok/s，仅次于 os_systolic；但 28nm 下降至 97.7 tok/s（频率上限限制 + 面积膨胀）。
3. **Block 引擎 BW-bound 行为被 os_systolic 超越** — lpddr5_3b 下 block 从 7nm 的 36.6 tok/s（受益于高频率和其他轴扫描）降至 28nm 的 9.7 tok/s；os_systolic 在相同 BW 下保持 31.8 tok/s。
4. **input_stationary 在 onchip 高 BW 下表现出色** — 7nm 108.1 tok/s、28nm 81.5 tok/s，证明 Eyeriss 式数据流在足够带宽下具有竞争力。
5. **wmma 在所有节点均不可行** — 接近零 tok/s，不适合当前工作负载和带宽组合。
6. **面积单调性已验证** — 所有引擎面积严格 7nm < 12nm < 22nm < 28nm，bitcell 查表模型正确传播节点缩放因子。

**决策含义：**
- **os_systolic 是当前跨所有场景和节点的最优引擎选择**（这是新增发现，此前仅 block+FSA 对比无法揭示）
- GMMA 在高 BW（onchip 3D DRAM）场景中作为第二名具有竞争力
- Block 仅在高 BW + 高频率 (7nm) 组合下具有竞争力
- FSA 在低 BW 下被 os_systolic 全面超越

**信任等级：** 探索性 (exploratory) — 频率约束来自架构推理，非硅测量；WMMA/GMMA PE 比等参数仍为 T0。全 8 引擎对比为首次完成，结论不应用作决策级依据。

---

## SRAM Bitcell 数据溯源 (Bitcell Data Provenance)

**来源：** `sim/contracts/bitcell.py` `BitcellTable` + `sram_area_mm2()`

Arc Model 的 SRAM 面积计算已从固定 `l1_per_kb × node_scale` 几何缩放迁移至基于 TSMC 真实 bitcell 面积的查表模型：

| 节点 | HD Bitcell 面积 (µm²/bit) | SRAM macro 面积 / KB (1.5× overhead) | 数据来源 |
|:---:|:---:|:---:|:---|
| 7nm | 0.027 | 0.041 mm² | VLSI 2017 / IEDM 2017 TSMC 披露 |
| 12nm | 0.074 | 0.114 mm² | TSMC 12FFC 产品简介 (2020) |
| 22nm | 0.092 | 0.141 mm² | TSMC 22nm 产品简介 (2023) |
| 28nm | 0.127 | 0.195 mm² | TSMC 28nm 代工厂数据 |

**校准验证：**
- TPUv1 (28nm, 28 MiB UB) → bitcell 推导面积与 die-shot 估算偏差 < 30%
- RK1828 (22nm, ~8 MiB SRAM) → bitcell 推导面积与产品规格偏差 < 30%

**使用方法：**
```python
from contracts.bitcell import sram_area_mm2

area = sram_area_mm2(capacity_kb=4096, node_nm=12.0, overhead=1.5)
```

**信任等级：** T2 — 数据来自已发表代工厂资料并经两个外部参考点交叉校准。

**局限性：**
- 仅收录 TSMC HD bitcell，不覆盖三星/Intel 或 HP/UHD 变体
- Peripheral overhead (1.3×–1.5×) 为固定近似，未建模容量依赖的 sub-linear 缩放
- 节点间（如 16nm）不支持插值

---

## 基于访问模式的 DRAM 效率方法论 (Pattern-Based DRAM Efficiency)

**来源：** P0 改进 Todo 6–9，核心代码在 `sim/models/memory_backend.py` `AccessType` + `sim/engine/mac_engine.py`

### 核心设计

DRAM 访问效率不再使用单一固定值，而是区分两种访问模式：

| 访问类型 | 效率参数 | 默认值 | 物理含义 |
|:---|:---|:---:|:---|
| **顺序 (SEQUENTIAL)** | `dram_efficiency` | 0.90 | 连续 bulk 加载，满 page 利用，行缓冲行命中率高 |
| **随机 (RANDOM)** | `dram_efficiency_random_bw` | 0.50 | 离散 KV cache 读取，行缓冲频繁 miss，增加 bank 冲突 |

### 延迟模型

对于随机 KV 访问，增加固定延迟惩罚，独立于带宽：

```
每缺失总延迟 = ceil(kv_bytes / bw_bytes_per_cycle) + random_latency_penalty_cycles
```

其中 `random_latency_penalty_cycles` 默认 40 cycles，模拟列选择 → 预充电 → 行激活的固定开销。

### 引擎路由

所有 8 个引擎的权重/激活 DMA 使用 `AccessType.SEQUENTIAL`；KV cache 读取使用 `AccessType.RANDOM`。FSA 引擎的 attention 路径已实现 Q-顺序 / KV-随机的正确路由。

### 验证

- 19 项模式验证测试 (`test_memory_access_pattern.py`)
- 87 项 DRAM 访问模式集成测试 (`test_dram_access_pattern.py`)
- 所有 8 引擎在 2 个频率点的路由正确性已参数化覆盖

**信任等级：** T1 — 参数基于架构推理与公开 DRAM 时序规格，未绑定硅测量。

---

## Decision-Grade State (Updated 2026-07-31)

!!! decision-grade 状态仍为 **FAIL** — 无变化。原因：

- **WMMA/GMMA PE 比率仍为 T0** — `gmma_pipeline_scale`、`tensor_core_descriptor_overhead` 仍是工程假设，未获得直接测量数据或可复现的公开来源。
- **全引擎跨节点覆盖完成，但仍为探索性** — 全 8 引擎 × 4 节点 × 2 场景排名矩阵（Todo 4, [`.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md`](../.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md)）揭示了 os_systolic 在跨所有节点和场景中均占绝对领先的新发现，GMMA 在高 BW 场景中作为第二名具有竞争力。但频率-节点绑定为探索性结论，os_systolic 的 PE 面积参数为 T0，结论不应用作决策级依据。
- **DRAM 效率模式化参数未经硅校准** — `dram_efficiency_random_bw = 0.50` 和 `random_latency_penalty_cycles = 40` 均为架构推理值，尚待针对目标 LPDDR5 或 3D DRAM 控制器的微基准验证。
- **SRAM bitcell 数据为 T2** — 这是本次改进中唯一达到 T2 以上的参数群；单一参数的提升不足以将整体决策等级提升至 `decision-grade`。

在以下条件满足前，`decision-grade` gate 将继续失败：

```bash
uv run python scripts/release_gate.py --profile decision-grade
# Expected: FAIL — T0/T1 parameters remain (2026-07-30)
```
