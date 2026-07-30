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

**来源：** `.omo/evidence/task-14-engine-selection-p0-cross-node-dse.md`

跨节点 DSE（Design Space Exploration）在 7/12/22/28nm 四个工艺节点上评估了 block 和 os_systolic 等引擎在两个带宽场景中的表现：

| 场景 | 节点 | 获胜引擎 | tok/s | 面积 (mm²) |
|:---|:---|:---|:---:|:---:|
| lpddr5_3b (51.2 GB/s) | 7nm | block | 36.6 | 99 |
| lpddr5_3b | 12nm | block | 20.8 | 119 |
| lpddr5_3b | 22nm | block | 20.8 | 195 |
| lpddr5_3b | 28nm | block | 20.8 | 261 |
| onchip_7b (500 GB/s) | 7nm | os_systolic | 310.9 | 99 |
| onchip_7b | 12nm | block | 50.3 | 119 |
| onchip_7b | 22nm | block | 50.3 | 195 |
| onchip_7b | 28nm | block | 50.3 | 261 |

**关键发现：**

1. **低 BW 场景 (lpddr5_3b) 下 block 引擎在所有节点获胜** — 面积效率引擎在内存墙下占优，结论跨节点一致。
2. **高 BW 场景 (onchip_7b) 下获胜引擎节点依赖** — 7nm 下 os_systolic 以 310.9 tok/s 大幅领先 (3.1× block)，但 28/22/12nm 仅有 block 的覆盖数据，无法判断 os_systolic 是否在粗节点保留领先优势。
3. **面积单调性已验证** — 同一引擎/配置下，面积严格 7nm < 12nm < 22nm < 28nm。block 128×128 从 99 mm² (7nm) 增长至 261 mm² (28nm)，比为 2.64×。
4. **覆盖限制** — ci-all-axes 模式在非 7nm 节点每个节点仅产生 1 个设计点（block 默认配置），跨节点多引擎对比深度不足。

**信任等级：** 探索性 (exploratory) — 28/22/12nm 面积使用平方律缩放，非硅校准；28/22/12nm 的多引擎对比待扩展扫描模式。

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

## Decision-Grade State (Updated 2026-07-30)

!!! decision-grade 状态仍为 **FAIL** — 无变化。原因：

- **WMMA/GMMA PE 比率仍为 T0** — `gmma_pipeline_scale`、`tensor_core_descriptor_overhead` 仍是工程假设，未获得直接测量数据或可复现的公开来源。
- **多节点覆盖仍为探索性** — 跨节点 DSE（Todo 14）产生了首次跨节点排名矩阵，但 28/22/12nm 仅有 block 引擎的覆盖数据。完整的多节点引擎对比需要更丰富的扫描模式。
- **DRAM 效率模式化参数未经硅校准** — `dram_efficiency_random_bw = 0.50` 和 `random_latency_penalty_cycles = 40` 均为架构推理值，尚待针对目标 LPDDR5 或 3D DRAM 控制器的微基准验证。
- **SRAM bitcell 数据为 T2** — 这是本次 P0 改进中唯一达到 T2 以上的参数群；单一参数的提升不足以将整体决策等级提升至 `decision-grade`。

在以下条件满足前，`decision-grade` gate 将继续失败：

```bash
uv run python scripts/release_gate.py --profile decision-grade
# Expected: FAIL — T0/T1 parameters remain (2026-07-30)
```
