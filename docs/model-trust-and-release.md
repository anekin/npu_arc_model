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

**来源：** `.omo/evidence/task-14-engine-selection-p0-cross-node-dse.md`、`.omo/evidence/investigate-fsa-cross-node-freq.md`

跨节点 DSE 在 7/12/22/28nm 四个工艺节点上评估 block 和 FSA 引擎在低带宽场景 (lpddr5_3b, 51.2 GB/s) 中的表现。每个节点使用物理可行的频率范围（源自 [`sim/config/dse_axes.yaml`](../sim/config/dse_axes.yaml) 的 frequency-bound constraints）：

| 节点 | 允许频率 (MHz) |
|:---:|:---|
| 7nm | 800–2000 |
| 12nm | 800–1200 |
| 22nm | 400–800 |
| 28nm | 200–600 |

各节点报告每个引擎的最佳 tok/s（固定配置：128×128，INT4，2048 KB L2，无 weight cache）：

| 节点 | 引擎 | 频率 (MHz) | tok/s | 面积 (mm²) | block/FSA tok/s 比 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 7nm | block | 800 | 20.8 | 99.0 | 1.000 |
| 7nm | FSA | 1200 | 20.8 | 97.2 | |
| 12nm | block | 800 | 20.8 | 119.1 | 1.000 |
| 12nm | FSA | 1200 | 20.8 | 114.3 | |
| 22nm | block | 600 | 20.8 | 195.4 | 1.137 |
| 22nm | FSA | 800 | 18.3 | 177.6 | |
| 28nm | block | 600 | 20.8 | 261.4 | 1.455 |
| 28nm | FSA | 600 | 14.3 | 232.6 | |

**关键发现：**

1. **Block 引擎是 BW-bound** — 在 51.2 GB/s 外存带宽下，tok/s 恒定 20.8，不受频率影响。频率从 800→2000 MHz 不会增加 tok/s，因为瓶颈在片外带宽而非片内计算。
2. **FSA 引擎是 compute-bound** — 脉动填充/排空开销使其计算受限，频率直接影响吞吐量。20.8 tok/s (7nm@1200MHz) → 14.3 tok/s (28nm@600MHz)，降幅 1.45×。
3. **先进节点 (7/12nm) 下两支引擎打平** — FSA 在 7nm 和 12nm 可借助 1200 MHz 高频率追平 block 的 BW 天花板（20.8 tok/s），且面积略小 1–4%。
4. **老旧节点 (22/28nm) 下 block 统治** — block 领先 1.14× (22nm) 和 1.46× (28nm)。FSA 受频率上限限制（22nm 最高 800 MHz, 28nm 最高 600 MHz），无法弥补 compute-bound 劣势。
5. **面积单调性已验证** — 同一引擎下面积严格 7nm < 12nm < 22nm < 28nm。FSA 在老旧节点的面积优势加大（7nm 小 1.9% → 28nm 小 12.4%），因为逻辑较轻的面积放大效应更小。
6. **频率对 BW-bound 引擎不是免费午餐** — 在 LPDDR5-51.2 GB/s 场景中，频率超过 ~600 MHz 后 block 的 tok/s 不再增长，仅增加功耗。瓶颈在片外，而非片内。

**决策含义：**
- 低带宽 (LPDDR5) 场景下 block 是更安全的引擎选择：所有节点均达到 BW 天花板（20.8 tok/s）
- FSA 仅在先进节点 (7/12nm) 具有竞争力（同等 tok/s，面积小 1–4%）
- 如果产品目标节点是 22/28nm，block 具有决定性吞吐量优势（14–46%）

**信任等级：** 探索性 (exploratory) — 频率约束来自架构推理，非硅测量；28/22/12nm 仅评估了 block 和 FSA，未覆盖其他引擎。

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
- **多节点覆盖仍为探索性** — 跨节点 DSE（Todo 14）产生了首次跨节点排名矩阵。后续频率感知分析（[`.omo/evidence/investigate-fsa-cross-node-freq.md`](../.omo/evidence/investigate-fsa-cross-node-freq.md)）揭示了 block 的 BW-bound 属性与 FSA 的 compute-bound 差异：7/12nm 下 FSA 追平 block（同等 tok/s，面积小 1–4%），22/28nm 下 block 领先 1.14–1.46×。但 28/22/12nm 仅覆盖 block 和 FSA 引擎，完整的多节点引擎对比需要更丰富的扫描模式。
- **DRAM 效率模式化参数未经硅校准** — `dram_efficiency_random_bw = 0.50` 和 `random_latency_penalty_cycles = 40` 均为架构推理值，尚待针对目标 LPDDR5 或 3D DRAM 控制器的微基准验证。
- **SRAM bitcell 数据为 T2** — 这是本次 P0 改进中唯一达到 T2 以上的参数群；单一参数的提升不足以将整体决策等级提升至 `decision-grade`。

在以下条件满足前，`decision-grade` gate 将继续失败：

```bash
uv run python scripts/release_gate.py --profile decision-grade
# Expected: FAIL — T0/T1 parameters remain (2026-07-30)
```
