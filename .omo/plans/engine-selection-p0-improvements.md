# engine-selection-p0-improvements - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** 四个精确修复，让 NPU 架构引擎选择的结论从"猜的"变成"算的"——面积模型拆开算（逻辑管逻辑、内存管内存）、带宽不再一刀切（顺序读写和随机读写分开处理）、中间场景补齐（不在两个极端之间猜测）、跨工艺节点验证（28nm 的老厂和 7nm 的新厂，引擎排名稳不稳）。

**Why this approach:** 当前"低带宽选 FSA、高带宽选 block"的结论只有两个场景支撑，且 SRAM 面积被错误地用同一把尺子缩放。这个计划用公开的管芯数据重建面积模型，加上两个"腰线"场景（中带宽和中高带宽），最后在四个工艺节点上跑一遍确认结论不翻车。

**What it will NOT do:** 不会碰性能公式（只改面积和带宽效率）、不会引入需要联网的仿真器、不会新增引擎类型、不会把"实验结论"包装成"决定级建议"。

**Effort:** Medium — 15 todos, 3 waves, 预计 4-6 小时 (agent-time)
**Risk:** Medium — 面积模型重构可能引入回归，但有完整回归测试门
**Decisions to sanity-check:** 工艺节点范围 (28/22/12/7nm)，DRAM random 效率系数 (0.50)，新场景定义 (LPDDR5x 68 GB/s + HBM2e 410 GB/s)

Your next move: approve this plan to begin execution, or request high-accuracy review first. Full execution detail follows below.

---

> TL;DR (machine): Medium effort, Medium risk. Separates SRAM area from logic scaling, adds access-pattern-based DRAM efficiency, 2 intermediate scenarios, and 4-node comparison. 15 todos in 3 waves.

## Scope

### Must have

- 将 `ppa_model.py` 中 SRAM 面积从 `KB × per_kb × node_scale` 改为 `size_bytes × bitcell_area(nm) × (1 + peripheral_overhead)`，PE 逻辑面积保持 density ratio 缩放。
- 构建公开可查的 SRAM bitcell 面积查表，覆盖至少 28nm/22nm/12FFC/7nm 四个候选节点。
- 将 `MemoryTopology` 和 `CalibrationRef` 中硬编码的 `process_node_nm=12.0` 改为从配置读取。
- 移除 `legacy_result.py` 中非 12nm 即丢失的守卫条件。
- 在 `MemoryAccessPattern` 中添加 `access_type` 字段（SEQUENTIAL / RANDOM）。
- 在 `mac_engine.py:148` 的 `eff_bw` 计算中区分权重流（sequential，效率 ~0.90）和 KV cache 流（random，效率 ~0.50），并将此区分应用到所有 engine 的 DMA 计算和 `kv_cache.py` 的访问延迟。
- 在 `scenarios.yaml` 中新增 `lpddr5x_7b`（LPDDR5x-64bit, ~68 GB/s + Qwen2.5-7B）和 `hbm2e_7b`（HBM2e, ~410 GB/s + Qwen2.5-7B）两个中间场景。
- 将新场景接入 `design_space_explorer.py:1124-1130` 的 cross-validation。
- 在 `dse_axes.yaml` 中添加 `process_node` 轴，支持 28/22/12/7nm 扫描。
- 跨节点重跑 DSE，生成每个 scenario×node 下的引擎排名对比表。
- 更新 `references/area_sources.md` 和 `references/calibration/parameters.yaml` 中的 bitcell 数据溯源和 trust level。
- 更新 `docs/model-trust-and-release.md` 加入跨节点引擎选择的结论。

### Must NOT have (guardrails, anti-slop, scope boundaries)

- 不修改任何 engine 的性能公式（cycle 计算逻辑不动，只改面积和 DRAM 效率）。
- 不引入 Ramulator/DRAMSim 等外部 DRAM 模拟器。
- 不新增或删除 engine 类型。
- 不修改历史 dated report（`reports/dse-engine-model-bugs-*.md`）。
- 不改变现有 CLI flags 和 exit codes 的契约。
- 不在首阶段引入真实硅数据校准——只使用公开资料。
- 不为 LPDDR5x 或 HBM2e 编写新的时序模型——仅使用带宽参数。
- 不修改 `scenarios.schema.Scenario` 的 Pydantic schema 结构，仅新增 YAML entry。
- 不破坏 `decision-grade` profile 的 fail-closed 行为。
- `.omo/ultraresearch/20260723-vla-models/sources/` 永不进入提交。
- **不将 DRAM 延迟从带宽推导**——延迟（tRC 等效）和带宽效率是两个独立参数，必须分开建模。

### Metis-flagged risks and mitigations (from pre-planning review)

| Risk | Mitigation |
|---|---|
| C1 bitcell 查表使 SRAM 面积缩小 ~6×，可能颠覆 Pareto 排名。 | 新增 **Todo 0（校准门）**：在实现前对比新旧 SRAM 面积，与 `area_sources.md` 和 TPUv1/RK1828 交叉校验，设定可接受偏差范围。 |
| C2 混用延迟和带宽——`dram_access_cycles` 是 tRC 式延迟参数，应从带宽推导。 | 拆分为两个独立参数：`random_bw_efficiency` + `random_latency_penalty_cycles`；保留现有 KV hit-rate 逻辑。 |
| C2 未提及已有的 `_kv_dram_efficiency()` 和 `_dram_eff_for_bytes()`。 | Audit 现有函数，明确替换/增强规则——不静默重复计算。 |
| C3 cross-validation auto-detect 仅支持 2 个场景。 | 扩展 `design_space_explorer.py:1124-1130` 的多路映射 + 为新场景添加 benchmark entry。 |
| C4 28nm/22nm/7nm 无 density ratio 数据，使用几何平方律。 | 标记跨节点排名为 T1/exploratory；12nm 以外的节点不做决策级声明。 |
| C4 process_node 轴需要补全 config→AreaModel→runner→legacy 的管线。 | 明确定义每个文件需要修改的 exact 行数和变更内容。 |
| DSE 运行时间爆炸（5 scenarios × 4 nodes）。 | `ci-all-axes` 模式每个 scenario ≤ 10 分钟；`full` 模式可选。 |

## Verification strategy
> Zero human intervention - all verification is agent-executed.

- Test decision: **TDD with pytest**. 每个 todo 先提交可复现的红色测试或 fixture，再做最小实现并跑全部回归。已有 63+ 测试必须保持通过。
- Engine formula invariants: 不修改任何 engine 的性能公式。面积和 DRAM 效率变更后，所有 `test_engine_physical_invariants.py` 的断言必须保持绿色。
- Area consistency oracle: SRAM 面积在新旧模型间保持物理一致（同节点下 ±5%）。跨节点方向正确（28nm > 22nm > 12nm > 7nm）。
- DRAM bandwidth monotonicity: sequential 访问的有效带宽 ≥ random 访问的有效带宽。
- Scenario coverage: 新场景必须产生至少 1 个 non-trivial design point（非零 tok/s 且面积 ≤ 约束上限）。
- Blocking-test policy: blocking suite 禁止 `skip`/`xfail`；focused suite、完整 pytest、quick DSE 和 full coverage dry-run 都必须记录 collected/passed/failed/skipped/exit code。
- Evidence: `.omo/evidence/task-<N>-engine-selection-p0.<ext>`；每条证据同时记录命令、exit code、git commit、config/hardware/lock digests。
- QA policy: 每个 todo 都有 happy path + failure/negative path；不得用 grep 命中、worker 自述或历史 JSON 代替实际执行。

## Execution strategy

### Parallel execution waves

> Target 5-8 todos per wave. Wave 3 is the final wave (3 todos, minimum allowed).

```
Wave 1 — C1 Logic/SRAM area separation (5 todos, sequential dependency chain)
├── Todo 1: SRAM bitcell lookup table
├── Todo 2: Refactor AreaModel (SRAM + logic split)
├── Todo 3: Fix hardcoded process_node_nm
├── Todo 4: Clean up legacy_result anti-patterns + update calibration
└── Todo 5: Cross-node area regression tests

Wave 2 — C2 DRAM patterns + C3 Scenarios (7 todos, two parallel lanes)
├── Lane A: C2 DRAM access patterns
│   ├── Todo 6: Add access_type to MemoryAccessPattern
│   ├── Todo 7: Implement pattern-based DRAM efficiency
│   ├── Todo 8: Replace fixed KV cache cycles
│   └── Todo 9: DRAM pattern validation tests
└── Lane B: C3 Intermediate scenarios
    ├── Todo 10: Add LPDDR5x_7B + HBM2e_7B scenarios
    ├── Todo 11: Wire into cross-validation
    └── Todo 12: Run scenario DSE comparison

Wave 3 — C4 Multi-node comparison (3 todos, depends on Wave 1)
├── Todo 13: Add process_node as DSE axis
├── Todo 14: Run cross-node DSE + engine ranking matrix
└── Todo 15: Update decision-grade documentation
```

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|---|---|---|---|
| 1 | — | 2 | — |
| 2 | 1 | 3, 4, 5, 13 | — |
| 3 | 2 | 4, 13 | — |
| 4 | 3 | 13 | 5 |
| 5 | 2 | 13, F1-F4 | 4 |
| 6 | — | 7 | 10 |
| 7 | 6 | 8 | 10,11 |
| 8 | 7 | 9 | 10,11,12 |
| 9 | 7,8 | F1-F4 | 12 |
| 10 | — | 11 | 6 |
| 11 | 10 | 12 | 7,8 |
| 12 | 1,7,8,11 | F1-F4 | 9 |
| 13 | 1,2,3,4 | 14 | — |
| 14 | 13 | 15 | — |
| 15 | 14 | F1-F4 | — |

Critical path: `1 → 2 → 3 → 4 → 13 → 14 → 15 → F1-F4`
Note: C3 (Todos 10-12) functionally benefits from C2 (Todos 6-9), but can execute in parallel; C4 (Todos 13-15) depends on C1 + benefits from C2/C3 for meaningful cross-node conclusions.

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. 构建 SRAM bitcell 面积查表与数据溯源

  **What to do**:
- **PRECONDITION (Calibration Gate):** Before implementing the bitcell lookup in code, run `scripts/p0_c1_sram_calibration_gate.py` which compares new bitcell-derived SRAM area against external reference points: TPUv1 28nm (~331mm² total, 92K MAC, SRAM area from die photo ~29mm² → ~0.0013 mm²/KB equivalent) and RK1828 22nm (~100mm² total). **Gate logic**: if new model agrees with external refs within ±30% but diverges from old model >30%, auto-pass with a warning and proceed. Only halt if it disagrees with external references (delta > ±30% vs TPUv1/RK1828). Evidence: `.omo/evidence/p0-c1-calibration-gate.json`.
  2. 填充已收集数据：TSMC 7nm HD=0.027µm²/bit、12FFC/16FFC=0.074µm²/bit、TSMC 22nm≈0.092µm²/bit、TSMC 28nm≈0.127µm²/bit。每个 entry 包含 `source_uri` 和 `provenance`。
   3. SRAM peripheral overhead 参数化（L1: 1.5×, L2: 1.3×，均可配置），公式：`sram_area_mm2 = size_bytes × 8 × bitcell_area_um2 × (1 + overhead) / 1e6`。L1 与 L2 使用独立 overhead 参数，匹配 `references/area_sources.md` 中 0.002/0.0015 per-KB 成本比。
  4. 新增 `sim/tests/test_bitcell_table.py`，验证查找成功、跨节点单调性、未知节点 fail-closed。
  5. 更新 `references/area_sources.md` §4 和 §7，补充 bitcell 数据溯源和 overhead 来源。

  **Must NOT do**:
  - 不使用 WikiChip 或论坛数据作为唯一来源——必须引用至少一个学术/官方文档。
  - 不删除现有的 `l1_per_kb` / `l2_per_kb` 配置项（保留兼容性直到 Todo 2 完成）。

  **Parallelization**: Wave 1 | Blocked by: — | Blocks: 2

  **References**:
  - `references/area_sources.md:77` — 当前 SRAM 7nm bitcell=0.027µm²/bit，overhead=1.5×。
  - `references/area_sources.md:133-135` — 已知限制："SRAM 面积按 KB 线性叠加"。
  - TSMC 7nm HD SRAM: 0.027µm²/bit (IEDM 2017, TSMC).
  - TSMC 16FFC SRAM: 0.074µm²/bit（12FFC 与 16FFC 相同）(TSMC 公开资料).

  **Acceptance criteria**:
  - `bitcell_table.area_um2_per_bit(7.0)` → 0.027；`(12.0)` → 0.074；`(22.0)` → ≈0.092。
  - `area_um2_per_bit(5.0)` 抛 `ConfigError`（未知节点）。
  - `sram_area_mm2(1024 KB, 12.0)` 返回合理值，overhead 可配置且默认 1.5×。
  - `uv run pytest sim/tests/test_bitcell_table.py -q` 通过。

  **QA scenarios**:
  - **Happy**: 四个已知节点的 bitcell 面积校验 + SRAM area 公式验证；Evidence `.omo/evidence/task-1-engine-selection-p0-bitcell.json`。
  - **Failure**: 注入未知节点 → `ConfigError`、负 overhead → `ConfigError`；Evidence `.omo/evidence/task-1-engine-selection-p0-bitcell-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_bitcell_table.py -q -k "valid or node" > .omo/evidence/task-1-engine-selection-p0-bitcell.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_bitcell_table.py -q -k "rejects or invalid or unknown" > .omo/evidence/task-1-engine-selection-p0-bitcell-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(contracts): add SRAM bitcell area lookup table` | `sim/contracts/bitcell.py`, `sim/tests/test_bitcell_table.py`, `references/area_sources.md`, `scripts/p0_c1_sram_calibration_gate.py`

- [x] 2. 重构 AreaModel — SRAM 走 bitcell，logic 走 node_scale

  **What to do**:
  1. 修改 `sim/engine/ppa_model.py:88-89`：`self.l1_per_kb` / `self.l2_per_kb` 不再乘以 `node_scale`，改为存储 `process_node` 并在 `estimate()` 中动态调用 `BitcellTable`。
  2. 新增 `self._bitcell_table: BitcellTable` 实例属性。
  3. 修改 `estimate()` 的 SRAM 计算（L192-195）：`l1 = KB × bitcell_area(nm) × 8 × (1+overhead) / 1e6`。
  4. PE 基线（L69-87）和 peripheral（L88-95）保持 `× node_scale` 不变——这些是 logic 面积。
   5. 更新 `PowerModel`（L256-260）：`logic_power_density` 和 `sram_power_density` 加上 per-node 区分（可选，纳入后续）。
   6. **更新 `PowerModel.estimate()`（L343-344）**：将 SRAM 面积计算从直接访问 `area_model.l1_per_kb` 改为通过 `BitcellTable` 计算。如果 `l1_per_kb` 已被弃用但尚未删除，添加弃用兼容路径（fallback 到 bitcell 表）。

  **Must NOT do**:
  - 不修改 PE baselines 的相对比值。
  - 不删除 legacy `l1_per_kb` / `l2_per_kb` config keys（向后兼容，标记 deprecated）。

  **Parallelization**: Wave 1 | Blocked by: 1 | Blocks: 3,4,5,13

  **References**:
  - `sim/engine/ppa_model.py:62-95` — AreaModel.__init__，所有 baseline 乘以 node_scale。
  - `sim/engine/ppa_model.py:171-250` — estimate()，SRAM 在 L192-195。
  - `sim/contracts/bitcell.py` — Todo 1 产出。

  **Acceptance criteria**:
  - 12nm 下 512KB L1 SRAM 面积在新旧模型中差异 <5%。
  - 跨节点 SRAM 面积单调递减（7nm 面积 < 12nm < 22nm < 28nm）。
  - `uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_result_contract.py -q` 通过（性能公式不变）。
  - 现有 legacy CLI DSE 输出中 `total_mm2` 变化 <10%（仅 SRAM 部分受影响）。

  **QA scenarios**:
  - **Happy**: 新旧模型面积对比，12nm 一致；跨节点 monotonicity 验证；Evidence `.omo/evidence/task-2-engine-selection-p0-areamodel.json`。
  - **Failure**: 错误 bitcell 数据导致面积异常；Evidence `.omo/evidence/task-2-engine-selection-p0-areamodel-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_engine_physical_invariants.py -q -k "area or node or bitcell" > .omo/evidence/task-2-engine-selection-p0-areamodel.json 2>&1
    test $? -eq 0
    PYTHONPATH=sim python3 -c "from sim.engine.ppa_model import AreaModel; ... " > .omo/evidence/task-2-engine-selection-p0-areamodel-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `refactor(ppa): separate SRAM bitcell from logic node-scale in AreaModel` | `sim/engine/ppa_model.py`

- [x] 3. 修复 MemoryTopology 和 CalibrationRef 中的硬编码 12nm

  **What to do**:
  1. `ppa_model.py:147` 和 L295：`process_node_nm=12.0` → 改为 `process_node_nm=self.node`（保存配置读取的 node 值）。
  2. `dse/runner.py:326-327`：`CalibrationRef(process_node_nm=12.0, node_scale=2.70)` → 从配置读取 `area_model.process_node` 并动态计算 `_node_scale_factor(node)`。
  3. `contracts/result.py:135-136`：`CalibrationRef.process_node_nm` 默认值保持 12.0（向后兼容），但改为显式传入。
  4. 更新所有创建 `MemoryTopology` 的测试文件（`test_memory_ppa.py:49`, `test_memory_backend.py:35`）。

  **Must NOT do**:
  - 不改变 `CalibrationRef` 的 schema 结构。

  **Parallelization**: Wave 1 | Blocked by: 2 | Blocks: 4,13

  **References**:
  - `sim/engine/ppa_model.py:147` — `MemoryTopology(process_node_nm=12.0)` 硬编码 #1。
  - `sim/engine/ppa_model.py:295` — 硬编码 #2。
  - `sim/dse/runner.py:326-327` — CalibrationRef 硬编码。
  - `sim/tests/test_memory_ppa.py:49` — 测试硬编码。

  **Acceptance criteria**:
  - 配置 `process_node=7.0` 时，MemoryTopology 的 `process_node_nm==7.0`。
  - 配置 `process_node=22.0` 时，CalibrationRef 的 `node_scale` 为 `(22/7)²`（≈9.88）。
  - `uv run pytest sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py sim/tests/test_result_schema.py -q` 通过。

  **QA scenarios**:
  - **Happy**: 改 process_node 配置后内存面积变化正确；Evidence `.omo/evidence/task-3-engine-selection-p0-hardcode.json`。
  - **Failure**: 非 12nm 配置下 CalibrationRef 仍为 12.0 → 失败；Evidence `.omo/evidence/task-3-engine-selection-p0-hardcode-negative.txt`。

  **Commit**: YES | `fix(ppa,dse): remove hardcoded 12nm from MemoryTopology and CalibrationRef` | `sim/engine/ppa_model.py`, `sim/dse/runner.py`, `sim/contracts/result.py`, 相关测试

- [x] 4. 清理 legacy_result 反模式 + 更新 calibration parameters

  **What to do**:
  1. `legacy_result.py:136` 和 L209：移除 `if any(r.calibration.process_node_nm != 12.0 ...)` 的守卫，改为正确传播 calibration 信息到 legacy projection。
  2. 更新 `references/calibration/parameters.yaml`：新增 `systolic_pe_area_28nm`、`systolic_pe_area_22nm`、`systolic_pe_area_7nm` entry，每个带独立 `source_uri` 和 `trust_level`。
  3. 更新 `sim/calibration/evaluate.py:65,84`：扩展 `_actual_value` 以支持新校准 ID。
  4. 更新 `references/area_sources.md` §1 TPUv1 推导链，补充 28nm→22nm→12nm→7nm 的递推公式。

  **Must NOT do**:
  - 不删除现有 12nm 校准 entry（`dram_phy_area_12nm` 等仍需保留）。
  - 不修改 calibration schema。

  **Parallelization**: Wave 1 | Blocked by: 3 | Blocks: 13 | Parallel with: 5

  **References**:
  - `sim/contracts/legacy_result.py:136,209` — 反模式守卫。
  - `references/calibration/parameters.yaml:17,31,46,60,74,87,100,114,128,144` — 现有 10 个 entry，仅 systolic 为 T2。
  - `sim/calibration/evaluate.py:42-46,65,84,126-129` — 校准评估器。

  **Acceptance criteria**:
  - `process_node=28.0` 的 DSE 结果在 legacy projection 中 `calibration.process_node_nm == 28.0`（不被标记为 loss）。
  - `calibration/parameters.yaml` 新增至少 4 个 entry。
  - `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` 通过。

  **QA scenarios**:
  - **Happy**: 非 12nm legacy projection 正确传递 calibration；Evidence `.omo/evidence/task-4-engine-selection-p0-legacy.json`。
  - **Failure**: 缺失 calibration ID → registry 报缺；Evidence `.omo/evidence/task-4-engine-selection-p0-legacy-negative.txt`。

  **Commit**: YES | `fix(contracts,calibration): remove non-12nm loss guard and expand calibration params` | `sim/contracts/legacy_result.py`, `references/calibration/parameters.yaml`, `sim/calibration/evaluate.py`, `references/area_sources.md`

- [x] 5. 跨节点面积回归测试

  **What to do**:
  1. 新建 `sim/tests/test_area_cross_node.py`，对 4 个节点 × 所有 8 个 engine 验证：
     - 面积单调递减（28nm > 22nm > 12nm > 7nm）。
     - 比值变化方向符合物理预期（SRAM-heavy 引擎在旧节点面积劣势更大，SRAM-light 引擎在新节点相对优势缩小），具体预测值从 bitcell 表与 geometric scaling 计算得出。
     - SRAM 占比随节点缩小而增加（logic 缩放快于 SRAM）。
   2. **重写 `sim/tests/oracles/ppa.py` 的 `_node_scale()` 函数以精确匹配 `ppa_model._node_scale_factor()`**。具体操作：
      - 将锚点从 12nm=1.0 改为 7nm=1.0（与 ppa_model 一致）。
      - 使用 `(node/7)²` 公式（且 12nm 返回 2.70，匹配 `ppa_model._node_scale_factor()` 的特殊处理；而非当前 `(node/12)² / 2.70`），消除 ~8× 的数值分歧。
      - 将所有 oracle 常量（如 `MEMORY_DIE_AREA_PER_GB_MM2` 等）从 12nm 基线重新锚定到 7nm 基线，使用 2.70× 密度比换算。
      - 验证：在 28nm 下新旧 oracle 结果应兼容（旧值 × 校正因子 ≈ 新值）。

  **Must NOT do**:
  - 不修改 engine 的性能测试。
  - 不依赖网络连接获取数据。

  **Parallelization**: Wave 1 | Blocked by: 2 | Blocks: 13, F1-F4 | Parallel with: 4

  **References**:
  - `sim/tests/oracles/ppa.py:22,42-43` — 当前以 12nm=1.0 为锚点，需统一。
  - `sim/tests/test_memory_ppa.py` — 当前仅测试 12nm。
  - `sim/tests/test_result_schema.py:515-521` — 当前仅断言 12nm/7nm。

  **Acceptance criteria**:
  - 跨节点测试对 8 engine × 4 node = 32 个组合全部通过。
  - Oracle 与 ppa_model 锚点统一（都用 7nm=1.0）。
  - `uv run pytest sim/tests/test_area_cross_node.py -q` 通过。

  **QA scenarios**:
  - **Happy**: monotonicity + relative ratios stable；Evidence `.omo/evidence/task-5-engine-selection-p0-cross-node.json`。
  - **Failure**: 注入异常 bitcell 数据 → 面积反转被捕获；Evidence `.omo/evidence/task-5-engine-selection-p0-cross-node-negative.txt`。

  **Commit**: YES | `test(ppa): add cross-node area regression and unify oracle anchor` | `sim/tests/test_area_cross_node.py`, `sim/tests/oracles/ppa.py`

- [x] 6. 为 MemoryAccessPattern 添加 access_type 字段

  **What to do**:
  1. 修改 `sim/models/memory_backend.py:22-47`：在 `MemoryAccessPattern` 中新增 `access_type: str = Field(default="sequential", pattern="^(sequential|random)$")`。
  2. 新建枚举 `AccessType`（SEQUENTIAL="sequential", RANDOM="random"）。
  3. 在所有创建 `MemoryAccessPattern` 的位置传入正确的 access_type：
     - 权重 DMA：SEQUENTIAL（`mac_engine.py` 和 `engine/compiler.py`）。
     - KV cache 访问：RANDOM（`kv_cache.py`）。
     - activation / scratch：标记为 SEQUENTIAL（保守）。
  4. 新增 `sim/tests/test_memory_access_pattern.py`，验证 schema、roundtrip、缺省 fail。

  **Must NOT do**:
  - 不修改 `MemoryAccessPattern` 中已有字段的语义。
  - 不在 schema 中使用 `Any` 类型。

  **Parallelization**: Wave 2, Lane A | Blocked by: — | Blocks: 7 | Parallel with: 10

  **References**:
  - `sim/models/memory_backend.py:22-47` — MemoryAccessPattern 定义。
  - `sim/engine/mac_engine.py:141-148` — eff_bw 计算处。
  - `sim/models/kv_cache.py:39,52` — 当前用 bw_bytes_per_cycle 和固定 80 cycles。

  **Acceptance criteria**:
  - `MemoryAccessPattern(access_type="sequential")` 合法，`("unknown")` 抛 `ValidationError`。
  - 所有权重路径的 access_type == "sequential"，所有 KV 访问的 access_type == "random"。
  - `uv run pytest sim/tests/test_memory_access_pattern.py -q` 通过。

  **QA scenarios**:
  - **Happy**: schema roundtrip + 创建验证；Evidence `.omo/evidence/task-6-engine-selection-p0-pattern.json`。
  - **Failure**: 非法 access_type → 被 Pydantic 拒绝；Evidence `.omo/evidence/task-6-engine-selection-p0-pattern-negative.txt`。

  **Commit**: YES | `feat(memory): add access_type to MemoryAccessPattern` | `sim/models/memory_backend.py`, `sim/engine/mac_engine.py`, `sim/models/kv_cache.py`, `sim/tests/test_memory_access_pattern.py`

- [x] 7. 实现基于访问模式的 DRAM 效率，并审计已有函数

  **What to do**:
  1. 先审计 `sim/engine/mac_engine.py:184` 的 `_dram_eff_for_bytes()` 和 L200 的 `_kv_dram_efficiency()`——这两个函数已实现基于 SRAM 驻留率的动态效率，返回 [0.55, 0.92]。文档化它们的职责，确保新模式不与其静默重复或冲突。
  2. 定义两个独立参数：
     - `dram_efficiency_random_bw = 0.50`（随机访问的带宽效率，低于 sequential 的 0.90）。
     - `random_latency_penalty_cycles = 40`（随机访问的额外延迟，独立于带宽）。**`random_latency_penalty_cycles = 40` 的推导依据**：源自 `tRC_cycles = 48`（`hardware.py:168`）——相当于 1GHz 下行冲突的近似 tRC，标记为 T0/exploratory。Todo 9 中的敏感性分析验证排名在 [20, 40, 60, 80] 范围内的稳定性。
     - **MUST NOT**: 从带宽公式推导延迟——延迟和带宽是独立的 DRAM 参数。
  3. 修改 `sim/engine/mac_engine.py:148`：`eff_bw` 分为 `eff_bw_weight`（sequential）和 `eff_bw_kv`（random × 已有的 `_kv_dram_efficiency`）。
  4. 为所有 8 个 engine 的 DMA 计算添加 access type：权重读 → SEQUENTIAL，激活读 → SEQUENTIAL，KV 读 → RANDOM。
  5. 修改 `sim/models/dma.py:70`：同样根据 access_type 应用效率。
  6. 新增配置项 `dram_efficiency_random_bw`（默认 0.50）和保持 `dram_efficiency` 为 sequential 默认值 0.90。
   7. 更新 `sim/contracts/hardware.py:159` 的 `MemoryConfig`，添加新字段和 provenance。
   8. **文档化 `dram_efficiency` 的角色**：Todo 7 完成后，scenario YAML 中的 `dram_efficiency` 字段仅作为文档参考（pre-improvement 基线）；实际带宽计算全部使用本 todo 定义的 per-pattern 效率（`dram_efficiency_random_bw` + sequential 默认值）。

  **Must NOT do**:
  - 不修改任何 engine 的 cycle 计算公式。
  - 不删除 `dram_efficiency` 配置项（向后兼容）。
  - 不从带宽推导延迟。
  - 不与 `_kv_dram_efficiency()` / `_dram_eff_for_bytes()` 静默重复。

  **Parallelization**: Wave 2, Lane A | Blocked by: 6 | Blocks: 8 | Parallel with: 10,11

  **References**:
  - `sim/engine/mac_engine.py:141-182` — eff_bw 计算和 `_dram_eff_for_bytes`、`_kv_dram_efficiency`。
  - `sim/engine/systolic_engine.py:41,61,96,120` — 各 engine 使用 eff_bw 的位置。
  - `sim/engine/block_engine.py:74,78,93` — 同上。
  - `sim/models/dma.py:65-88` — DMA transfer cycle 计算。

  **Acceptance criteria**:
  - Audit 报告记录在 `.omo/notepads/engine-selection-p0-improvements/learnings.md`，列出现有函数、冲突/重复点、替换策略。
  - sequential 效率 0.90 时，权重 DMA cycles < random 效率 0.50 时。
  - `random_latency_penalty_cycles` 在 KV miss 时正确加到总 cycles（独立于带宽），在 KV hit 时不加。
  - 所有 8 engine 的 DMA 计算接入 access_type。
  - `uv run pytest sim/tests/test_engine_physical_invariants.py -q` 通过（性能公式不变）。

  **QA scenarios**:
  - **Happy**: sequential ≈ 1.06× faster than current 0.85；random BW 效率 0.50 + 延迟 40 cycles 生效；Evidence `.omo/evidence/task-7-engine-selection-p0-pattern-eff.json`。
  - **Failure**: 缺少 audit 报告 → Gate 阻断；与 `_kv_dram_efficiency` 产生双重计算 → 测试检测到异常 KV 延迟；Evidence `.omo/evidence/task-7-engine-selection-p0-pattern-eff-negative.txt`。

  **Commit**: YES | `feat(engine): implement access-pattern DRAM efficiency with latency/bandwidth separation` | `sim/engine/mac_engine.py`, `sim/engine/*.py` (all engines), `sim/models/dma.py`, `sim/contracts/hardware.py`, `.omo/notepads/...` (audit)

- [x] 8. 替换 kv_cache.py 中的固定延迟和带宽模型

  **What to do**:
  1. 修改 `sim/models/kv_cache.py:39`：`self.bw_bytes_per_cycle` 由 raw BW 改为应用 `dram_efficiency_random_bw`（0.50）。
  2. 修改 L52：删除 `dram_access_cycles = 80`，改为两层模型：
     - 带宽部分：`kv_bw_cycles = math.ceil(kv_bytes_per_token / effective_bw_bytes_per_cycle_for_random)`。
     - 延迟部分：`kv_latency_cycles = random_latency_penalty_cycles`（40 cycles per miss，来自 Todo 7 的新配置项）。
     - 总 = `kv_bw_cycles + (kv_latency_cycles if miss else 0)`。
  3. 保留 `KVCacheModel.access()` 中现有的 SRAM hit/miss 逻辑——仅对 miss 部分应用 random 成本。
  4. 新增 `access_type="random"` 标记到 KV access。

  **Must NOT do**:
  - 不修改 KVCacheModel 的 SRAM hit/miss 概率逻辑。
  - 不从带宽公式推导延迟——延迟使用独立的 `random_latency_penalty_cycles`。
  - 不对 SRAM hit 施加 random 惩罚。

  **Parallelization**: Wave 2, Lane A | Blocked by: 7 | Blocks: 9 | Parallel with: 10,11,12

  **References**:
  - `sim/models/kv_cache.py:35-52,78-121` — 当前固定 80 cycles + hit/miss 逻辑。
  - `sim/npu_sim.py:100-102,137-140,273-275` — KV cache 接入全仿真的位置。

  **Acceptance criteria**:
  - KV access cycles 随 bandwidth 变化（51.2→102.4 GB/s 时带宽部分减半）。
  - KV miss 延迟 40 cycles 在 SRAM miss 时加入，hit 时不加。
  - sequential 访问不影响 KV（KV 始终为 random）。
  - `uv run pytest sim/tests/test_legacy_compatibility.py -q` 通过。

  **QA scenarios**:
  - **Happy**: 多种 BW 下 KV 延迟 = bw_cycles + latency_cycles（miss 时）；SRAM hit 时仅 2 cycles；Evidence `.omo/evidence/task-8-engine-selection-p0-kv-pattern.json`。
  - **Failure**: 零带宽 → `ConfigError`；hit 时误加 random 延迟 → assert；Evidence `.omo/evidence/task-8-engine-selection-p0-kv-pattern-negative.txt`。

  **Commit**: YES | `fix(kv_cache): split KV cost into bandwidth + latency, respect hit/miss` | `sim/models/kv_cache.py`

- [x] 9. DRAM 访问模式验证测试

  **What to do**:
  1. 新建 `sim/tests/test_dram_access_pattern.py`，验证：
     - sequential 效率 > random 效率（对所有 memory type → 真实内存也是如此）。
     - 权重 DMA cycles 在 sequential 模式＜在 random 模式下。
     - KV access 在 random 模式下正确乘以效率。
     - 两种模式的 wall-time 差异随带宽缩放保持稳定。
  2. 验证所有 8 个 engine 的 DMA 路径都传递了正确的 access_type。

  **Must NOT do**:
  - 不修改 engine 的物理公式 assert。

  **Parallelization**: Wave 2, Lane A | Blocked by: 7,8 | Blocks: F1-F4 | Parallel with: 12

  **References**:
  - `sim/tests/test_engine_physical_invariants.py:165,180,489` — 现有 util/ops 断言。
  - `sim/tests/test_memory_ppa.py` — 现有 memory PPA 测试。

  **Acceptance criteria**:
  - `uv run pytest sim/tests/test_dram_access_pattern.py -q` 通过，包含至少 20 个 assert。
  - 覆盖所有 8 个 engine × 2 个 access_type × 2 个 frequency 的组合。

  **QA scenarios**:
  - **Happy**: pattern-aware bandwidth monotonicity；Evidence `.omo/evidence/task-9-engine-selection-p0-dram-test.json`。
  - **Failure**: 缺失 access_type → 测试失败（fail-closed）；Evidence `.omo/evidence/task-9-engine-selection-p0-dram-test-negative.txt`。

  **Commit**: YES | `test(dram): add access-pattern sensitivity and coverage tests` | `sim/tests/test_dram_access_pattern.py`

- [x] 10. 新增 LPDDR5x_7B 和 HBM2e_7B 场景

  **What to do**:
  1. 在 `sim/config/scenarios.yaml:82` 后新增两个 scenario entry：
     - `lpddr5x_7b`: model=qwen2.5-7b, seq_len=128, process_nm=12, memory type=lpddr5x, bandwidth_gbps=68.0, dram_efficiency=0.85, effective_bw_gbps=57.8, area_mm2_max=80, components required=dram_phy/pcie。
     - `hbm2e_7b`: model=qwen2.5-7b, seq_len=128, process_nm=12, memory type=hbm2e, bandwidth_gbps=410.0, dram_efficiency=0.95（HBM 效率更高）, effective_bw_gbps=389.5, area_mm2_max=120, components required=dram_phy/pcie/tsv。
  2. 确保 `memory_component_rules` 中 `lpddr5x` 和 `hbm2e` 已有规则（当前 YAML L90-105 已有，验证即可）。
   3. 新增 run_manifest 引用到 `docs/publication-manifest.yaml`。
   4. **文档化场景 `dram_efficiency` 的角色**：Todo 7（pattern-based DRAM 效率）完成后，scenario YAML 中的 `dram_efficiency` 字段仅作文档参考；实际带宽计算由 per-pattern 效率决定。

  **Must NOT do**:
  - 不修改 `scenarios.schema.Scenario` 的 Pydantic 模型。
  - 不删除或修改现有 3 个场景。

  **Parallelization**: Wave 2, Lane B | Blocked by: — | Blocks: 11 | Parallel with: 6

  **References**:
  - `sim/config/scenarios.yaml:6-82` — 现有 lpddr5_3b、onchip_7b、onchip_7b_chat。
  - `sim/config/scenarios.yaml:84-105` — memory_component_rules。
  - `sim/config/design_space.yaml:88-119` — 7nm baselines。

  **Acceptance criteria**:
  - `scenarios.yaml` 可被 `dse_scenario.load_scenario("lpddr5x_7b")` 正确加载，返回 `Scenario` Pydantic model。
  - `dse_scenario.check_requirements("lpddr5x_7b")` 无 CRITICAL 缺口。
  - 5 个场景（含现有的 3 个）的 `--scenario` CLI 都能成功启动 DSE（exit 0）。

  **QA scenarios**:
  - **Happy**: YAML 加载 → Scenario 验证 → DSE 启动；Evidence `.omo/evidence/task-10-engine-selection-p0-new-scenarios.json`。
  - **Failure**: 非法 bandwidth/constraints → 被 YAML 或 schema 拒绝；Evidence `.omo/evidence/task-10-engine-selection-p0-new-scenarios-negative.txt`。

  **Commit**: YES | `feat(scenarios): add lpddr5x_7b and hbm2e_7b intermediate scenarios` | `sim/config/scenarios.yaml`, `docs/publication-manifest.yaml`

- [x] 11. 将新场景接入 cross-validation

  **What to do**:
  1. 修改 `sim/design_space_explorer.py:1124-1130`：将 binary auto-detect 替换为基于 memory_type + seq_len 的多路检测。
  2. 新逻辑（注意：比较前对 `config["memory"]["type"].lower()` 做归一化，确保大小写不影响匹配）：
     - `memory_type == "on_chip_3d_dram" and seq_len > 256` → `onchip_7b`
     - `memory_type == "on_chip_3d_dram" and seq_len ≤ 256` → `onchip_7b_chat`
     - `memory_type == "hbm2e"` → `hbm2e_7b`
     - `memory_type == "lpddr5x"` → `lpddr5x_7b`
     - else → `lpddr5_3b`
  3. 新增 `--scenario` flag 到 legacy cross-validation 参数中。
  4. 确保 `_resolve_scenario()` 别名映射包含新场景。

  **Must NOT do**:
  - 不改变 legacy 路径未指定 `--scenario` 时的 auto-detect 行为。
  - 不引入新的 CLI flag 依赖。

  **Parallelization**: Wave 2, Lane B | Blocked by: 10 | Blocks: 12 | Parallel with: 7,8

  **References**:
  - `sim/design_space_explorer.py:1124-1130` — 当前 binary 检测。
  - `sim/design_space_explorer.py:651-676` — _resolve_scenario 别名映射。

  **Acceptance criteria**:
  - hbm2e config 被正确识别为 `hbm2e_7b`（而非 `onchip_7b` 或 `lpddr5_3b`）。
  - legacy 路径未指定 scenario 时向后兼容（lpddr5_3b 仍为默认）。
  - `uv run python sim/design_space_explorer.py --scenario hbm2e_7b --space ci-all-axes` exit 0。

  **QA scenarios**:
  - **Happy**: 5 种 config 被正确路由到 5 个 scenario；Evidence `.omo/evidence/task-11-engine-selection-p0-cross-validate.json`。
  - **Failure**: 未知 memory type → fallback 到 lpddr5_3b 并 emit warning；Evidence `.omo/evidence/task-11-engine-selection-p0-cross-validate-negative.txt`。

  **Commit**: YES | `fix(dse): wire new scenarios into cross-validation detection` | `sim/design_space_explorer.py`

- [x] 12. 运行全场景 DSE 对比

  **What to do**:
  1. 对 5 个场景（lpddr5_3b, lpddr5x_7b, hbm2e_7b, onchip_7b, onchip_7b_chat）分别运行 `--space ci-all-axes` 的 scenario DSE。
  2. 收集每个场景的 Pareto frontier 引擎排名。
  3. 生成对比表：scenario × engine → ranking position、area_mm2、tok_s。
  4. 验证引擎选择结论的连续性：从 lpddr5_3b (51.2 GB/s) → lpddr5x_7b (68 GB/s) → hbm2e_7b (410 GB/s) → onchip_7b (500 GB/s)，引擎偏好应平稳过渡而非在某个 BW 区间骤变。

  **Must NOT do**:
  - 不修改 DSE 结果数据——本 todo 仅运行和收集。
  - 不硬编码新结论到 README 中（留给 Todo 15）。

  **Parallelization**: Wave 2, Lane B | Blocked by: 1,7,8,11 | Blocks: F1-F4 | Parallel with: 9

  **References**:
  - `sim/design_space_explorer.py:711-798` — _run_scenario_dse 的现代路径。
  - `sim/dse/runner.py:341-482` — ScenarioDseRunner.run()。
  - `sim/dse/pareto.py:312-361` — MultiObjectivePareto。

  **Acceptance criteria**:
  - 全部 5 个场景 DSE exit 0。
  - 对比表中每个 scenario 至少有 1 个 engine 上榜。
  - sequential/random 效率差异在 Pareto 排名中产生可观测的影响（至少 1 个 scenario 下 KV-heavy 场景的内存绑定引擎排名下降）。
  - `ci-all-axes` 模式每个 scenario ≤ 10 分钟。

  **QA scenarios**:
  - **Happy**: 5-scenario × 8-engine 排名矩阵；Evidence `.omo/evidence/task-12-engine-selection-p0-scenario-dse.json`。
  - **Failure**: 任何 scenario DSE 出错或被 constraint 排空 → fail；Evidence `.omo/evidence/task-12-engine-selection-p0-scenario-dse-negative.txt`。

  **Commit**: YES | `evidence(dse): collect 5-scenario engine ranking comparison` | `.omo/evidence/task-12-engine-selection-p0-scenario-dse.json`

- [x] 13. 将 process_node 加入 DSE 轴

  **What to do**:
  1. 修改 `sim/config/dse_axes.yaml:10-142`：在 `axes` 区域新增 `process_node` 轴，values: `[28, 22, 12, 7]`。同时在 `bandwidth_gbps` 轴的 values 中添加 `68` 和 `410`。
  2. 新增约束：当 28nm 或 22nm 时，SRAM L2 上限为 4096KB（面积约束）。
  3. 更新 `sim/dse/hardware_builder.py`：在 `build_config()` 中正确传播 `process_node` 到 `area_model.process_node`。
  4. 确保 `design_space.yaml:90` 中的默认 `process_node: 7` 可以被 axis 覆盖。
  5. **添加约束**：当 scenario 指定了 `bandwidth_gbps` 时，DSE axis 的 `bandwidth_gbps` 值必须与 scenario 匹配（不允许轴值与 scenario 定义冲突）。

  **Must NOT do**:
  - 不改变现有 axes 的 value list。
  - 不新增 engine types。

  **Parallelization**: Wave 3 | Blocked by: 1,2,3,4 | Blocks: 14

  **References**:
  - `sim/config/dse_axes.yaml:10-142` — 当前 23 轴不含 process_node。
  - `sim/dse/hardware_builder.py:13-69` — build_config()。
  - `sim/dse/space.py:53-255` — DesignSpace 生成逻辑。

  **Acceptance criteria**:
  - `process_node` 轴生成 4 个 design point 变体（其他轴固定时）。
  - 28nm 配置下 `area_model.process_node_nm == 28.0` 且 `node_scale == (28/7)²`。
  - 7nm 配置下 `node_scale == 1.0`（不是 2.70）。
  - `ci-all-axes` 模式生成的 design point 数增加 ≤ 4× 原始数。

  **QA scenarios**:
  - **Happy**: process_node 轴在 DSE 中正确作用 + area 变化符合预期；Evidence `.omo/evidence/task-13-engine-selection-p0-node-axis.json`。
  - **Failure**: 非法 node 值被约束拒绝；Evidence `.omo/evidence/task-13-engine-selection-p0-node-axis-negative.txt`。

  **Commit**: YES | `feat(dse): add process_node as scannable DSE axis` | `sim/config/dse_axes.yaml`, `sim/dse/hardware_builder.py`, `sim/config/design_space.yaml`

- [x] 14. 运行跨节点 DSE 并生成引擎排名矩阵

  **What to do**:
  1. 对 lpddr5_3b 场景以 `--space ci-all-axes` 运行跨 4 node 的 DSE（Todo 13 加入后自动跨 node）。
  2. 同样对 onchip_7b 场景运行跨 node DSE。
  3. 生成 `scenario × node × engine` 的排名矩阵（Markdown 表格）。
  4. 验证关键假设：
     - FSA 在低 BW (lpddr5) 下在所有节点都优于 block（面积效率持久）。
     - Block 在高 BW (onchip) 下在所有节点都优于 FSA。
     - 28nm 节点的效率差异可能比 7nm 更极端（logic 缩放快于 SRAM → SRAM-heavy 设计在旧节点更吃亏）。

  **Must NOT do**:
  - 不对所有 5 个场景 × 4 个 node 做完整 DSE——仅对 lpddr5_3b 和 onchip_7b 做跨 node（其他场景固定 12nm）。

  **Parallelization**: Wave 3 | Blocked by: 13 | Blocks: 15

  **References**:
  - `sim/design_space_explorer.py:711-798` — _run_scenario_dse。
  - `sim/dse_scenario.py:475-647` — preflight + 瓶颈预测 + 交叉验证。

  **Acceptance criteria**:
  - 两场景 × 四节点 DSE exit 0，各产生 valid Pareto frontier。
  - 排名矩阵中每个 cell 都有非空 engine list。
  - 关键假设 FSA-wins-at-low-BW 在所有 4 个节点上成立。
  - 生成 evidence JSON + Markdown 报告。

  **QA scenarios**:
  - **Happy**: 跨节点排名矩阵 + 关键假设验证；Evidence `.omo/evidence/task-14-engine-selection-p0-cross-node-dse.json`。
  - **Failure**: 任一 node 下 DSE 无可行 point → fail 并标记 coverage gap；Evidence `.omo/evidence/task-14-engine-selection-p0-cross-node-dse-negative.txt`。

  **Commit**: YES | `evidence(dse): cross-node engine ranking matrix for lpddr5_3b and onchip_7b` | `.omo/evidence/task-14-engine-selection-p0-cross-node-dse.json`, markdown report

- [ ] 15. 更新决策级文档

  **What to do**:
  1. 更新 `docs/model-trust-and-release.md`：加入跨节点引擎选择的结论、bitcell 数据溯源、pattern-based DRAM 效率的方法论。
  2. 更新 README 的 "关键技术决策" 表：修正 process node 行（不再写 2.94×），补充 SRAM bitcell 数据来源。
  3. 更新 `references/area_sources.md` §7 "模型局限性"：移除 "SRAM 面积按 KB 线性叠加" 这条（已通过 bitcell 查表修复）。
  4. 标记 decision-grade 仍为 FAIL：WMMA/GMMA PE ratio 仍为 T0，多节点对比为 exploratory。
  5. 在 README "双场景技术路线" 下方加一行 "跨节点验证结论"，附 Todo 14 产生的证据链接。

  **Must NOT do**:
  - 不修改历史 dated report。
  - 不删减 Must NOT have 中的限制表述。

  **Parallelization**: Wave 3 | Blocked by: 14 | Blocks: F1-F4

  **References**:
  - `README.md:55-57` — 双场景技术路线表和关键技术决策表。
  - `docs/model-trust-and-release.md` — Todo 18 产出。
  - `references/area_sources.md:132-136` — §7 模型局限性。

  **Acceptance criteria**:
  - README 中 process_node 行显示 2.70×（非 2.94×），包含 bitcell 数据源。
  - `docs/model-trust-and-release.md` 包含跨节点引擎选择 section。
  - `references/area_sources.md` §7 的 "SRAM 线性 KB 近似" 标记为已修复。
  - `uv run ruff check .` 通过。

  **QA scenarios**:
  - **Happy**: 文档一致、无过期 claim、无未溯源声明；Evidence `.omo/evidence/task-15-engine-selection-p0-docs.json`。
  - **Failure**: 检测到仍在引用 2.94× → fail；Evidence `.omo/evidence/task-15-engine-selection-p0-docs-negative.txt`。

  **Commit**: YES | `docs(release): update engine selection conclusions with cross-node and pattern-based findings` | `README.md`, `docs/model-trust-and-release.md`, `references/area_sources.md`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [ ] F1. Plan compliance audit

  Verify every Todo 1–15 acceptance criterion against actual files/evidence/commits.

  ```bash
  uv run python scripts/verify_evidence_ledger.py \
    --plan .omo/plans/engine-selection-p0-improvements.md \
    --evidence-root .omo/evidence \
    --output .omo/evidence/final-p0-f1-plan-compliance.json
  ```

  **APPROVE iff** exit 0、Todos/F1-F4 schema 可解析、Todo 1–15 每项有匹配 commit/evidence、all blocking skipped/xfail=0。

- [ ] F2. Code quality and model-integrity review

  ```bash
  uv run ruff format --check . > .omo/evidence/final-p0-f2-code-quality.txt 2>&1
  uv run ruff check . >> .omo/evidence/final-p0-f2-code-quality.txt 2>&1
  uv run basedpyright >> .omo/evidence/final-p0-f2-code-quality.txt 2>&1
  uv run pytest sim/tests/test_engine_physical_invariants.py \
    sim/tests/test_area_cross_node.py sim/tests/test_dram_access_pattern.py \
    sim/tests/test_memory_ppa.py -q >> .omo/evidence/final-p0-f2-code-quality.txt 2>&1
  uv run python scripts/verify_model_integrity.py \
    --output .omo/evidence/final-p0-f2-code-quality.json
  ```

  **APPROVE iff** 全部命令 exit 0、blocking skip/xfail=0、verifier output `verdict=PASS`。

- [ ] F3. Real CLI/scenario/replay QA

  ```bash
  uv run python scripts/release_gate.py \
    --profile experimental \
    --clean-checkout \
    --exercise-legacy \
    --exercise-all-workloads \
    --space ci-all-axes \
    --output .omo/evidence/final-p0-f3-manual-qa.json
  ```

  **APPROVE iff** exit 0、`legacy_failures=[]`、`workload_failures=[]`、`coverage.missing=[]`、`errors=0`、`experimental_gate=pass`。

- [ ] F4. Scope and evidence fidelity

  ```bash
  uv run python scripts/verify_scope.py \
    --plan .omo/plans/engine-selection-p0-improvements.md \
    --baseline-commit "$(git merge-base HEAD origin/main)" \
    --publication-manifest docs/publication-manifest.yaml \
    --output .omo/evidence/final-p0-f4-scope-fidelity.json
  ```

  **APPROVE iff** exit 0、`forbidden_dependencies=[]`、`ultraresearch_changes=[]`、`historical_report_changes=[]`、`unbound_current_claims=[]`。

## Commit strategy

- 使用 conventional commits；每个 todo 的 implementation+test 为一个原子 commit。
- 不 amend、不 squash 已发布历史；修复使用独立 `fix(...)` commit。
- 每个 commit 只包含该 todo 的 Files 列表与其证据；`.omo/ultraresearch/20260723-vla-models/sources/` 永不 stage。
- Wave 1 完成后创建 evidence manifest；Wave 2 两条 lane 独立完成；Wave 3 只依赖 Wave 1 完成。
- 只在全部 F1-F4 通过后才标记计划完成。

## Success criteria

- `AreaModel.estimate()` 中 SRAM 面积从 bitcell 查表计算、logic 面积从 node_scale 计算——两条路径独立可验证。
- 28nm/22nm/12nm/7nm 四个候选节点的 SRAM bitcell 数据均有公开来源引用。
- `MemoryTopology`、`CalibrationRef` 和 `legacy_result` 不再硬编码 12nm——process_node 从配置正确传播。
- `MemoryAccessPattern` 携带 `access_type`，所有权重 DMA 标记 SEQUENTIAL、所有 KV 访问标记 RANDOM。
- `mac_engine.py` 和各 engine 的 DMA 计算根据 access_type 应用不同的 DRAM 效率。
- `kv_cache.py` 不再使用固定 `dram_access_cycles=80`，改为基于带宽公式和 random 效率。
- `scenarios.yaml` 新增 `lpddr5x_7b` 和 `hbm2e_7b`，cross-validation 自动检测全部 5 个场景。
- 跨 5 个 BW level (51.2→68→410→500 GB/s) 的引擎排名形成连续梯度，验证 "低 BW 选 FSA、高 BW 选 block" 的结论区间。
- **排名指标：** 排名以原始 `tok/s` 为主（primary），`tok/s/mm²` 为辅助验证（secondary check）。
- **跨 4 个工艺节点：** 已测试场景（lpddr5_3b, onchip_7b）下排名方向一致（FSA 在低 BW 胜出，block 在高 BW 胜出）；12nm 排名作为参考，其他节点标记 exploratory。
- F1、F2、F3、F4 全部以 `verdict=PASS` 和 exit 0 完成。
