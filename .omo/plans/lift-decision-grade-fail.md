# lift-decision-grade-fail - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** README 不再显示 FAIL；最后一个 T0 参数升级到 T1；频率-节点绑定从"探索性"变为有公开源引的 T1；DRAM 效率从"无注册"变为有 JEDEC 源引的 T1。决策级代码门控（release_gate）仍不通过（需要全部参数 T3）——诚实标注、不谎报。

**Why this approach:** 不做虚假的 "authoritative"（需要全部参数 T3，`runner.py` 代码实现决定，`model-trust-and-release.md` 旧写的 T2+ 需要修正），而是消除剩余 T0、补上公开源引、诚实重标决策级状态。ARM PL330 DMA / JEDEC LPDDR5 / TSMC 产品参考都是公开可查的锚点。

**What it will NOT do:** 不让 release_gate.py 通过。不把决策级标为 PASS。不修改任何引擎公式或面积模型。不碰 DSE 扫描。

**Effort:** Medium
**Risk:** Low — 仅修改 YAML 条目、evaluate.py 映射、和文档文字；不涉及性能模型改动
**Decisions to sanity-check:** 频率上限源引是否足够权威（产品参考 vs PDK 数据）；DMA 描述符 5-cycle 开销从 ARM PL330 手册能否充分支撑 T1

Your next move: approve 本计划即可开始执行。Full execution detail follows below.

---

> TL;DR (machine): Medium, Low. T0→T1 for tensor_core descriptor overhead + fix broken tests; add 4 frequency-bound + 3 DRAM-efficiency calibration entries (ranges aligned to dse_axes.yaml); wire into evaluate.py; update docs to remove README FAIL clause and correct T2→T3 model-trust inconsistency.

## Scope
### Must have
- 将最后一个 T0 参数 `tensor_core_descriptor_overhead` 升级到 T1，添加公开源引
- 在 `references/calibration/parameters.yaml` 中新增 4 个频率-节点校准条目（`max_freq_7nm/12nm/22nm/28nm`），全部 T1，含 source_uri
- 在 `references/calibration/parameters.yaml` 中新增 3 个 DRAM 效率校准条目（`dram_efficiency`, `dram_efficiency_random_bw`, `random_latency_penalty_cycles`），全部 T1，含 source_uri
- 更新 `sim/calibration/evaluate.py`，将新的频率和 DRAM 条目接入 `calibration_ids_for_design_point()` 和 `_actual_value()`
- 更新 `docs/model-trust-and-release.md` Decision-Grade State 章节：记录 T0→T1 升级，移除 FAIL 原因
- 更新 `README.md` §1.4：移除 "FAIL（频率-节点绑定为探索性结论，多节点覆盖不完全）"，替换为诚实的改进状态
- 更新 `docs/NPU_Engines_Architecture_Guide.md` 信任声明

### Must NOT have (guardrails, anti-slop, scope boundaries)
- 不试图通过 `release_gate.py --profile decision-grade`（需要全部 T3，不可行）
- 不修改 DSE 引擎公式、AreaModel、MACEngine ABC 接口
- 不修改 `dse_axes.yaml` 的频率约束值本身——仅添加源引
- 不修改历史 dated report
- 不修改 `EngineResult` dataclass 或 CLI 契约
- 不将 decision-grade 标记为 PASS——仅移除 README 的 FAIL 子句，保持诚实限制说明

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: **tests-after** — 现有测试覆盖 YAML 解析、注册表完整性、evaluate.py 逻辑；新增条目由 `test_calibration_registry.py` 和 `test_calibration_evaluate.py` 的已有断言自动验证
- Parameter registry integrity: `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` — 确保所有新增条目可解析、T1 source_uri 非空、EXPECTED_IDS 匹配
- evaluate.py regression: `uv run python sim/calibration/evaluate.py` exit 0
- YAML syntax: `python3 -c "import yaml; yaml.safe_load(open('references/calibration/parameters.yaml'))"` exit 0
- Documentation consistency: grep 验证 README、model-trust-and-release、architecture guide 中不再出现已删除的 FAIL clause 或过时的 T0 声明
- Evidence: `.omo/evidence/task-<N>-lift-decision-grade-fail.<ext>`；每条证据同时记录命令、exit code、git commit、config/hardware/lock digests

## Execution strategy
### Parallel execution waves
> Target 5-8 todos per wave. Fewer than 3 (except the final) means you under-split.

```
Wave 1 — Calibration Entries (4 todos, parallel where possible)
├── Todo 1: Upgrade tensor_core_descriptor_overhead T0→T1
├── Todo 2: Add frequency-bound calibration entries (4 new entries)
├── Todo 3: Add DRAM efficiency calibration entries (3 new entries)
└── Todo 4: Wire new entries into evaluate.py + update EXPECTED_IDS

Wave 2 — Documentation Updates (3 todos, sequential)
├── Todo 5: Update model-trust-and-release.md Decision-Grade State
├── Todo 6: Update README §1.4 FAIL clause
└── Todo 7: Update NPU_Engines_Architecture_Guide.md trust disclaimer
```

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1 | — | 5 | 2, 3 |
| 2 | — | 4 | 1, 3 |
| 3 | — | 4 | 1, 2 |
| 4 | 2, 3 | 5 | — |
| 5 | 1, 4 | 6 | — |
| 6 | 5 | 7 | — |
| 7 | 6 | F1-F4 | — |

Critical path: `(1||2||3) → 4 → 5 → 6 → 7 → F1-F4`

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. 将 tensor_core_descriptor_overhead 从 T0 升级到 T1

  **What to do**:
   1. 在 `references/calibration/parameters.yaml` 中修改 `tensor_core_descriptor_overhead` 条目（第 248-260 行）：
      - `source_uri` 从 `null` 改为 ARM PL330 DMA PrimeCell 公开手册链接（https://developer.arm.com/documentation/ddi0424/），描述符读取开销 4-8 周期是本参数 5-cycle 值的数量级代理
      - `trust_level` 从 `T0` 改为 `T1`
      - `description` 补充类比说明："DMA descriptor issue overhead range (4–8 cycles) from ARM PL330 DMA PrimeCell; applied as order-of-magnitude proxy for Tensor Core per-sub-tile descriptor setup. The 5-cycle value is a conservative mid-point."
   2. 更新 `references/calibration/parameters.yaml` 顶部 header comment（第 1-10 行附近）：
      - 将 "Parameters marked T0/T1 are engineering assumptions" 改为 "Parameters marked T1 are engineering assumptions"（或等效表述，反映 T0 已消除）
      - 将 "Decision-grade ranking requires every Pareto-driving parameter to be T2+" 改为 "T3+"（与 `model-trust-and-release.md:51` 和 `release_gate.py` 一致）
   3. 更新 `sim/tests/test_calibration_evaluate.py:142-161` 的信任门测试，采用**单一方案：合成/手工打造的 T0 注册表条目**（不再提供两种备选修复）：
      - 在测试内构造一个合成 T0 校准条目（通过 `CalibrationRegistry` 构造函数注入或 monkeypatch），用于信任门测试，保留测试意图（decision-grade 拒绝 T0 / exploratory 允许 T0），不依赖真实 T0 参数
      - 重命名测试：`test_trust_gate_t2_required_with_t0_fails` → `test_trust_gate_t2_required_with_synthetic_t0_fails`；`test_trust_gate_exploratory_allows_t0` → `test_trust_gate_exploratory_allows_synthetic_t0`；相应更新 docstring
   4. 重写 `sim/tests/test_calibration_evaluate.py:230-319` 的 `test_runner_exploratory_marks_t0_points_exploratory` 和 `test_runner_decision_grade_fails_on_t0`：通过 `CalibrationRegistry` 构造函数或 monkeypatch 注入合成 T0 校准条目，使它们在真实 `tensor_core_descriptor_overhead` 升级为 T1 后仍然通过
   5. 验证 `evaluate.py` 中已有的 `_actual_value()` 提取逻辑仍正确（第 155-156 行，从 `dma.descriptor_overhead_cycles` 读取，默认 5）
   6. 添加 `test_tensor_core_overhead_trust()` 到 `sim/tests/test_calibration_evaluate.py`：验证该条目 trust_level=T1、source_uri 非空、calibration_range [0, 10] 保持
   7. 验证 `sim/tests/test_calibration_evaluate.py` 端到端通过且注册表中**零真实 T0 参数**（合成 T0 仅存在于测试内部）

  **Must NOT do**:
   - 不修改参数值（保持 5 cycles）
   - 不修改 `dse_axes.yaml` 或引擎公式
   - 不修改 `tensor_core_engine.py` 的 `DEFAULT_DESCRIPTOR_OVERHEAD_CYCLES`

  **Parallelization**: Wave 1 | Blocked by: — | Blocks: 5 | Parallel with: 2, 3

  **References**:
   - `references/calibration/parameters.yaml:248-260` — 当前 T0 条目
   - `sim/calibration/evaluate.py:88,155-156` — 已有的 calibration ID 和 _actual_value 提取逻辑
   - ARM PL330 DMA PrimeCell 公开文档：https://developer.arm.com/documentation/ddi0424/ — 描述符读取 ~4-8 周期是 DMA 控制器的业界共识；或 Synopsys DesignWare DW_axi_dmac
   - `sim/tests/test_calibration_evaluate.py:81,122,146,150,157` — 现有测试点

  **Acceptance criteria**:
   - `parameters.yaml` 中 `tensor_core_descriptor_overhead` 的 `trust_level: T1` 且 `source_uri` 非空
   - `uv run pytest sim/tests/test_calibration_evaluate.py sim/tests/test_calibration_registry.py -q` 通过
   - `uv run pytest sim/tests/test_calibration_evaluate.py::test_runner_exploratory_marks_t0_points_exploratory sim/tests/test_calibration_evaluate.py::test_runner_decision_grade_fails_on_t0 -q` 通过
   - 所有 trust-gate 测试（`test_trust_gate_*`、`test_runner_*`）均使用合成 T0 测试夹具，不依赖真实 T0 参数
   - `references/calibration/parameters.yaml` header 不再出现 "T0/T1" 或 "T2+" 字样
   - `uv run python sim/calibration/evaluate.py` exit 0
   - 注册表检查通过：
     ```bash
     PYTHONPATH=sim uv run python -c "
     from calibration.registry import CalibrationRegistry
     r = CalibrationRegistry.from_yaml()
     e = r.get('tensor_core_descriptor_overhead')
     assert e.trust_level.value == 'T1', e.trust_level
     assert e.source_uri, 'source_uri missing'
     print('PASS: tensor_core_descriptor_overhead is T1 with source_uri')
     " > .omo/evidence/task-1-lift-decision-grade-fail-registry-check.txt 2>&1
     test $? -eq 0
     ```

  **QA scenarios**:
   - **Happy**: T1 + source_uri 验证通过；Evidence `.omo/evidence/task-1-lift-decision-grade-fail-tc-trust.json`
   - **Failure**: 故意将 trust_level 改回 T0 → 断言失败；Evidence `.omo/evidence/task-1-lift-decision-grade-fail-tc-trust-negative.txt`

  **Commit**: YES | `calibrate(tensor_core): upgrade descriptor_overhead T0→T1 with DMA controller source` | `references/calibration/parameters.yaml`, `sim/tests/test_calibration_evaluate.py`

- [x] 2. 添加频率-节点绑定校准条目到 parameters.yaml

  **What to do**:
   1. 在 `references/calibration/parameters.yaml` 中新增 4 个校准条目（上限约束语义：每个条目的 `value` 是该节点的最大可行频率，`calibration_range` 是该节点允许的频率范围，TrustGate 验证 design point 的实际 `frequency_mhz` 在其范围内）：
      - `max_freq_28nm`: value=600, unit=MHz, trust_level=T1, calibration_range=[200, 600], range_min=200.0, range_max=600.0, source_uri 引用 TSMC 28HPM 公开性能数据或产品参考（如 TPUv1 700MHz @28nm, RK356x ~500-800MHz）
      - `max_freq_22nm`: value=800, unit=MHz, trust_level=T1, calibration_range=[400, 800], range_min=400.0, range_max=800.0, source_uri 引用 22nm 产品参考（如 RK1828 22nm 时钟）
      - `max_freq_12nm`: value=1200, unit=MHz, trust_level=T1, calibration_range=[800, 1200], range_min=800.0, range_max=1200.0, source_uri 引用 TSMC 12FFC 公开数据或产品参考（如 RK3588 ~1.8GHz, MediaTek Dimensity 系列）
      - `max_freq_7nm`: value=2000, unit=MHz, trust_level=T1, calibration_range=[800, 2000], range_min=800.0, range_max=2000.0, source_uri 引用 TSMC N7 公开数据或产品参考（如 Apple A13 ~2.6GHz, NVIDIA A100 ~1.4GHz）
   2. 每个条目的 `source_hash` 设为 `null`（T1 不需要）；`description` 说明范围来源
   3. 每个新条目的 `range_min`/`range_max` 必须显式设置，与 `calibration_range` 的 [min, max] 一致，确保 `TrustGate.is_in_range()` 能正确检查 design point 频率。
   4. **calibration_range 严格对齐 `dse_axes.yaml:262-288` 的 `node_*_frequency_bound` 约束（reason codes 见 290-303）**（28nm [200,600], 22nm [400,800], 12nm [800,1200], 7nm [800,2000]）

  **Must NOT do**:
   - 不修改 `dse_axes.yaml` 的频率约束值
   - 不修改 DSE 引擎公式或面积模型

  **Parallelization**: Wave 1 | Blocked by: — | Blocks: 4 | Parallel with: 1, 3

  **References**:
   - `references/calibration/parameters.yaml` — 添加条目处（建议在现有 PE area 条目后）
   - `sim/config/dse_axes.yaml:262-288` — `node_*_frequency_bound` 约束（频率上限）
   - `.omo/evidence/investigate-all-engines-cross-node-freq.md` — 频率感知分析证据
   - `references/area_sources.md` §1 — TPUv1 28nm MXU 700MHz（现有锚点）
   - 产品参考：Apple A13 7nm ~2.6GHz, Snapdragon 8cx 7nm ~2.84GHz, RK3588 12nm, RK1828 22nm

  **Acceptance criteria**:
   - `parameters.yaml` 包含 4 个新条目，全部 `trust_level: T1`，`source_uri` 非空
   - `python3 -c "import yaml; yaml.safe_load(open('references/calibration/parameters.yaml'))"` exit 0
   - 每个 `max_freq_*nm` 条目的 `range_min`/`range_max` 与 `dse_axes.yaml` 对应 `node_*_frequency_bound` 的 [min, max] 严格一致；验证脚本：
     ```bash
     python3 -c "
     import yaml
     with open('references/calibration/parameters.yaml') as f:
         entries = {e['calibration_id']: e for e in yaml.safe_load(f)['parameters']}
     with open('sim/config/dse_axes.yaml') as f:
         axes = yaml.safe_load(f)
     freq_bounds = {}
     for c in axes['constraints']:
         if c['name'].startswith('node_') and c['name'].endswith('_frequency_bound'):
             fmin, fmax = c['require']['frequency_mhz'][0], c['require']['frequency_mhz'][-1]
             freq_bounds[c['name']] = (fmin, fmax)
     node_bounds = {
         'max_freq_28nm': freq_bounds['node_28_frequency_bound'],
         'max_freq_22nm': freq_bounds['node_22_frequency_bound'],
         'max_freq_12nm': freq_bounds['node_12_frequency_bound'],
         'max_freq_7nm': freq_bounds['node_7_frequency_bound'],
     }
     for cid, bounds in node_bounds.items():
         e = entries[cid]
         assert e['range_min'] == bounds[0], f'{cid} range_min mismatch'
         assert e['range_max'] == bounds[1], f'{cid} range_max mismatch'
     print('PASS: frequency bounds aligned')
     " > .omo/evidence/task-2-lift-decision-grade-fail-freq-bounds.json 2>&1
     test $? -eq 0
     ```
   - `uv run pytest sim/tests/test_calibration_registry.py -q` 通过（需先更新 EXPECTED_IDS — 见 Todo 4）

  **QA scenarios**:
   - **Happy**: 4 个新条目解析 + T1 验证；Evidence `.omo/evidence/task-2-lift-decision-grade-fail-freq-params.json`
   - **Failure**: 缺条目 → 断言失败；Evidence `.omo/evidence/task-2-lift-decision-grade-fail-freq-params-negative.txt`

  **Commit**: YES | `calibrate(freq): add per-node frequency bound calibration entries with product references` | `references/calibration/parameters.yaml`

- [x] 3. 添加 DRAM 效率校准条目到 parameters.yaml

  **What to do**:
   1. 在 `references/calibration/parameters.yaml` 中新增 3 个校准条目：
      - `dram_efficiency`: value=0.85, unit=ratio, trust_level=T1, calibration_range=[0.80, 0.90], range_min=0.80, range_max=0.90, source_uri 引用 JEDEC LPDDR5 tRFCpb=140ns/tREFI=3900ns（per-bank refresh ~3.6%）+ 控制器调度余量分析
      - `dram_efficiency_random_bw`: value=0.50, unit=ratio, trust_level=T1, calibration_range=[0.40, 0.60], range_min=0.40, range_max=0.60, source_uri 引用公开 LPDDR5 随机访问效率研究（page-hit rate 约 50% 的场景）
      - `random_latency_penalty_cycles`: value=40, unit=cycles, trust_level=T1, calibration_range=[30, 60], range_min=30.0, range_max=60.0, source_uri 引用 JEDEC LPDDR5 tRC≈48ns @1GHz
   2. 所有条目的 `source_hash` 设为 `null`（需硅验证才能到 T2+）
   3. 每个新条目的 `range_min`/`range_max` 必须显式设置，与 `calibration_range` 的 [min, max] 一致，确保 `TrustGate.is_in_range()` 能正确检查 DRAM 参数。
   4. `description` 中记录推导链：tRFCpb/tREFI → 3.6% → 0.964（保留）→ 再乘控制器/命令总线/bank 冲突余量 → 0.85
   5. **注意**：`sim/contracts/hardware.py:302-320` 的 `DEFAULT_DRAM_EFFICIENCY_RANDOM_BW_PROVENANCE` 和 `DEFAULT_RANDOM_LATENCY_PENALTY_PROVENANCE` 仍是 T0。本 Todo 仅升级 YAML 注册表；保留代码中的默认 provenance 不变，在 Todo 5 的文档中注明此差异。

  **Must NOT do**:
   - 不修改 DRAM 效率的实际数值（保持 0.85 / 0.50 / 40）

  **Parallelization**: Wave 1 | Blocked by: — | Blocks: 4 | Parallel with: 1, 2

  **References**:
   - `sim/contracts/hardware.py:170-194` — MemoryConfig 定义 + JEDEC tRFCpb/tREFI 注释
   - `sim/contracts/hardware.py:290-320` — Provenance 默认值（T1/T0/T0）
   - `sim/config/npu_config.yaml:90-92` — 当前默认值
   - JEDEC LPDDR5: tRFCpb=140ns, tREFI=3900ns → per-bank refresh ~3.6%
   - 公开 DRAM 效率研究：DRAM controller scheduling overhead typically 5-10%

  **Acceptance criteria**:
   - `parameters.yaml` 包含 3 个新条目，全部 `trust_level: T1`，`source_uri` 非空
   - 每个 DRAM 条目的 `range_min`/`range_max` 与 `calibration_range` 的 [min, max] 严格一致
   - DRAM 范围边界验证脚本通过：
     ```bash
     python3 -c "
     import yaml
     with open('references/calibration/parameters.yaml') as f:
         entries = {e['calibration_id']: e for e in yaml.safe_load(f)['parameters']}
     expected = {
         'dram_efficiency': (0.80, 0.90),
         'dram_efficiency_random_bw': (0.40, 0.60),
         'random_latency_penalty_cycles': (30.0, 60.0),
     }
     for cid, (lo, hi) in expected.items():
         e = entries[cid]
         assert e['range_min'] == lo, f'{cid} range_min mismatch: {e[\"range_min\"]} != {lo}'
         assert e['range_max'] == hi, f'{cid} range_max mismatch: {e[\"range_max\"]} != {hi}'
     print('PASS: DRAM entry range bounds verified')
     " > .omo/evidence/task-3-lift-decision-grade-fail-dram-bounds.json 2>&1
     test $? -eq 0
     ```
   - YAML 语法有效
   - 数值与 `npu_config.yaml` 一致
   - `uv run pytest sim/tests/test_calibration_registry.py -q` 通过（需先更新 EXPECTED_IDS — 见 Todo 4）

  **QA scenarios**:
   - **Happy**: 3 个新条目解析 + T1 验证；Evidence `.omo/evidence/task-3-lift-decision-grade-fail-dram-params.json`
   - **Failure**: 数值不匹配 npu_config.yaml → 断言失败；Evidence `.omo/evidence/task-3-lift-decision-grade-fail-dram-params-negative.txt`

  **Commit**: YES | `calibrate(dram): register DRAM efficiency params with JEDEC LPDDR5 timing sources` | `references/calibration/parameters.yaml`

- [x] 4. 将新校准条目接入 evaluate.py 并更新注册表

  **What to do**:
   1. 更新 `sim/calibration/evaluate.py`：
      - 在 `calibration_ids_for_design_point()` 中添加频率-节点 ID：根据 design point 的 `process_node` 值添加对应的 `max_freq_{node}nm`
      - 在 `calibration_ids_for_design_point()` 中添加 DRAM ID：`dram_efficiency`, `dram_efficiency_random_bw`, `random_latency_penalty_cycles`（适用于所有 engine type）
      - 在 `_actual_value()` 中添加对应提取逻辑：
        - `max_freq_*nm`：从 `hw_config["mac_engine"]["frequency_mhz"]` 读取；如果缺失，使用与 `parameters.yaml` 中该条目一致的硬编码默认值（`max_freq_28nm`→600, `max_freq_22nm`→800, `max_freq_12nm`→1200, `max_freq_7nm`→2000）。语义：TrustGate 检查设计点的实际频率是否在 `calibration_range` 内。
        - `dram_efficiency`：从 `memory.dram_efficiency` 读取，默认 0.85
        - `dram_efficiency_random_bw`：从 `memory.dram_efficiency_random_bw` 读取，默认 0.50
        - `random_latency_penalty_cycles`：从 `memory.random_latency_penalty_cycles` 读取，默认 40
      - 注意：`max_freq_*nm` 的 `calibration_ids_for_design_point()` 应按 design point 的 `process_node` 值只添加匹配的那一个 ID（如 `process_node=12` → `max_freq_12nm`），而非全部 4 个
   2. 更新 `sim/tests/test_calibration_registry.py` 的 `EXPECTED_IDS`，添加 7 个新 ID（4 频率 + 3 DRAM）

  **Must NOT do**:
   - 不修改 TrustGate 逻辑或 `_trust_rank` 排序
   - 不修改现有的校准 ID 提取逻辑

  **Parallelization**: Wave 1 | Blocked by: 2, 3 | Blocks: 5

  **References**:
   - `sim/calibration/evaluate.py:57-88` — `calibration_ids_for_design_point()` 当前实现
   - `sim/calibration/evaluate.py:102-156` — `_actual_value()` 提取逻辑
   - `sim/tests/test_calibration_registry.py:36-70` — `EXPECTED_IDS` 集合
   - `sim/calibration/schema.py` — CalibrationEntry schema 定义

  **Acceptance criteria**:
   - `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` 通过
   - `uv run python sim/calibration/evaluate.py` exit 0
   - 对于包含 `process_node=12` 和 `memory.dram_efficiency=0.85` 的 design point，`calibration_ids_for_design_point()` 返回包含 `max_freq_12nm` 和 `dram_efficiency`
   - `uv run ruff check sim/calibration/evaluate.py` 通过

  **QA scenarios**:
   - **Happy**: evaluate.py exit 0 + 新 ID 正确映射；Evidence `.omo/evidence/task-4-lift-decision-grade-fail-evaluate.json`
   - **Failure**: 故意从 EXPECTED_IDS 移除一个 ID → 断言失败；Evidence `.omo/evidence/task-4-lift-decision-grade-fail-evaluate-negative.txt`

  **Commit**: YES | `feat(calibration): wire frequency-bound and DRAM-efficiency params into trust gate` | `sim/calibration/evaluate.py`, `sim/tests/test_calibration_registry.py`

- [x] 5. 更新 model-trust-and-release.md Decision-Grade State

  **What to do**:
   1. 更新 `docs/model-trust-and-release.md` 第 244-258 行的 Decision-Grade State 章节：
      - 记录 `tensor_core_descriptor_overhead` T0→T1 升级
      - 记录新增 4 个频率-节点校准条目 T1
      - 记录新增 3 个 DRAM 效率校准条目 T1
      - 移除 "tensor_core_descriptor_overhead 仍是 T0" 的句子
      - 修正第 28 行：`Run Trust Levels` 表格中 `authoritative` 行的 "all ranking parameters T2+ and in range" 改为 "T3+ and in range"（与 line 51 的 decision-grade 阈值、runner.py 的 T3→authoritative 映射、release_gate.py 的 authoritative 要求保持一致）
      - 修正第 51 行："Every Pareto-driving parameter must be T2+" → "T3+"。对齐 `scripts/release_gate.py:137-142`（decision-grade 要求结果 trust_level 为 authoritative）和 `sim/dse/runner.py:323-326`（T3 落入 authoritative 分支）
      - 修正第 58 行：将 "Because the current calibration registry keeps several ranking drivers at T0/T1" 中的 "T0/T1" 改为 "T1"（全部 T0 已消除后，仅存在 T1 及以下已不再准确）
      - 修正第 65 行（release_gate 代码块注释）：将 `# Expected to fail until T2+ evidence is added for T0/T1 parameters.` 改为 `# Expected to fail until T3+ evidence is available for remaining T1 parameters.`（反映 T0 已消除、决策级需要 T3+）
      - 修正第 257 行（Decision-Grade State 代码块注释）：将 `# Expected: FAIL — T0/T1 parameters remain ...` 更新为反映当前状态的注释（T0 已消除、tensor_core_descriptor_overhead 已 T1、decision-grade 仍因需要全部 T3+ 而失败）
      - 修正 `scripts/release_gate.py:7-8` 的模块 docstring：将 "requires every ranking-driving parameter to be T2+" 改为 "T3+"，与代码中 `trust_level == authoritative` 的实际要求保持一致
      - 明确说明：全部 T0 参数已消除，所有校准条目至少 T1
      - 保留诚实限制：频率和 DRAM 基于公开代理/架构推导，非硅测量；`release_gate.py --profile decision-grade` 仍不通过（需要全部 T3）
   2. 更新 `docs/model-trust-and-release.md` 第 249 行 Decision-Grade State 中全引擎跨节点覆盖条目的信任等级描述：从 "探索性" 改为 "T1 — 基于公开产品频率参考与 TSMC 节点性能特征"

  **Must NOT do**:
   - 不修改 DRAM 效率模式化方法章节（§Pattern-Based DRAM Efficiency）——数值和推导不变
   - 不谎称决策级通过

  **Parallelization**: Wave 2 | Blocked by: 4 | Blocks: 6

  **References**:
   - `docs/model-trust-and-release.md:28` — Run Trust Levels authoritative 行
   - `docs/model-trust-and-release.md:244-258` — Decision-Grade State 当前文本
   - `docs/model-trust-and-release.md:51` — decision-grade 阈值说明
   - `docs/model-trust-and-release.md:58` — T0/T1 参数存在声明
   - `docs/model-trust-and-release.md:65` — release_gate 示例注释
   - `docs/model-trust-and-release.md:257` — Decision-Grade State 代码块注释
   - `docs/model-trust-and-release.md:249` — Decision-Grade State 中跨节点覆盖信任等级描述
   - `scripts/release_gate.py:7-8` — decision-grade docstring（当前写 T2+，应改为 T3+）
   - Todos 1-4 的产出（更新后的 parameters.yaml、evaluate.py）

  **Acceptance criteria**:
   - 文本中不再出现 "tensor_core_descriptor_overhead 仍是 T0"
   - 文本明确记录所有 WMMA/GMMA/TensorCore 周期参数已 T1
   - 文本明确记录频率节点和 DRAM 效率已 T1（含 source_uri）
   - 文本诚实标注非硅测量限制
   - docs/code 一致性检查通过：
     ```bash
      grep -n "requires authoritative trust level" scripts/release_gate.py && \
      grep -n "authoritative" sim/dse/runner.py
      ```
    - `docs/model-trust-and-release.md:28` 不再出现 "T2+"（authoritative 行已改为 T3+）
    - `scripts/release_gate.py:7-8` 的 docstring 不再出现 "T2+"（已改为 T3+）
    - `docs/model-trust-and-release.md:65` 和 `:257` 的代码块注释不再出现 "T2+" 或 "T0/T1 parameters"
    - `uv run ruff check .` 通过

  **QA scenarios**:
   - **Happy**: 文档一致，无过时 T0 声明；Evidence `.omo/evidence/task-5-lift-decision-grade-fail-trust-doc.json`
   - **Failure**: 仍在引用 T0 → 断言失败；Evidence `.omo/evidence/task-5-lift-decision-grade-fail-trust-doc-negative.txt`

   **Commit**: YES | `docs(trust): update decision-grade state after T0 elimination and calibration upgrade` | `docs/model-trust-and-release.md`, `scripts/release_gate.py`

- [x] 6. 更新 README §1.4 移除 FAIL 子句

  **What to do**:
   1. 更新 `README.md` 第 78 行 `跨节点验证结论` 表格行的 `决策级状态` 字段：
      - 移除 `**FAIL（频率-节点绑定为探索性结论，多节点覆盖不完全）**`
      - 替换为：`**已升级** — 全部 T0 参数已消除；频率节点绑定 + DRAM 效率已引入 T1 源引（基于 JEDEC LPDDR5 时序 + TSMC 节点产品参考）。需更高信任等级证据（T3+）方可标记为决策级就绪。`
   2. 同步更新第 67 行的免责声明：将 "包含 T0/T1 参数" 改为 "包含 T1 参数"（全部 T0 已消除后不再准确）
   3. 检查 README 全文，确认不再出现旧 FAIL clause 或过时的 "WMMA/GMMA PE 比仍 T0" 表述

  **Must NOT do**:
   - 不将决策级标记为 PASS
   - 不修改 §1.4 场景技术路线的数值（引擎/面积/tok/s）

  **Parallelization**: Wave 2 | Blocked by: 5 | Blocks: 7

  **References**:
   - `README.md:78` — 跨节点验证结论行
   - `README.md:67` — T0/T1 参数声明（可能需要微调措辞）

  **Acceptance criteria**:
   - `README.md` 中不出现 "FAIL（频率-节点绑定为探索性结论" 字符串
   - 新状态文字明确说明改进和限制
   - `grep -r "WMMA/GMMA PE 比仍 T0" README.md` 无结果
   - `uv run ruff check .` 通过（README 是 markdown，ruff 不检查，但确保无意外引入的代码错误）

  **QA scenarios**:
   - **Happy**: README 无 FAIL clause、新状态正确；Evidence `.omo/evidence/task-6-lift-decision-grade-fail-readme.json`
   - **Failure**: 旧 FAIL clause 残留 → 断言失败；Evidence `.omo/evidence/task-6-lift-decision-grade-fail-readme-negative.txt`

  **Commit**: YES | `docs(readme): remove FAIL clause after calibration upgrade; note remaining limits` | `README.md`

- [x] 7. 更新 NPU_Engines_Architecture_Guide.md 信任声明

  **What to do**:
   1. 更新 `docs/NPU_Engines_Architecture_Guide.md` 第 8 行的信任声明：
      - 将 `tensor_core_descriptor_overhead` 从 "T0/T1" 列表移除（现已 T1）
      - 添加频率-节点绑定和 DRAM 效率已升级 T1 的说明
   2. 检查全文，更新任何引用 "WMMA/GMMA 周期与 PE 参数已于 2026-07-31 校准升级 T1" 的附近文本，补充 TensorCore/Freq/DRAM 升级

  **Must NOT do**:
   - 不修改引擎性能数据或排名
   - 不修改架构指南中的数值表格

  **Parallelization**: Wave 2 | Blocked by: 6 | Blocks: F1-F4

  **References**:
   - `docs/NPU_Engines_Architecture_Guide.md:8` — 当前信任声明
   - `docs/NPU_Engines_Architecture_Guide.md:31,186,191` — 可能有旧数值引用

  **Acceptance criteria**:
   - 信任声明反映最新校准状态（无过时的 T0 声明）
   - `uv run ruff check .` 通过

  **QA scenarios**:
   - **Happy**: 信任声明更新正确；Evidence `.omo/evidence/task-7-lift-decision-grade-fail-arch-guide.json`
   - **Failure**: 仍引用过时状态 → 断言失败；Evidence `.omo/evidence/task-7-lift-decision-grade-fail-arch-guide-negative.txt`

  **Commit**: YES | `docs(arch): update trust disclaimer after calibration upgrade` | `docs/NPU_Engines_Architecture_Guide.md`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [x] F1. Plan compliance audit

  Verify every Todo 1–7 acceptance criterion against actual files/evidence/commits.

  ```bash
  uv run python scripts/verify_evidence_ledger.py \
    --plan .omo/plans/lift-decision-grade-fail.md \
    --evidence-root .omo/evidence \
    --output .omo/evidence/final-lift-fail-f1-plan-compliance.json
  ```

  **APPROVE iff** exit 0、Todos 1–7 每项有匹配 commit/evidence、all blocking skipped/xfail=0。

- [x] F2. Code quality and calibration-integrity review

  ```bash
  uv run ruff format --check . > .omo/evidence/final-lift-fail-f2-code-quality.txt 2>&1
  uv run ruff check . >> .omo/evidence/final-lift-fail-f2-code-quality.txt 2>&1
  uv run basedpyright >> .omo/evidence/final-lift-fail-f2-code-quality.txt 2>&1
  uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py \
    sim/tests/test_engine_result_contract.py sim/tests/test_engines.py -q \
    >> .omo/evidence/final-lift-fail-f2-code-quality.txt 2>&1
  uv run python sim/calibration/evaluate.py > .omo/evidence/final-lift-fail-f2-evaluate.txt 2>&1
  uv run python scripts/verify_model_integrity.py \
    --output .omo/evidence/final-lift-fail-f2-code-quality.json
  ```

  **APPROVE iff** 全部命令 exit 0、verifier output `verdict=PASS`、evaluate.py 无 crash。

- [x] F3. Documentation consistency audit

  Run terminal QA:
  ```bash
  # Verify no stale FAIL clause or old T0 claims remain (incl. T0/T1 phrasings)
   grep -r "频率-节点绑定为探索性结论，多节点覆盖不完全\|WMMA/GMMA PE 比仍 T0\|tensor_core_descriptor_overhead.*仍是 T0\|包含 T0/T1 参数\|ranking drivers at T0/T1\|requires every ranking-driving parameter to be T2+" \
    README.md docs/model-trust-and-release.md docs/NPU_Engines_Architecture_Guide.md scripts/release_gate.py \
    > .omo/evidence/final-lift-fail-f3-stale-grep.txt 2>&1
  test $? -eq 1  # grep should find NOTHING (exit 1 = no matches)

  # Verify parameters.yaml contains zero T0 references after Todo 1 (header + entries)
  grep -i "T0" references/calibration/parameters.yaml > .omo/evidence/final-lift-fail-f3-header-grep.txt 2>&1
  test $? -eq 1  # grep should find NOTHING (exit 1 = no matches)

  # Verify docs/model-trust-and-release.md:58 no longer uses "T0/T1" phrasing
  # (rephrase to "T1" in Todo 5 once all T0 parameters are gone)
  sed -n '58p' docs/model-trust-and-release.md | grep -n "T0/T1" > .omo/evidence/final-lift-fail-f3-line58-grep.txt 2>&1
  test $? -eq 1  # grep should find NOTHING (exit 1 = no matches)

  # Verify zero T0 entries remain in calibration registry
  PYTHONPATH=sim uv run python -c "
  from calibration.registry import CalibrationRegistry
  from contracts.hardware import TrustLevel
  r = CalibrationRegistry.from_yaml()
  t0 = [e.calibration_id for e in r.entries() if e.trust_level == TrustLevel.T0]
  assert not t0, f'T0 entries still exist: {t0}'
  print(f'PASS: all {len(r.entries())} entries are T1 or higher, zero T0 remaining')
  " > .omo/evidence/final-lift-fail-f3-zero-t0.json 2>&1
  test $? -eq 0

  # Verify new T1 entries are present in parameters.yaml
  python3 -c "
  import yaml
  with open('references/calibration/parameters.yaml') as f:
      entries = {e['calibration_id']: e for e in yaml.safe_load(f)['parameters']}
  for id in ['tensor_core_descriptor_overhead','max_freq_12nm','dram_efficiency']:
      e = entries[id]
      assert e['trust_level'] == 'T1', f'{id}: {e[\"trust_level\"]}'
      assert e.get('source_uri'), f'{id}: source_uri missing'
  print('PASS: all T1 entries verified')
  " > .omo/evidence/final-lift-fail-f3-manual-qa.json 2>&1
  test $? -eq 0
  ```

  **APPROVE iff** no stale FAIL clause found、parameters.yaml header 不再声称有 T0 条目、model-trust-and-release.md:58 不再使用 "T0/T1" 措辞、zero T0 entries remain in registry、all new entries T1 with non-empty source_uri。

- [x] F4. Scope and evidence fidelity

  ```bash
  uv run python scripts/verify_scope.py \
    --plan .omo/plans/lift-decision-grade-fail.md \
    --baseline-commit "$(git merge-base HEAD origin/main)" \
    --publication-manifest docs/publication-manifest.yaml \
    --output .omo/evidence/final-lift-fail-f4-scope-fidelity.json
  ```

  **APPROVE iff** exit 0、`forbidden_dependencies=[]`、`ultraresearch_changes=[]`、`historical_report_changes=[]`、`unbound_current_claims=[]`。

## Commit strategy

- 使用 conventional commits；每个 todo 的 implementation+test 为一个原子 commit
- Wave 1 的 Todo 1, 2, 3 可并行完成（独立添加条目），Todo 4 依赖 2+3
- Wave 2 的 Todo 5 → 6 → 7 严格顺序（文档内容逐级引用）
- 只在全部 F1-F4 通过后才标记计划完成

## Success criteria

- `references/calibration/parameters.yaml` 中不再有 `trust_level: T0` 的条目
- 新增 7 个校准条目（4 频率 + 3 DRAM），全部 T1，source_uri 非空
- `tensor_core_descriptor_overhead` 从 T0 升级到 T1
- `sim/calibration/evaluate.py` exit 0，所有新条目可提取
- `README.md` §1.4 不再包含 FAIL 子句
- `docs/model-trust-and-release.md` Decision-Grade State 反映最新校准状态（含 T2→T3 修正）
- F1-F4 全部以 exit 0 和 verdict 通过完成
- `README.md:67` 免责声明已同步更新（"包含 T0/T1" → "包含 T1"）
- 零 T0 残留：`parameters.yaml` 中所有条目 trust_level ≥ T1
