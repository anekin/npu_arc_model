# cross-node-all-engines-dse - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** 四工艺节点 × 八种引擎 × 两种带宽场景的完整排名矩阵，搞清楚在 28nm 与 7nm、LPDDR5 与片上 3D DRAM 之间到底哪个引擎胜出——不再只有 block 和 fsa 的"瞎子摸象"结论。

**Why this approach:** ci-all-axes 已有 process_node 轴和频率绑定，再加 28 个 engine×node 交叉乘积（每场景不到十分钟）即可覆盖全部引擎，不必重跑全量笛卡尔积。

**What it will NOT do:** 不修改任何引擎的性能公式——只扫不修。不加新场景，不加新引擎。不把跨节点"探索性结论"升为"决策级"。

**Effort:** Short — 6 todos, agent-time ~1-2 hours
**Risk:** Low — space.py 改 ~25 行；DSE 重跑用现有自动化命令，无交互依赖
**Decisions to sanity-check:** `ci-all-axes` 的 engine×node 交叉是追加而非替换（不破坏 7nm 的现有全引擎行为）；独立脚本从 block+fsa 扩展到全部八引擎时需确认频率约束一致性

Your next move: approve 本计划即可开始执行；与 Plan A (WMMA/GMMA) 互不依赖，可同时启动。Full execution detail follows below.

---

> TL;DR (machine): Short, Low. Min-code (~25 LOC space.py) → 2-scenario × 4-node × 8-engine DSE → full ranking matrix + standalone script + docs. No engine formula changes.

## Scope

### Must have

- 修改 `sim/dse/space.py` 的 `_ci_all_axes_combinations()` 方法，在 process_node 轴迭代时对每个非默认节点生成全部 8 个引擎的设计点（而非仅 block 默认引擎）。
- 对 `lpddr5_3b` 和 `onchip_7b` 两个场景分别运行全引擎跨节点 DSE（`--space ci-all-axes` 模式）。
- 生成 `scenario × node × engine` 的完整排名矩阵（Markdown 表格 + JSON），覆盖 2 scenarios × 4 nodes × 8 engines = 64 个 cell。
- 将 `.omo/evidence/investigate-fsa-cross-node-freq.py` 独立脚本泛化为支持全部 8 引擎的快速跨节点对比。
- 更新 `docs/model-trust-and-release.md` 和 `README.md` 以反映完整跨节点引擎排名的发现。

### Must NOT have (guardrails, anti-slop, scope boundaries)

- 不新增 DSE 空间模式（`ci-all-axes` / `full` / `quick` 三种保持不变）——仅在 `ci-all-axes` 内部增加 engine×node 交叉。
- 不修改任何引擎的性能公式——本计划只做扫描和收集，不改计算模型。
- 不新增引擎类型，不删除现有引擎。
- 不修改历史 dated report。
- 不对全部 5 个场景 × 4 个 node 做完整 DSE——仅对 `lpddr5_3b` 和 `onchip_7b` 做跨 node（其他场景固定 12nm，与 P0 计划一致）。
- 不改变 `process_node` 轴的定义和频率约束——这些已在 P0 计划中完成。
- 不修改 `dse_axes.yaml` 的 axes values。
- 不将跨节点结论从 exploratory 升级为 decision-grade（多节点对比仍标记为探索性，除非 Plan A 同时完成）。
- 不破坏 `decision-grade` profile 的 fail-closed 行为。
- 不将 `input_stationary` 引擎重新命名为 `is_systolic`（文件名保留但注册 ID 不变）。
- **不修改** `process_node` 轴的默认值（保持 7nm）和该轴的默认引擎（保持 block）——两者是交叉乘积逻辑的前提条件。
- **不修改** `sim/config/scenarios.yaml`、`sim/engine/registry.py`、或任何引擎 Python 实现。
- **不削弱**现有测试的断言（`test_engine_physical_invariants.py`、`test_engine_result_contract.py` 等）来使新行为变绿。

## Verification strategy
> Zero human intervention - all verification is agent-executed.

- Test decision: **tests-after** — 代码变更（Todo 1）先写测试验证交叉乘积逻辑，再实现。DSE 运行（Todo 2-3）无新测试，但验证 exit 0 和完整性。Todo 4-6 为纯运行+文档类型。
- Engine formula invariants: 不修改任何 engine 的性能公式。所有 `test_engine_physical_invariants.py` 的断言保持通过。
- Scenario coverage: 每个 DSE 必须产生至少 1 个 non-trivial design point（非零 tok/s 且面积 ≤ 约束上限）。
- Evidence: `.omo/evidence/task-<N>-cross-node-all-engines-dse.<ext>`；每条证据同时记录命令、exit code、git commit、config/hardware/lock digests。
- QA policy: 每个 todo 都有 happy path + failure/negative path；不得用 grep 命中、worker 自述或历史 JSON 代替实际执行。

## Execution strategy

### Parallel execution waves

> Target 5-8 todos per wave. Wave 2 is the final wave (3 todos, minimum allowed).

```
Wave 1 — Infrastructure + DSE Run (3 todos, sequential dependency chain)
├── Todo 1: Add engine × process_node cross-product to ci-all-axes
├── Todo 2: Run cross-node DSE for 8 engines on lpddr5_3b
└── Todo 3: Run cross-node DSE for 8 engines on onchip_7b

Wave 2 — Analysis + Documentation (3 todos, two parallel lanes)
├── Lane A: Analysis
│   ├── Todo 4: Generate complete 8×4×2 ranking matrix
│   └── Todo 5: Generalize cross-node standalone script to all engines
└── Lane B: Documentation
    └── Todo 6: Update decision-grade documentation
```

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|---|---|---|---|
| 1 | — | 2, 3 | — |
| 2 | 1 | 4, 5 | — |
| 3 | 1 | 4, 5 | — |
| 4 | 2, 3 | 6 | 5 |
| 5 | 2, 3 | — | 4 |
| 6 | 4 | F1-F4 | — |

Critical path: `1 → 2 → 4 → 6 → F1-F4`

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. 在 ci-all-axes 模式中添加 engine × process_node 交叉乘积

  **What to do**:
   1. 修改 `sim/dse/space.py:143-187` 的 `_ci_all_axes_combinations()` 方法。在现有的 `process_node` 轴迭代循环之后（或取代其内部逻辑），对于 `process_node` 轴的每个值（28, 22, 12, 7），跳过默认节点（7nm，因为 engine 轴自身迭代已覆盖 7nm），对非默认节点生成全部 8 个引擎的组合。
   2. 交叉乘积逻辑：当前 `ci-all-axes` 中 `process_node` 轴的非默认值（28/22/12nm）仅生成 1 个设计点（默认引擎=block）。修改后，对于每个 `node_val in [28, 22, 12]`，对每个 `engine_val in engine_full_ids()` 生成一个组合——共增加 **21 个**设计点（7 个非 block 引擎 × 3 个节点；block 在 28/22/12nm 原有 3 个点由原来的 process_node 轴生成，不重复计算）。
   3. **引擎 ID 注意**：使用 `registry.engine_full_ids()`（返回 8 个 ID：`systolic, os_systolic, block, tensor_core, wmma, gmma, input_stationary, fsa`）。注意 `"is_systolic"` 不在注册表中——对应的 ID 是 `"input_stationary"`。
   4. 新增 `sim/tests/test_dse_cross_node_coverage.py`，验证 `ci-all-axes` 模式生成的 design points 包含全部 4 个节点下的全部 8 个引擎（增加 21 个组合：7 个非 block 引擎 × 3 个节点；block 在 28/22/12nm 已在原有 process_node 轴中覆盖）。

  **Must NOT do**:
   - 不添加新的空间模式——只修改 `_ci_all_axes_combinations()`。
   - 不修改 `generate_configs()` 的旧版路径。
   - 不把 `is_systolic` 当作有效引擎 ID——始终使用 `input_stationary`。

  **Parallelization**: Wave 1 | Blocked by: — | Blocks: 2, 3

  **References**:
   - `sim/dse/space.py:143-187` — `_ci_all_axes_combinations()` 的当前实现，只对非 process_node 轴使用默认引擎。
   - `sim/dse/space.py:53-255` — `DesignSpace.__init__()` 和 `generate_with_exclusions()`。
   - `sim/engine/registry.py:20-28` — `engine_full_ids()` 返回全部 8 个引擎 ID。
   - `sim/engine/registry.py:46-49` — `lookup_by_prefix()` 和 `is_valid_engine()` 用于验证。
   - `sim/config/dse_axes.yaml:13-22` — `engine` 轴的 values 定义。
   - `sim/config/dse_axes.yaml:89-92` — `process_node` 轴的值定义。

  **Acceptance criteria**:
   - `ci-all-axes` 模式下 `process_node=28` 生成 8 个 design points（每种引擎一个），而非原来的 1 个（仅 block）。
   - `process_node=22` 同样生成 8 个 points。
   - `process_node=12` 同样生成 8 个 points。
   - `process_node=7` 的行为不变（8 个引擎，engine 轴自身迭代生成）。
   - 约束正确应用——例如 28nm 下的频率约束仍然排除不合法的频率值。
   - `uv run pytest sim/tests/test_dse_cross_node_coverage.py -q` 通过。

  **QA scenarios**:
   - **Happy**: 验证交叉乘积逻辑 + 约束过滤；Evidence `.omo/evidence/task-1-cross-node-all-engines-dse-coverage.json`。
   - **Failure**: 注入非法 node 值（如 5nm）→ 约束拒绝；Evidence `.omo/evidence/task-1-cross-node-all-engines-dse-coverage-negative.txt`。
   - **Commands**:
     ```bash
     uv run pytest sim/tests/test_dse_cross_node_coverage.py -q > .omo/evidence/task-1-cross-node-all-engines-dse-coverage.json 2>&1
     test $? -eq 0
     uv run pytest sim/tests/test_dse_cross_node_coverage.py -q -k "invalid" > .omo/evidence/task-1-cross-node-all-engines-dse-coverage-negative.txt 2>&1
     test $? -eq 0
     ```

  **Commit**: YES | `feat(dse): add engine × process_node cross-product in ci-all-axes mode` | `sim/dse/space.py`, `sim/tests/test_dse_cross_node_coverage.py`

- [x] 2. 在 lpddr5_3b 场景上运行全引擎跨节点 DSE

  **What to do**:
   1. 使用 `--space ci-all-axes` 模式对 `lpddr5_3b` 场景运行 DSE（Todo 1 完成后自动包含 engine×node 交叉乘积）。
   2. 验证所有 4 个节点下的全部 8 种引擎都产生了设计点（部分引擎可能被约束过滤——记录过滤原因）。
   3. 确认 exit 0，无 crash，Pareto frontier 非空。
   4. 输出文件: `.omo/evidence/task-2-cross-node-all-engines-dse-lpddr5.json`。

  **Must NOT do**:
   - 不使用 `--space full` 模式（组合爆炸）。
   - 不修改 DSE 结果——只运行和收集。

  **Parallelization**: Wave 1 | Blocked by: 1 | Blocks: 4, 5

  **References**:
   - `sim/design_space_explorer.py:708-714` — scenario 路由逻辑。
   - `sim/dse/runner.py:199-226` — `_evaluate_ppa()` process_node 传播。
   - `sim/dse/space.py:143-187` — 修改后的 `_ci_all_axes_combinations()`。

  **Acceptance criteria**:
   - 命令 exit 0。
   - 每个 node 下至少有 3 种引擎产生有效 design points（block 至少 + 2 个额外引擎）。
   - **仅作观测（不作为 pass/fail 条件）**：28nm 和 22nm 下的引擎排名与 7nm 的差异程度。排名一致的节点记录为发现（而非失败）。
   - 无引擎在运行中 crash。
   - 输出 JSON 格式有效且可被 `python3 -c "import json; json.load(...)"` 解析。

  **QA scenarios**:
   - **Happy**: DSE exit 0，每节点 ≥3 引擎有效；Evidence `.omo/evidence/task-2-cross-node-all-engines-dse-lpddr5.json`。
   - **Failure**: 任何 crash 或零有效引擎节点 → fail；Evidence `.omo/evidence/task-2-cross-node-all-engines-dse-lpddr5-negative.txt`。
   - **Commands**:
     ```bash
     uv run python sim/design_space_explorer.py --scenario lpddr5_3b --space ci-all-axes --output .omo/evidence/task-2-cross-node-all-engines-dse-lpddr5.json
     test $? -eq 0
     python3 -c "import json; d=json.load(open('.omo/evidence/task-2-cross-node-all-engines-dse-lpddr5.json')); assert len(d) > 0"
     ```

  **Commit**: NO（evidence-only，通过 evidence 文件记录；无需代码提交）

- [x] 3. 在 onchip_7b 场景上运行全引擎跨节点 DSE

  **What to do**:
   1. 与 Todo 2 相同流程，但针对 `onchip_7b` 场景（高带宽，500 GB/s）。
   2. 输出文件: `.omo/evidence/task-3-cross-node-all-engines-dse-onchip.json`。

  **Must NOT do**:
   - 不运行全部 5 个场景——仅 lpddr5_3b 和 onchip_7b。

  **Parallelization**: Wave 1 | Blocked by: 1 | Blocks: 4, 5

  **References**:
   - 同 Todo 2。

  **Acceptance criteria**:
   - 同 Todo 2，但针对 `onchip_7b` 场景。
   - 高带宽（500 GB/s）下引擎排名方向与低带宽（51.2 GB/s）的差异程度**仅作观测**（记录为发现，不作为 pass/fail 条件）。

  **QA scenarios**:
   - **Happy**: DSE exit 0，高 BW 场景排名与低 BW 不同；Evidence `.omo/evidence/task-3-cross-node-all-engines-dse-onchip.json`。
   - **Failure**: 所有引擎在高 BW 下被约束排空 → fail；Evidence `.omo/evidence/task-3-cross-node-all-engines-dse-onchip-negative.txt`。

  **Commit**: NO（evidence-only）

- [x] 4. 生成完整的 8×4×2 排名矩阵

  **What to do**:
   1. 从 Todo 2 和 Todo 3 的 JSON 输出中提取每个 `(node, engine)` 组合的最佳 tok/s 和面积。
   2. 生成 Markdown 表格：行 = engine（8 行，按 tok/s 排序），列 = node × scenario（8 列：lpddr5_28nm, lpddr5_22nm, ... onchip_7nm）。
   3. 每个 cell 包含 tok/s（主要指标）和 area_mm²（次要指标）。缺失引擎（因约束被完全过滤）标记为"constraint-filtered"，未运行的引擎标记为"—"。
   4. **仅记录观测结果，不预设强制行为假设。**以下预期行为作为观测项（记录实际表现，不将其作为 pass/fail 判断）：
      - block 在各节点保持 BW-bound 的倾向性（lpddr5 下 tok/s 稳定）。
      - FSA compute-bound 在低带宽老旧节点的表现趋势。
      - GMMA/block 在高带宽 onchip 场景下的相对排名。
   5. 输出文件: `.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md` + `.json`。

  **Must NOT do**:
   - 不对每个 cell 手动计算 tok/s——从 JSON 提取自动完成。
   - 不修改 DSE 输出数据。

  **Parallelization**: Wave 2, Lane A | Blocked by: 2, 3 | Blocks: 6 | Parallel with: 5

  **References**:
   - `.omo/evidence/task-2-cross-node-all-engines-dse-lpddr5.json`
   - `.omo/evidence/task-3-cross-node-all-engines-dse-onchip.json`
   - `.omo/evidence/task-14-engine-selection-p0-cross-node-dse.md` — 旧的仅 block 排名矩阵（作为格式参考）。

  **Acceptance criteria**:
   - 矩阵中 64 个 cell 均被填充——有数据的填充 tok/s+area；因约束被完全过滤的引擎标注 "constraint-filtered"（附约束名称）；未运行的引擎标注 "—"。
   - block 在 lpddr5 下各节点 tok/s 差异 <5%（**观测项**：如果不满足则记录为发现，不判定为 plan 失败）。
   - Markdown 表格在 GitHub/GitLab 上可渲染（手工检查，不做自动化渲染验证）。
   - JSON 输出可被 `python3 -c "import json; json.load(...)"` 解析。

  **QA scenarios**:
   - **Happy**: 矩阵 64 cell 全部填充（含 constraint-filtered 标注）；每个 scenario×node 组合至少有 3 种引擎产生有效 design points；Evidence `.omo/evidence/task-4-cross-node-all-engines-dse-matrix.json`。
   - **Failure**: 某个 scenario×node 组合下全部引擎被排空（0 个有效 point）→ fail；Evidence `.omo/evidence/task-4-cross-node-all-engines-dse-matrix-negative.txt`。

  **Commit**: YES | `evidence(dse): complete 8-engine × 4-node ranking matrix for lpddr5_3b and onchip_7b` | `.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md`, `.json`

- [x] 5. 泛化跨节点独立脚本到全部引擎

  **What to do**:
   1. 修改 `.omo/evidence/investigate-fsa-cross-node-freq.py`，将 `ENGINES = ("block", "fsa")` 改为 `ENGINES = engine_full_ids()`（全部 8 种引擎），并重命名为 `investigate-all-engines-cross-node-freq.py`。
   2. 更新脚本以使用 `registry.create_engine()` 工厂函数（而非硬编码的引擎类导入），确保与 registry 完全一致。
   3. 验证所有引擎在所有 4 个节点下正确运行（使用每节点频率约束）。
   4. 输出 JSON 和 Markdown 到 `.omo/evidence/investigate-all-engines-cross-node-freq.*`。

  **Must NOT do**:
   - 不删除原有的 `investigate-fsa-cross-node*.py` 文件——保留作为历史证据。
   - 不修改引擎模型——仅读取和对比。

  **Parallelization**: Wave 2, Lane A | Blocked by: 2, 3 | Blocks: — | Parallel with: 4

  **References**:
   - `.omo/evidence/investigate-fsa-cross-node-freq.py` — 当前仅 block + fsa 的脚本。
   - `sim/engine/registry.py:20-28` — `engine_full_ids()`。
   - `sim/config/dse_axes.yaml` — 频率约束定义（`node_28_frequency_bound` 等）。

  **Acceptance criteria**:
   - 脚本 exit 0，全部 8 种引擎在全部 4 个节点下成功运行。
   - 每节点频率约束正确应用（与 dse_axes.yaml 一致）。
   - 输出 JSON 包含 engine_type、node、频率、tok_s、area_mm² 字段。

  **QA scenarios**:
   - **Happy**: 全部 32 个 (engine, node) 组合成功运行；Evidence `.omo/evidence/investigate-all-engines-cross-node-freq.json`。
   - **Failure**: 任何 engine 在运行时 crash → fail；Evidence `.omo/evidence/investigate-all-engines-cross-node-freq-negative.txt`。

  **Commit**: YES | `evidence(dse): generalize cross-node standalone to all 8 engines` | `.omo/evidence/investigate-all-engines-cross-node-freq.py`, `.json`, `.md`

- [x] 6. 更新决策级文档

  **What to do**:
   1. 更新 `docs/model-trust-and-release.md` 的 "跨节点引擎选择发现" section：替换仅 block+FSA 的表为完整 8 引擎排名矩阵的摘要（highlight 最关键的排名变化——例如 os_systolic 在 28nm lpddr5 的表现，GMMA 在 onchip_7b 的优势等）。
   2. 更新 `README.md` 的 "跨节点验证结论" 行（§1.4 双场景技术路线表格最后一列）：将原来的 "block 和 FSA 仅对比" 改为全引擎排名摘要，引用 Todo 4 的证据链接。
   3. 保留 FAIL 状态标注（WMMA/GMMA PE 比仍 T0——除非 Plan A 同时完成并升级了 trust level）。
   4. 在 `README.md` §7 "关键洞见" 中新增第 6 条跨节点引擎排名的关键发现。

  **Must NOT do**:
   - 不修改历史 dated report。
   - 不将跨节点结论从 exploratory 升级为 decision-grade。
   - 不删除或削弱 Must NOT have 中的限制表述。
   - 不引用 `is_systolic` 引擎名——使用 `input_stationary`。

  **Parallelization**: Wave 2, Lane B | Blocked by: 4 | Blocks: F1-F4

  **References**:
   - `docs/model-trust-and-release.md` — "跨节点引擎选择发现" section。
   - `README.md` — "跨节点验证结论" 行（§1.4 表格最后一列）。
   - `README.md` — "关键洞见" section（§7）。
   - `.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md` — Todo 4 产出。

  **Acceptance criteria**:
   - `docs/model-trust-and-release.md` 的跨节点发现表包含全部 8 种引擎。
   - README 跨节点结论行更新为引用 Todo 4 产出的全引擎排名。
   - 无对 `is_systolic` 的引用（始终使用 `input_stationary`）。
   - `uv run ruff check .` 通过。

  **QA scenarios**:
   - **Happy**: 文档一致 + 无过期 claim；Evidence `.omo/evidence/task-6-cross-node-all-engines-dse-docs.json`。
   - **Failure**: 与 DSE 证据不一致的 claim → fail；Evidence `.omo/evidence/task-6-cross-node-all-engines-dse-docs-negative.txt`。

  **Commit**: YES | `docs(dse): update cross-node conclusions with full 8-engine ranking` | `docs/model-trust-and-release.md`, `README.md`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [x] F1. Plan compliance audit — verify every Todo 1–6 acceptance criterion against actual files/evidence/commits.

  ```bash
  uv run python scripts/verify_evidence_ledger.py \
    --plan .omo/plans/cross-node-all-engines-dse.md \
    --evidence-root .omo/evidence \
    --output .omo/evidence/final-cross-node-f1-plan-compliance.json
  ```
  **APPROVE iff** exit 0、Todos/F1-F4 schema 可解析、Todo 1–6 每项有匹配 commit/evidence。

- [x] F2. Code quality review — ruff + basedpyright + pytest on changed modules.

  ```bash
  uv run ruff format --check . > .omo/evidence/final-cross-node-f2-code-quality.txt 2>&1
  uv run ruff check . >> .omo/evidence/final-cross-node-f2-code-quality.txt 2>&1
  uv run basedpyright >> .omo/evidence/final-cross-node-f2-code-quality.txt 2>&1
  uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_result_contract.py \
    sim/tests/test_dse_cross_node_coverage.py -q >> .omo/evidence/final-cross-node-f2-code-quality.txt 2>&1
  ```
  **APPROVE iff** 全部命令 exit 0、blocking skip/xfail=0。

- [x] F3. Real CLI/scenario/replay QA — release gate with all workloads.

  ```bash
  uv run python scripts/release_gate.py \
    --profile experimental \
    --clean-checkout \
    --exercise-legacy \
    --exercise-all-workloads \
    --space ci-all-axes \
    --output .omo/evidence/final-cross-node-f3-manual-qa.json
  ```
  **APPROVE iff** exit 0、`legacy_failures=[]`、`workload_failures=[]`、`errors=0`。

- [x] F4. Scope fidelity — verify no unauthorized changes beyond plan boundary.

  ```bash
  uv run python scripts/verify_scope.py \
    --plan .omo/plans/cross-node-all-engines-dse.md \
    --baseline-commit "$(git merge-base HEAD origin/main)" \
    --publication-manifest docs/publication-manifest.yaml \
    --output .omo/evidence/final-cross-node-f4-scope-fidelity.json
  ```
  **APPROVE iff** exit 0、`forbidden_dependencies=[]`、`historical_report_changes=[]`、`unbound_current_claims=[]`。

## Commit strategy

- 使用 conventional commits；每个 todo 的 implementation+test 为一个原子 commit。
- 不 amend、不 squash 已发布历史。
- 每个 commit 只包含该 todo 的 Files 列表与其证据。
- Evidence 文件在 Todo 2-3（DSE 运行）不产生 code commit，仅记录 evidence。
- 只在全部 F1-F4 通过后才标记计划完成。

## Success criteria

- `_ci_all_axes_combinations()` 对非 7nm 节点生成 8 个引擎组合（而非仅 block）。
- `ci-all-axes` 模式下的组合数增加 21 个（7 个非 block 引擎 × 3 个非默认节点；block 在 28/22/12nm 已有 process_node 轴覆盖，不重复计算）。
- 2 个场景 × 4 个节点 DSE exit 0，每节点至少 3 种引擎产生有效 points。
- 完整 8×4×2 排名矩阵中 64 个 cell 全部填充（含 constraint-filtered 标注和未运行标注）。
- 排名矩阵至少包含 4 个引擎类型在所有节点/场景组合下产生有效数据（每节点每场景 ≥3 种引擎有效）。
- 独立跨节点脚本支持全部 8 种引擎。
- 文档更新与 DSE 证据一致，无对新引擎 ID 的错误引用。
- F1、F2、F3、F4 全部以 `verdict=PASS` 和 exit 0 完成。
