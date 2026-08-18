# arc-prefill-ttft-dse - Work Plan

## TL;DR (For humans)

**What you'll get:** Arc Model DSE 真实建模 prefill：`--batch-m` 从 [1,2] 放宽为任意 ≥1 整数，新增 `simulate_prefill()` 与 `ttft_ms_from_prefill()`，所有 DSE 结果（v2 + legacy 投影）携带 `ttft_ms` 字段，quick 模式补上 Block 64×64。最终产出 Gate 1b 闭环所需的两组 DSE TTFT 目标值（M=128 / M=2000）+ Func Model 比值判定 + 已入库证据，交回 CaduceusCore。

**Why this approach:** 需求文档（docs/arc-model-prefill-ttft-requirements.md）已把 FR-1..7/NFR-1..6/AC-1..5 定义完整；规划前实测（当前引擎公式 + M=128/M=2000 trace）比值已落在 PASS 窗口内（1.22× / 1.27× @WC），证明不改任何引擎公式即可闭环——只做 trace 参数化、TTFT 换算与结果契约穿线。

**What it will NOT do:** 不改任何引擎 GEMM 公式。不建 KV prefill 写回模型（写=0 为已知简化，文档声明）。不做 CV 模式 TTFT。不碰 CaduceusCore 仓库（§8 A1-A3 另行安排）。不为了过比值窗口而调模型参数——超窗则如实记录 FAIL 诊断。

**Effort:** Medium — 10 todos, agent-time ~2-4 hours
**Risk:** Low-Medium — 主要风险是 legacy golden 契约重生成（有意打破精确键集测试）；比值窗口实测有 1.2-1.3× 余量，超窗风险低
**Decisions to sanity-check:** Gate 1b 目标取 weight_cache=True 为主（同时记录 False 行）；`evaluate_config` 增加 keyword-only `batch_m`，decode tok_s 永远 M=1、TTFT 按 batch_m 计算（层数取自 model spec）；`ttft_ms` 进入 v2 EngineMetrics + legacy 投影每个非 CV 条目（FR-5 明文要求，有意更新 golden）；scenario 路径（dse/runner）同规则计算 ttft_ms（spec 层数，非 `_NUM_LAYERS=28`）

Your next move: approve 本计划即可开始执行（`/start-work`）。Full execution detail follows below.

---

> TL;DR (machine): Medium, Low-Medium. Trace batch_m 参数化 + 去 KV 全局副作用 → simulate_prefill + ttft_ms_from_prefill(units) → PPA/EngineMetrics/legacy 穿线 + evaluate_config 双指标 → CLI --batch-m ≥1 + quick (64,64) → 验收不变量套件 → M=128/2000 证据 + Gate 1b 目标抽取 → 文档 + 回流产物。10 todos, 4 waves + F1-F4。

## Scope

### Must have

- 消除模块级固定 M=1 trace（`_LLM_TRACE` 全局）与 `generate_trace_from_spec` 对 `_KV_HEADS/_HEAD_DIM` 的全局副作用；KV 几何改为参数传递（FR-1, NFR-5）。
- `simulate_prefill(config, batch_m, model_alias)` 返回 per-layer prefill cycles（KV 写回=0，代码注释 + 文档声明）（FR-2）。
- `ttft_ms_from_prefill(prefill_cycles, num_layers, freq_mhz)`，换算强制走 `sim/contracts/units.py`（FR-3）。
- CLI `--batch-m` 放宽为 `type=int` + `>=1` 校验；仅驱动 prefill/TTFT 指标，decode `tok_s` 保持 batch_m=1 语义（help 文案说明）（FR-4）。
- `ttft_ms` 字段进入结果契约：`PPA` → `EngineMetrics`(v2, extra="forbid") → legacy 投影每个非 CV 条目；CV 模式 0.0（FR-5）。
- quick dims 增加 `(64, 64)`，保证 Block 64×64 可通过 `--quick` 直接获取（FR-6）。
- 默认（无 `--batch-m`）运行 tok_s/area/power 与实施前逐配置一致；现有测试全绿（FR-7, AC-2）。
- TDD + mutation（NFR-4）；证据 JSON 提交 git（NFR-2）；文档 stale 清除 + README DSE 能力矩阵（NFR-3）；§7 回流产物（AC-5）。

### Must NOT have (guardrails, anti-slop, scope boundaries)

- 不修改任何引擎 GEMM 公式（block_engine/systolic/os_systolic/… 一律不动）——§9 明确排除。
- 不建 KV prefill 写回模型（写=0 为已知简化，未来随 arc-model-v3 规划处理）。
- 不实现 CV 模式 TTFT（保持 0.0 约定）。
- 不修改 `evaluate_config` 的三个位置参数顺序/名称（`test_dse_strict.py` monkeypatch 依赖）；新参数为 keyword-only。
- 不修改 `_NUM_LAYERS = 28` 模块默认值（FR-7 向后兼容；CLI main() 已按 spec 设置 36，证据运行正确）。
- 不修改历史 dated report、`verify_evidence_ledger.py`/`release_gate.py`/`verify_scope.py`/`verify_model_integrity.py` 等验证基础设施。
- 不为了落入 [0.5×, 2.0×] 窗口而调引擎/校准参数——超窗时如实记录 FAIL + 诊断（BW vs compute 分解）并汇报，禁止数值凑数。
- 不触及 CaduceusCore 仓库（§8 A1-A3 不属于本计划）。
- 不新增依赖；Python 版本契约不变。

## Verification strategy
> Zero human intervention - all verification is agent-executed.

- Test decision: **TDD with pytest**。每个 todo：先 RED 测试 → 最小实现 → 全量回归 + mutation 验证（NFR-4）。
- FR-7 基线锁：T1 先把当前默认运行输出冻结为 golden baseline，后续每个 todo 必须保持其绿色。
- Golden 契约：`sim/tests/golden/legacy_cli_contract.json` 为 CLI 输出契约描述——新增 `ttft_ms` 字段属有意契约变更，T4/T5 各自同步对应段落（item_schema 段 vs flags 段）。
- 证据：`.omo/evidence/task-<N>-arc-prefill-ttft-dse.<ext>`；每条证据记录命令、exit code、git commit、sha256。
- QA policy: 每个 todo 有 happy + failure/negative path；不得用 grep 命中、worker 自述或历史 JSON 代替实际执行。

## Execution strategy

### Parallel execution waves

> 依赖链 T2→T3→T4 为核心串行通路；T7/T8 为独立证据双车道。

```
Wave 0 — Prep（顺序）：
├── T0: 提交 lift-decision-grade-fail 遗留改动 + 需求文档入库
└── T1: FR-7 基线锁定（默认运行输出 golden baseline + 回归测试）

Wave 1 — 核心链路（串行 lane）：
├── T2: trace 参数化 + 消除 KV 全局副作用（含 dse/runner.py 适配）
├── T3: simulate_prefill + ttft_ms_from_prefill（TDD）
└── T4: ttft_ms 结果契约穿线（PPA/EngineMetrics/legacy + evaluate_config 双指标 + golden/pinned 同步）

Wave 2 — CLI 与验收：
├── T5: --batch-m 放宽 + quick (64,64) + FR-4 decode 语义
└── T6: 交叉验收不变量套件（线性/BW floor/单调/CV/mutation）

Wave 3 — 证据与文档：
├── T7: M=128 证据 + Gate 1b 目标抽取  ∥  T8: M=2000 证据 + Gate 1b 目标抽取
└── T9: 文档同步（arc_vs_func + README 能力矩阵）+ §7 回流产物

Final Wave: F1 ∥ F2 ∥ F3 ∥ F4
```

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|---|---|---|---|
| 0 | — | 1 | — |
| 1 | 0 | 2,3,4,5,6,7,8 | — |
| 2 | 1 | 3 | — |
| 3 | 2 | 4 | — |
| 4 | 3 | 5 | — |
| 5 | 4 | 6 | — |
| 6 | 5 | 7,8 | — |
| 7 | 6 | 9 | 8 |
| 8 | 6 | 9 | 7 |
| 9 | 7,8 | F1-F4 | — |

Critical path: `0 → 1 → 2 → 3 → 4 → 5 → 6 → (7∥8) → 9 → F1-F4`

## Todos
> Implementation + Test = ONE todo. Never separate.

- [x] 0. 提交 lift-decision-grade-fail 遗留改动 + 需求文档入库

  **What to do**:
   1. `git status --porcelain` 列出全部未提交改动（当前已知：README.md、docs/NPU_Engines_Architecture_Guide.md、docs/model-trust-and-release.md、pyproject.toml、references/calibration/parameters.yaml、scripts/release_gate.py、sim/calibration/evaluate.py、sim/tests/test_calibration_evaluate.py、sim/tests/test_calibration_registry.py + .omo/ 下该计划的 plan/evidence/notepad 文件；以实际 status 为准）。
   2. 按 `lift-decision-grade-fail` 计划约定的 conventional commit 分组合并提交（该计划已完成 F1-F4 全绿，内容已验收；commit message 使用该计划各 todo 的 Commit 行）。**逐组 stage、逐组提交，每组提交前核对 `git diff --cached --stat`**（防连带提交无关文件）。
   3. `git add docs/arc-model-prefill-ttft-requirements.md && git commit`，message: `docs(arc): add Func Model gate-1b prefill/TTFT requirement report`。
   4. 验证 `git status --porcelain` 为空。NFR-6 说明：`wmma-gmma-pe-recalibration` boulder 已于 2026-08-13 完成关闭，无文件冲突。

  **Must NOT do**:
   - 不修改任何文件内容（仅提交）。
   - 不 amend 已发布历史；不 squash 上两个计划的提交。
   - 不 push 之外做任何远程操作（push 按仓库既有惯例执行）。

  **Parallelization**: Wave 0 | Blocked by: — | Blocks: 1

  **References**:
   - `.omo/plans/lift-decision-grade-fail.md` — 各 todo 的 Commit 行（message 来源）。
   - 仓库既有 commit 历史 `git log --oneline -10`（convention 参照）。

  **Acceptance criteria**:
   - `git status --porcelain` 输出为空。
   - `git log --oneline -3` 显示本 todo 的提交。
   - 证据含 `git diff --cached --stat`（每次提交前核对）。
   - `uv run pytest -q` exit 0（提交内容不破坏测试——该计划 F1-F4 已验证）。

  **QA scenarios**:
   - **Happy**: 提交后 status 干净 + pytest 绿；Evidence `.omo/evidence/task-0-arc-prefill-ttft-dse-git-clean.txt`（内容：git status、git log -3、pytest 末尾摘要）。
   - **Failure**: 若 status 仍有文件（例如新出现的 untracked），逐项列出并停止（不得连带提交未知文件）。
   - **Commands**:
     ```bash
     git status --porcelain | tee .omo/evidence/task-0-arc-prefill-ttft-dse-git-clean.txt
     # 分组提交后：
     git add docs/arc-model-prefill-ttft-requirements.md
     git commit -m "docs(arc): add Func Model gate-1b prefill/TTFT requirement report"
     git status --porcelain && git log --oneline -3 >> .omo/evidence/task-0-arc-prefill-ttft-dse-git-clean.txt
     uv run pytest -q >> .omo/evidence/task-0-arc-prefill-ttft-dse-git-clean.txt
     ```

  **Commit**: YES | 分组提交（见 What to do）+ `docs(arc): add Func Model gate-1b prefill/TTFT requirement report` | 9 个遗留文件 + `docs/arc-model-prefill-ttft-requirements.md` + `.omo/` 遗留产物

- [x] 1. FR-7 基线锁定：默认运行输出 golden baseline + 回归测试

  **What to do**:
   1. **在改任何代码之前**运行当前 CLI：`uv run python sim/design_space_explorer.py --quick --output .omo/evidence/task-1-arc-prefill-ttft-dse-baseline.json`（当前代码，无 --batch-m）。
   2. 从该输出生成精简 golden：`sim/tests/golden/dse_default_quick_baseline.json`，每行键：`engine_type`, `H`, `W`, `w_bits`, `freq_mhz`, `weight_cache`, `dram_label`, `tok_s`, `area_mm2`, `power_w`（数值 6 位小数；**不含 ttft_ms**——基线只锁 FR-7 的三个量）。
   3. 新建 `sim/tests/test_fr7_backcompat.py`：构建 `AreaModel`/`PowerModel`（方式同 `design_space_explorer.py` main() L1011-1012），遍历 baseline 条目，对每个 (engine_type,H,W,w_bits,freq,wc,dram) 重建 config 调 `evaluate_config`，断言 `tok_s/area_mm2/power_w` 与 golden 绝对相等（tol=1e-9）。
   4. 测试**只断言交集**：不要求配置集合相等（T5 会新增 (64,64) 行，允许新增行，禁止已有行漂移）。

  **Must NOT do**:
   - 不修改任何源码——本 todo 只有测试 + golden 文件。
   - 不把 ttft_ms 写入 baseline（实施前不存在该字段）。

  **Parallelization**: Wave 0 | Blocked by: 0 | Blocks: 2

  **References**:
   - `sim/design_space_explorer.py:169-261` — `generate_configs`（quick 配置空间定义）。
   - `sim/design_space_explorer.py:1011-1012` — AreaModel/PowerModel 构造方式。
   - `sim/design_space_explorer.py:264-304` — `evaluate_config` 返回 PPA。
   - `sim/design_space_explorer.py:1208-1260` — legacy result dict 字段（tok_s/area/power 来源）。

  **Acceptance criteria**:
   - `uv run pytest sim/tests/test_fr7_backcompat.py -q` 绿（且是真实执行，非 skip）。
   - `sim/tests/golden/dse_default_quick_baseline.json` 存在且包含 ≥12 条配置。
   - 证据 `task-1-*baseline.json` 存在（原始 CLI 输出）。

  **QA scenarios**:
   - **Happy**: 新测试绿；Evidence `.omo/evidence/task-1-arc-prefill-ttft-dse-baseline.json` + `task-1-arc-prefill-ttft-dse-baseline-test.txt`（pytest 输出）。
   - **Failure**（mutation 验证）: 临时改 `simulate_layer` 加 1 cycle → 测试必须红；撤销后恢复绿。Evidence `.omo/evidence/task-1-arc-prefill-ttft-dse-baseline-negative.txt`。
   - **Commands**:
     ```bash
     uv run python sim/design_space_explorer.py --quick --output .omo/evidence/task-1-arc-prefill-ttft-dse-baseline.json
     uv run pytest sim/tests/test_fr7_backcompat.py -q > .omo/evidence/task-1-arc-prefill-ttft-dse-baseline-test.txt 2>&1
     ```

  **Commit**: YES | `test(fr7): lock default-run DSE outputs as backward-compat baseline` | `sim/tests/golden/dse_default_quick_baseline.json`, `sim/tests/test_fr7_backcompat.py`, `.omo/evidence/task-1-*`

- [x] 2. Trace 参数化 + 消除 KV 全局副作用（FR-1, NFR-5）

  **What to do**:
   1. `sim/design_space_explorer.py` L30-34：删除 `_LLM_TRACE`/`_KV_HEADS`/`_HEAD_DIM` 全局；保留 `_NUM_LAYERS: int = 28`；新增 `_DEFAULT_LLM_SPEC = "qwen2.5-3b"`。
   2. L37-56 `generate_trace_from_spec(alias, batch_m=1)`：删除 `global _KV_HEADS, _HEAD_DIM` 与 L44-45 赋值；保持返回 `list[tuple]` 不变。
   3. L67 `_compute_kv_cycles(config, batch_m=1, kv_heads: int = 0, head_dim: int = 0)`：L78 改为参数；`kv_heads/head_dim` 缺省时由调用方传入（来自 model spec）。
   4. 抽取循环体为 `_simulate_ops(config, ops, batch_m, kv_heads, head_dim) -> tuple[int, int]`，**返回 `(total_cycles, weight_bytes)` 二元组**（保持 `simulate_layer` 现有 `(layer_cycles, weight_bytes)` 返回契约不变）；`simulate_layer(config, batch_m: int | None = None, kv_heads: int | None = None, head_dim: int | None = None)`：batch_m=None→1（decode 向后兼容）；kv 几何 None→`get_spec(_DEFAULT_LLM_SPEC).kv_heads/head_dim`；直接透传 `_simulate_ops` 的二元组。
   5. L59 删除模块级 trace 生成；main() L1004-1006 改为：`_NUM_LAYERS = get_spec(model_spec).layers`（保留），删除 trace 生成语句；**L980 的 `global` 语句移除 `_LLM_TRACE`**；decode 侧由 `evaluate_config` 内部以 M=1 调用 `simulate_layer`（不再需要模块级 trace）。
   6. `sim/dse/runner.py` L219-221（现设置 `dse_module._LLM_TRACE` 与 `_NUM_LAYERS=28`）：删除 `_LLM_TRACE` 赋值，保留 `_NUM_LAYERS=28`；确认 scenario 路径输出不变（FR-7）。
   7. TDD：新建 `sim/tests/test_trace_batch_m.py`：batch_m=2 trace 的 Q/K/V/O 行 M=2；`simulate_layer(cfg, batch_m=2) > simulate_layer(cfg)`；默认调用（无 batch_m）与 T1 baseline 一致（由 test_fr7_backcompat 保证）；断言模块不再存在 `_LLM_TRACE/_KV_HEADS/_HEAD_DIM` 属性（`not hasattr(dse, "_LLM_TRACE")`）。

  **Must NOT do**:
   - 不修改任何引擎公式或 `_compute_kv_cycles` 的数值逻辑（仅参数来源变化）。
   - 不修改 `evaluate_config` 签名（3 位置参数保持不变）。
   - 不删除 `_NUM_LAYERS`（tok_s_from_layer 仍依赖它；默认 28 保持）。

  **Parallelization**: Wave 1 | Blocked by: 1 | Blocks: 3

  **References**:
   - `sim/design_space_explorer.py:30-34` — 待删全局；`:37-56` generate_trace_from_spec；`:67-108` _compute_kv_cycles；`:111-151` simulate_layer；`:59` 模块级生成；`:1004-1006` main() 设置。
   - `sim/dse/runner.py:219-221` — 引用 `_LLM_TRACE` 处。
   - `sim/model_specs.py:27` — qwen2.5-3b kv_heads/head_dim/layers。
   - 需求文档 §1.3（参考实现设计意图：`_DEFAULT_LLM_SPEC` + 按需生成 + 已知缺点回避）。

  **Acceptance criteria**:
   - `uv run pytest sim/tests/test_dse_strict.py sim/tests/test_dse_reproducibility.py sim/tests/test_frequency_bandwidth_scaling.py sim/tests/test_fr7_backcompat.py sim/tests/test_trace_batch_m.py -q` 全绿。
   - `grep -n "_LLM_TRACE\|_KV_HEADS\|_HEAD_DIM" sim/design_space_explorer.py sim/dse/runner.py` 无命中。
   - 默认 quick 运行输出与 T1 baseline 逐配置一致。

  **QA scenarios**:
   - **Happy**: 上述 pytest 集合绿 + grep 无命中；Evidence `.omo/evidence/task-2-arc-prefill-ttft-dse-refactor.json`。
   - **Failure**: mutation——把 `_simulate_ops` 中 batch_m 强制为 1 → `test_trace_batch_m.py` 红；恢复后绿。Evidence `.omo/evidence/task-2-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `refactor(dse): parameterize trace by batch_m, remove KV global side effects` | `sim/design_space_explorer.py`, `sim/dse/runner.py`, `sim/tests/test_trace_batch_m.py`

- [x] 3. 实现 simulate_prefill + ttft_ms_from_prefill（FR-2, FR-3）

  **What to do**:
   1. `sim/design_space_explorer.py` 新增：
      ```python
      def simulate_prefill(config: dict[str, Any], batch_m: int, model_alias: str = _DEFAULT_LLM_SPEC) -> int:
          """Per-layer prefill cycles for a batch of M tokens.
          KV prefill write-back cost = 0 (known simplification, see requirement §9)."""
          spec = get_spec(model_alias)
          trace = generate_trace_from_spec(model_alias, batch_m)
          return _simulate_ops(config, trace, batch_m, spec.kv_heads, spec.head_dim)[0]

      def ttft_ms_from_prefill(prefill_cycles: int, num_layers: int, freq_mhz: float) -> float:
          from contracts.units import cycles_to_microseconds
          return round(cycles_to_microseconds(prefill_cycles * num_layers, freq_mhz) / 1000.0, 2)
      ```
   2. 校验 `sim/contracts/units.py` 中 `cycles_to_microseconds` 的真实签名与语义（cycles / freq_mhz），若签名不同按实际调整调用——**禁止裸公式**。
   3. TDD：新建 `sim/tests/test_prefill_ttft.py`：
      - `test_ttft_ms_formula`: `ttft_ms_from_prefill(1_000_000, 36, 1000) == 36.0`。
      - `test_ttft_uses_units`: monkeypatch `contracts.units.cycles_to_microseconds` 记录调用，断言被调用且参数正确。
      - `test_prefill_linear_in_m`: `simulate_prefill(cfg, 200)/simulate_prefill(cfg, 20)` ≈ 10 ± 2%（compute-bound 线性）。
      - `test_prefill_gt_decode`: `simulate_prefill(cfg, 2) > simulate_layer(cfg)[0]`（同配置；simulate_layer 返回二元组，取 [0] 为 cycles）。
      - `test_kv_prefill_write_zero`: 传入 `kv_heads=spec.kv_heads, head_dim=spec.head_dim`（spec = qwen2.5-3b）时，batch_m=2 → `_compute_kv_cycles(...)` 返回 0，而 batch_m=1 同参数返回 >0（固定已知简化；显式传参防默认值 0 造成假绿）。

  **Must NOT do**:
   - 不实现 KV prefill 写回建模（写=0，docstring 已声明）。
   - 不使用裸公式 `cycles*freq/...`——必须经 `contracts.units`。
   - 不修改引擎层文件。

  **Parallelization**: Wave 1 | Blocked by: 2 | Blocks: 4

  **References**:
   - `sim/design_space_explorer.py:37-56` generate_trace_from_spec；`:111-151` simulate_layer；`:67-108` _compute_kv_cycles（batch_m>1→0 路径）。
   - `sim/contracts/units.py:60-64` cycles_to_microseconds。
   - `sim/model_specs.py:27` qwen2.5-3b。
   - 需求文档 §3 FR-2/FR-3、§5（参考实现数值仅对照）。

  **Acceptance criteria**:
   - `uv run pytest sim/tests/test_prefill_ttft.py -q` 全绿。
   - `uv run ruff check sim/design_space_explorer.py sim/tests/test_prefill_ttft.py` 通过。

  **QA scenarios**:
   - **Happy**: 新测试绿；Evidence `.omo/evidence/task-3-arc-prefill-ttft-dse-prefill.json`。
   - **Failure**: mutation——把 `* num_layers` 删掉 → `test_ttft_ms_formula` 红；把 units 调用换成裸公式 → `test_ttft_uses_units` 红。Evidence `.omo/evidence/task-3-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `feat(dse): add simulate_prefill and ttft_ms_from_prefill via contracts.units` | `sim/design_space_explorer.py`, `sim/tests/test_prefill_ttft.py`

- [x] 4. ttft_ms 结果契约穿线（FR-5）：PPA/EngineMetrics/legacy + evaluate_config 双指标 + golden 同步

  **What to do**:
   1. `sim/engine/ppa_model.py` L19-40：`PPA` 增加 `ttft_ms: float = 0.0`。
   2. `sim/contracts/result.py` L95-127：`EngineMetrics` 增加 `ttft_ms: float = 0.0`（保持 `extra="forbid"`）。**三处构造点均新增 `ttft_ms=ppa.ttft_ms` 传参**（PPA 默认 0.0 兜底；加字段不传参会因 extra=forbid 校验失败或输出恒 0.0）：
      - `sim/design_space_explorer.py` `_build_v2_output` L591-599 附近；
      - `sim/dse/runner.py` L291-311；
      - `sim/contracts/result.py` `result_standalone_from_ppa` L259-267。
   3. `sim/contracts/legacy_result.py`：`_legacy_point`（L263-271，形参 `r`，函数内已 `m = r.metrics`）增加 `"ttft_ms": round(m.ttft_ms, 2)`；`_cv_legacy_point`（L274-297）增加 `"ttft_ms": 0.0`；**`legacy_result_dict_from_ppa`（L38-71）返回的基础 dict 增加 `"ttft_ms": round(ppa.ttft_ms, 2)`**（否则 T4 的 pinned 测试 `test_legacy_result_dict_llm` 断言 `d["ttft_ms"]` 会 KeyError）。
   4. `sim/design_space_explorer.py` `_result_dict`（L1215-1233，legacy CLI 输出条目的实际来源）：LLM 分支（L1216）增加 `"ttft_ms": round(p.ttft_ms, 2)`；CV 分支（L1218-1232）增加 `"ttft_ms": 0.0`（FR-5：非 CV 全条目含；CV 按 0.0 约定保持 schema 一致）。
   5. `evaluate_config(cfg, area_model, power_model, *, batch_m: int = 1)`：**FR-4 解耦——decode `tok_s` 永远由 `simulate_layer(cfg)`（M=1 语义）计算，与 `--batch-m` 无关**；TTFT 由 `prefill_cycles = simulate_prefill(cfg, batch_m, _MODEL_ALIAS)` + `ttft_ms_from_prefill(prefill_cycles, get_spec(_MODEL_ALIAS).layers, freq)` 计算（**层数取自 spec，不取 `_NUM_LAYERS` 全局**——scenario 路径 `_NUM_LAYERS=28` 会对 qwen2.5-3b 算错）；`PPA(..., ttft_ms=ttft)`。CV 分支 `ttft_ms=0.0`。模块级新增 `_MODEL_ALIAS = _DEFAULT_LLM_SPEC`；main() 设置 `_MODEL_ALIAS = model_spec`，**并同步更新 L980 `global` 语句为 `global _CV_MODEL, _CV_TRACE, _CV_ONNX_PATH, _NUM_LAYERS, _MODEL_ALIAS`**（否则赋值只创建局部变量，evaluate_config 仍读默认 alias）；runner 路径不设置，用默认值 qwen2.5-3b。
   6. 更新 pinned 测试（注意：现有断言均为**存在性断言**，加字段不会破坏，更新 = 增加新契约断言，非修复破坏）：
      - `sim/tests/test_result_schema.py`：文件顶部 `_FakePPA` 增加 `ttft_ms` 字段（默认 0.0，供 `legacy_result_dict_from_ppa` 读取）；`test_legacy_result_dict_llm`（L299-306）增加断言 `d["ttft_ms"] == 0.0`（用 fake 默认值）；`test_dse_v2_output_produces_valid_json`（L388-428）增加断言每个 `r["metrics"]["ttft_ms"] >= 0`。
      - 新增 `test_evaluate_config_ttft_uses_spec_layers`（放 test_prefill_ttft.py 或 test_result_schema.py）：构造 block 64×64 config，`evaluate_config(cfg, am, pm, batch_m=128)`，断言 `ppa.ttft_ms == round(simulate_prefill(cfg,128)*36/1000/1000, 2)`（36 = qwen2.5-3b spec layers；防 U5 回归到 `_NUM_LAYERS=28`）。
      - `sim/tests/golden/legacy_cli_contract.json`：`dse_json_output.fields.pareto_frontier.item_schema` 与 `top_results.item_schema` 各增加 `"ttft_ms": {"type": "float", "unit": "ms"}`（文档契约同步；无测试按 item 精确键集断言——golden 是契约描述不是输出快照）；**`flags["--batch-m"]` 段不动**（留给 T5 改）。
   7. 全量回归。

  **Must NOT do**:
   - 不改 `EngineMetrics`/`DesignPointResult`/`DesignSpaceResultV2` 的 `extra="forbid"` 策略。
   - 不改 golden 的顶层键结构（test_legacy_compatibility.py L184-194 精确顶层键断言必须继续成立）。
   - 不把 ttft_ms 塞进 config dict（参考实现的已知缺点——必须放结果结构体字段）。

  **Parallelization**: Wave 1 | Blocked by: 3 | Blocks: 5

  **References**:
   - `sim/engine/ppa_model.py:19-40`；`sim/contracts/result.py:95-127,239-284`；`sim/contracts/legacy_result.py:38-71,263-297`；`sim/dse/runner.py:199-226,291-311`；`sim/design_space_explorer.py:552-648,1215-1266`。
   - `sim/tests/test_result_schema.py:290-345,388-428`（含文件顶部 `_FakePPA`）；`sim/tests/test_legacy_compatibility.py:184-231`（item 存在性断言）；`sim/tests/golden/legacy_cli_contract.json:135-173`（item_schema 段）。

  **Acceptance criteria**:
   - `uv run pytest -q` 全绿（含更新后的 pinned 测试与 golden 对比）。
   - `uv run python sim/design_space_explorer.py --quick --output /tmp/t4.json` exit 0，且每个非 CV 结果含 `ttft_ms`（值 >0）。
   - `uv run python -c "import json; d=json.load(open('/tmp/t4.json')); assert all('ttft_ms' in r for r in d['results'])"` 通过。
   - 默认运行 tok_s/area/power 与 T1 baseline 一致（test_fr7_backcompat 绿）。

  **QA scenarios**:
   - **Happy**: 全量 pytest 绿 + ttft_ms 全条目存在；Evidence `.omo/evidence/task-4-arc-prefill-ttft-dse-contract.json`。
   - **Failure**: mutation——把 `_legacy_point` 的 `ttft_ms` 行删掉 → `test_result_schema.py` 红；把 EngineMetrics 的 `ttft_ms` 删掉 → v2 shape 测试红。Evidence `.omo/evidence/task-4-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `feat(contracts): thread ttft_ms through PPA, v2 EngineMetrics and legacy projection` | `sim/engine/ppa_model.py`, `sim/contracts/result.py`, `sim/contracts/legacy_result.py`, `sim/design_space_explorer.py`, `sim/dse/runner.py`, `sim/tests/test_result_schema.py`, `sim/tests/golden/legacy_cli_contract.json`

- [x] 5. CLI --batch-m 放宽 + quick dims (64,64) + FR-4 decode 语义（FR-4, FR-6）

  **What to do**:
   1. `sim/design_space_explorer.py` L933-935：`parser.add_argument("--batch-m", type=int, default=None, help="Prefill batch M (>=1). Drives prefill/TTFT metrics only; decode tok_s keeps batch_m=1 semantics")`（删除 `choices=[1,2]`）。
   2. L975-978 之后加校验：`if args.batch_m is not None and args.batch_m < 1: parser.error("--batch-m must be >= 1")`；互斥检查（L955-975）保持不变。
   3. main() L1004-1006：`_MODEL_ALIAS = model_spec`；`_NUM_LAYERS = get_spec(model_spec).layers`（保留）。L1037 调用改为 `evaluate_config(cfg, area_model, power_model, batch_m=batch_m)`（batch_m 为 None 时传 1：`batch_m = args.batch_m if args.batch_m is not None else 1`）。
   4. L1260 legacy dict 的 `"batch_m"` 键 = 上述 batch_m。
   5. `generate_configs` L188：quick dims 改 `[(64, 64), (128, 128), (128, 256), (256, 256)]`。
   6. 更新 `sim/tests/golden/legacy_cli_contract.json` 的 `design_space_explorer_cli.flags["--batch-m"]` 段（L135-139）：`type` 改 `"integer"`、删除 `choices`、description 改为新 help 文案（契约文档同步；现有测试只断言 flag 名存在，不比对 choices）。
   7. `sim/tests/test_dse_strict.py` L28/41：若 monkeypatch stub 不接受关键字参数，更新 stub 签名接受 `**kwargs`（main() 现在传 `batch_m=`）。
   8. FR-4 验证测试（放 test_prefill_ttft.py）：`--batch-m 128` 与默认运行同配置的 `tok_s` 完全一致（**两次运行结果按 `config_label` 做字典匹配，不依赖输出顺序**）；`batch_m` 键=128；`ttft_ms` >0 且 > 默认运行 M=1 的 ttft_ms。
   9. 参数校验测试（放 test_prefill_ttft.py 或 test_legacy_compatibility.py）：subprocess 实测 `--batch-m 0` exit≠0 且 stderr 含 "must be >= 1"；`--batch-m 2.5` argparse 拒绝（exit≠0，非整数类型错误）；`--batch-m 3 --quick --output <tmp>` exit 0 且 JSON `batch_m == 3`。

  **Must NOT do**:
   - 不改变 `--replay/--scenario/--cv-model/--model-spec` 与 `--batch-m` 的互斥逻辑（L955-975 原文保留）。
   - 不改 `--quick` 其它维度（dram/precisions/freqs/sram）。

  **Parallelization**: Wave 2 | Blocked by: 4 | Blocks: 6

  **References**:
   - `sim/design_space_explorer.py:917,933-935,955-978,1004-1006,1037,1259-1260,187-190`。
   - `sim/tests/test_legacy_compatibility.py:234-259`；`sim/tests/golden/legacy_cli_contract.json` flags 段。
   - `sim/tests/test_dse_strict.py:28,41`。
   - 需求文档 §3 FR-4/FR-6。

  **Acceptance criteria**:
   - `uv run pytest -q` 全绿（含更新的 legacy compat 与 golden）。
   - `uv run python sim/design_space_explorer.py --batch-m 0 --quick 2>&1 | tail -1; echo exit=$?` → parser.error，exit≠0。
   - `uv run python sim/design_space_explorer.py --help | grep -q "decode tok_s keeps batch_m=1 semantics"`（FR-4 help 文案存在）。
   - `uv run python sim/design_space_explorer.py --quick --batch-m 128 --output /tmp/t5.json` exit 0；Block 64×64 行存在且 `ttft_ms>0`；同配置 `tok_s` 与默认运行一致（脚本比对）。

  **QA scenarios**:
   - **Happy**: 上述三条命令逐条实测；Evidence `.omo/evidence/task-5-arc-prefill-ttft-dse-cli.json`。
   - **Failure**: `--batch-m 0` 若被接受（exit 0）→ 红；Evidence `.omo/evidence/task-5-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `feat(dse): widen --batch-m to >=1, add (64,64) to quick dims` | `sim/design_space_explorer.py`, `sim/tests/golden/legacy_cli_contract.json`, `sim/tests/test_legacy_compatibility.py`, `sim/tests/test_dse_strict.py`, `sim/tests/test_prefill_ttft.py`

- [ ] 6. 交叉验收不变量套件（AC-1, NFR-4 mutation）

  **What to do**:
   扩展 `sim/tests/test_prefill_ttft.py`（复用 T5 的 /tmp 跑法改为测试内构造）：
   1. `test_ttft_monotonic_in_m`: 同一 Block 64×64 配置（wc=True, LPDDR5-64b, 1000MHz），ttft_ms 严格递增：M=32 < 64 < 128。
   2. `test_ttft_bw_floor`（sanity，非硬门——物理下界，10% 容差）: wc=False 时，`ttft_ms >= 0.9 × weights_bytes_total/(bandwidth_gbps*1e9/8)*1000`，其中 `weights_bytes_total = 36 × (spec.hidden × spec.intermediate × 3 + spec.intermediate × spec.hidden) × 0.5`（INT4 = 0.5 字节/参数；`spec = get_spec("qwen2.5-3b")`，hidden=2048、intermediate=11008）——独立于引擎代码的物理下界；**测试内打印 `margin = ttft_ms / floor_ms`（验收直接读取实际裕量）**。
   3. `test_ttft_linearity_2000_128`: `ttft(M=2000)/ttft(M=128)` ∈ [14.0, 17.5]（≈2000/128=15.625 ±10%，容忍 FFN/attn 占比差异）。
   4. `test_decode_unchanged_with_batch_m`（FR-4）：CLI 级或 evaluate_config 级断言 batch_m=128 时 `tok_s` == batch_m=1 时 `tok_s`。
   5. `test_cv_ttft_zero`：CV 分支 PPA.ttft_ms == 0.0。
   6. `test_all_quick_engines_prefill_finite`（U2）：`--quick --batch-m 128` 运行后，输出中**每个**结果 `ttft_ms > 0` 且有限（不只 Block——systolic/gmma 等 quick 引擎也必须安全）。
   7. `test_prefill_output_deterministic`（NFR-1）：同一命令 `--quick --batch-m 128 --result-schema v2 --output <tmp1>/<tmp2>` 跑两次，两文件 sha256 相同。
   8. Mutation 演示（QA 阶段执行，不留代码）：任意破坏一处（如 ttft 公式乘 1.01）→ 至少一条不变量红。

  **Must NOT do**:
   - 不新增对 Func Model 数值的硬编码依赖（比值判断只在 T7/T8 证据里做）。
   - 不引入 sleep/网络/外部依赖。

  **Parallelization**: Wave 2 | Blocked by: 5 | Blocks: 7,8

  **References**:
   - `sim/tests/test_prefill_ttft.py`（T3/T5 已建）；`sim/design_space_explorer.py:169-261` generate_configs；`sim/model_specs.py:27`。
   - 需求文档 §5/§6（AC-1）、§2（PASS 区间背景）。

  **Acceptance criteria**:
   - `uv run pytest sim/tests/test_prefill_ttft.py -q` 全绿且 ≥10 个测试。
   - BW floor 测试真实执行（非 skip）并给出实际裕度打印。

  **QA scenarios**:
   - **Happy**: 全绿 + 裕度打印；Evidence `.omo/evidence/task-6-arc-prefill-ttft-dse-invariants.json`。
   - **Failure**: mutation（ttft 公式 ×1.01）→ 红；Evidence `.omo/evidence/task-6-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `test(dse): prefill TTFT acceptance invariants (linearity, BW floor, FR-4, CV, mutation)` | `sim/tests/test_prefill_ttft.py`

- [ ] 7. M=128 证据 + Gate 1b 目标抽取（NFR-1/2, AC-4）

  **What to do**:
   1. `uv run python sim/design_space_explorer.py --quick --batch-m 128 --result-schema v2 --output .omo/evidence/task-7-arc-prefill-ttft-dse-m128.json`
   2. 用脚本抽取行：`engine=="block"` 且 dims==[64,64] 且 freq==1000 且 dram=="LPDDR5-64b" 且 wc∈{true,false} → 记录 `ttft_ms`。**若该行缺失，立即失败并打印输出中所有 block 行**（防 U1 静默抽空）。
   3. 计算比值判定：`Func 3911.05 / ttft_ms ∈ [0.5, 2.0]`（wc=True 与 wc=False 各一条判定；Gate 1b 目标以 **wc=True** 为准，wc=False 记录为参考）。判定用一次性 python 断言执行：`assert 0.5 <= 3911.05/ttft_wc_true <= 2.0`。
   4. **参考值（规划期实测推算，36 层）**：wc=T ≈ 3211 ms（比值 ≈1.22）；wc=F ≈ 6457 ms（比值 ≈0.61）。偏差 >10% 时必须在证据中给出解释（BW/compute 分解）。
   5. 记录 sha256（`sha256sum`），证据 JSON 提交 git（NFR-2）。
   6. 输出 `.omo/evidence/task-7-arc-prefill-ttft-dse-m128-extract.txt`：抽取行 + 比值 + 判定 + sha256。

  **Must NOT do**:
   - 不为落入窗口调任何模型/引擎/校准参数。若超窗：记录 FAIL 判定 + 诊断（该配置的 BW-bound vs compute-bound 分解、与 Func Model 的 7-op/17-op 结构差异），照常提交并汇报——禁止数值凑数。
   - 不改 `--quick` 其它维度来"碰巧"得到目标配置。

  **Parallelization**: Wave 3 | Blocked by: 6 | Blocks: 9 | Parallel with: 8

  **References**:
   - `sim/design_space_explorer.py:187-190`（quick dims 已含 64×64）、`:212`（freq=1000）。
   - 需求文档 §2（PASS 区间 0.5×-2.0×、Func Model 3911.05 ms）、§5（参考实现 2649.49 ms 仅对照）、§7（回流产物清单）。
   - 规划期探针记录（`.omo/drafts/arc-prefill-ttft-dse.md`）：当前公式实测 wc=T M=128 = 2497.4 ms@28层 → ×36/28 ≈ 3211 ms。

  **Acceptance criteria**:
   - 证据 JSON + extract txt 存在且已 git 提交。
   - extract 含 wc=True 行，比值判定 ∈ [0.5, 2.0]（预期 ≈1.22，容差 ±10% 需解释）。
   - `test $? -eq 0`（CLI 运行成功，errors=0）。

  **QA scenarios**:
   - **Happy**: 上述抽取与判定完整；Evidence 即产出文件本身。
   - **Failure**: 故意在 extract 脚本里把窗口改成 [0.9,1.1] → 必须输出 FAIL 判定路径（验证判定逻辑真实生效）；Evidence `.omo/evidence/task-7-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `evidence(dse): M=128 prefill TTFT gate-1b target extraction` | `.omo/evidence/task-7-arc-prefill-ttft-dse-*.json/.txt`

- [ ] 8. M=2000 证据 + Gate 1b 目标抽取（同 T7，规模 M=2000）

  **What to do**:
   同 T7，参数替换：`--batch-m 2000`；Func Model 实测 **63,924.19 ms**；参考值 wc=T ≈ **50,172 ms**（比值 ≈1.27）、wc=F ≈ 100,897 ms（比值 ≈0.63）。产物 `.omo/evidence/task-8-arc-prefill-ttft-dse-m2000.json` / `-extract.txt`。

  **Must NOT do**: 同 T7（禁止凑数；超窗则如实记录）。

  **Parallelization**: Wave 3 | Blocked by: 6 | Blocks: 9 | Parallel with: 7

  **References**: 同 T7 + 需求文档 §2（M=2000 → 63,924.19 ms）。

  **Acceptance criteria**: 同 T7（比值预期 ≈1.27 @wc=T）。

  **QA scenarios**: 同 T7。

  **Commit**: YES | `evidence(dse): M=2000 prefill TTFT gate-1b target extraction` | `.omo/evidence/task-8-arc-prefill-ttft-dse-*.json/.txt`

- [ ] 9. 文档同步 + §7 回流产物（NFR-3, AC-3/4/5）

  **What to do**:
   1. `docs/arc_vs_func.md` L141-143：stale 行（"prefill latency ❌ 未实现 / TTFT ❌ 未实现 / Arc 只算 decode M=1"）改为已实现描述 + 证据路径引用（`simulate_prefill`、`--batch-m`、`.omo/evidence/task-7/8-*`）。L156-157 roadmap：勾除 `simulate_prefill` 条目并标注完成日期/证据。
   2. `README.md`：新增**最小化** "DSE 能力矩阵" 表（只补 NFR-3 要求的 prefill TTFT 行：decode tok_s（M=1 语义）、prefill TTFT（`--batch-m ≥1`）、CV FPS、replay/seed 可复现、v2 schema、legacy 投影、证据入库；**不撰写完整能力矩阵**——避免超范围）；并检查 L34/38/77/207/311 的 TTFT 相关表述是否 stale（如 "TTFT 45ms/160ms 估算" 属场景估算，不改数值，仅在能力矩阵说明 TTFT 现可经 DSE 直接产出）。
   3. 回流产物 `.omo/evidence/gate-1b-dse-ttft-targets.md`（§7 清单）：
      - 目标值：Block 64×64 @1GHz LPDDR5-64b INT4（WC=True 为主，False 参考）M=128 / M=2000 ttft_ms + Func Model 比值判定（[0.5×, 2.0×]）；
      - 证据路径 + SHA-256（task-7/8 JSON）；
      - 实施说明：KV prefill 写回=0 简化、7-op vs 17-op 结构差异、与参考实现（2649.49/41398.27 ms）数值差异原因（引擎公式已修正 + 36 vs 28 层）；
      - arc_vs_func.md 更新片段引用。
   4. 全仓 `uv run ruff check .`。

  **Must NOT do**:
   - 不修改历史 dated report（reports/*.md）。
   - 不把回流产物写成"要求一致"——只报告数值、差异与判定。
   - 不碰 CaduceusCore 侧文档（§8 A2 由对方执行）。

  **Parallelization**: Wave 3 | Blocked by: 7,8 | Blocks: F1-F4

  **References**:
   - `docs/arc_vs_func.md:141-143,153-157`；`README.md`（§1.4、§五、TTFT 提及处）。
   - 需求文档 §7（回流产物清单）、§5（参考数值与差异说明要求）。
   - `.omo/evidence/task-7/8-arc-prefill-ttft-dse-*-extract.txt`（目标值来源）。

  **Acceptance criteria**:
   - `grep -c "TTFT.*未实现\|prefill.*未实现" docs/arc_vs_func.md` = 0。
   - README 含 "DSE 能力矩阵" 标题且表内有 prefill TTFT 行。
   - `.omo/evidence/gate-1b-dse-ttft-targets.md` 存在、含 M=128/M=2000 数值 + 比值判定 + 两个 sha256，且已 git 提交。
   - `uv run ruff check .` exit 0。

  **QA scenarios**:
   - **Happy**: 三条 grep/存在性检查逐条实测；Evidence `.omo/evidence/task-9-arc-prefill-ttft-dse-docs.json`。
   - **Failure**: mutation——故意在 docs 里留一条 "未实现" → grep 命中即红；Evidence `.omo/evidence/task-9-arc-prefill-ttft-dse-negative.txt`。

  **Commit**: YES | `docs(arc): sync TTFT capability docs + publish gate-1b deliverable` | `docs/arc_vs_func.md`, `README.md`, `.omo/evidence/gate-1b-dse-ttft-targets.md`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [ ] F1. Plan compliance audit

  预检查（D3）：先列齐预期证据文件（`.omo/evidence/task-0..9-arc-prefill-ttft-dse-*`、`final-arc-prefill-ttft-f1..f4-*`、`gate-1b-dse-ttft-targets.md`）并逐项 `ls` 确认存在后再跑 verifier。

  Verify every Todo 0-9 acceptance criterion against actual files/evidence/commits.

  ```bash
  uv run python scripts/verify_evidence_ledger.py \
    --plan .omo/plans/arc-prefill-ttft-dse.md \
    --evidence-root .omo/evidence \
    --output .omo/evidence/final-arc-prefill-ttft-f1-plan-compliance.json
  ```

  **APPROVE iff** exit 0、Todos/F1-F4 schema 可解析、Todo 0-9 每项有匹配 commit/evidence、all blocking skipped/xfail=0。

- [ ] F2. Code quality and model-integrity review

  ```bash
  uv run ruff format --check . > .omo/evidence/final-arc-prefill-ttft-f2-code-quality.txt 2>&1
  uv run ruff check . >> .omo/evidence/final-arc-prefill-ttft-f2-code-quality.txt 2>&1
  uv run basedpyright >> .omo/evidence/final-arc-prefill-ttft-f2-code-quality.txt 2>&1
  uv run pytest -q >> .omo/evidence/final-arc-prefill-ttft-f2-code-quality.txt 2>&1
  uv run python scripts/verify_model_integrity.py \
    --output .omo/evidence/final-arc-prefill-ttft-f2-code-quality.json
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
    --output .omo/evidence/final-arc-prefill-ttft-f3-manual-qa.json

  # 新增 prefill 专项（本计划特有）：
  uv run python sim/design_space_explorer.py --quick --batch-m 128 --result-schema v2 --output /tmp/f3-m128.json
  uv run python sim/design_space_explorer.py --quick --batch-m 0 2>&1 | tail -2   # 必须 parser.error
  ```

  **APPROVE iff** release_gate exit 0、`legacy_failures=[]`、`workload_failures=[]`、`coverage.missing=[]`、`errors=0`、`experimental_gate=pass`；/tmp/f3-m128.json 生成成功且含 Block 64×64 `ttft_ms>0`；`--batch-m 0` 被拒绝。

- [ ] F4. Scope and evidence fidelity

  ```bash
  uv run python scripts/verify_scope.py \
    --plan .omo/plans/arc-prefill-ttft-dse.md \
    --baseline-commit "$(git merge-base HEAD origin/main)" \
    --publication-manifest docs/publication-manifest.yaml \
    --output .omo/evidence/final-arc-prefill-ttft-f4-scope-fidelity.json
  ```

  **APPROVE iff** exit 0、`forbidden_dependencies=[]`、`ultraresearch_changes=[]`、`historical_report_changes=[]`、`unbound_current_claims=[]`。

## Commit strategy

- conventional commits；每个 todo 的 implementation+test 为一个原子 commit。
- 不 amend、不 squash 已发布历史；修复使用独立 `fix(...)` commit。
- 每个 commit 只包含该 todo 的 Files 列表与其证据。
- T7/T8 可并行（独立证据运行）；T9 依赖 T7/T8 的 extract 数值。
- 只在全部 F1-F4 通过后才标记计划完成。

## Success criteria

- `_LLM_TRACE`/`_KV_HEADS`/`_HEAD_DIM` 全局消失；trace 按 batch_m 参数化；KV 几何作为参数传递。
- `simulate_prefill` + `ttft_ms_from_prefill`（经 contracts.units）可用；KV 写回=0 简化在代码与文档中显式声明。
- `--batch-m` 任意 ≥1；`--batch-m 0` 被拒绝；decode `tok_s` 与 batch_m 解耦（FR-4）。
- 所有非 CV 结果（v2 + legacy）含 `ttft_ms`；CV 模式 0.0。
- quick dims 含 (64,64)；默认运行与 T1 baseline 逐配置一致（FR-7）。
- M=128 / M=2000 证据 JSON 已入库（sha256 可查），Block 64×64 @1GHz LPDDR5-64b INT4 的比值判定 ∈ [0.5×, 2.0×]（预期 ≈1.2-1.3× @WC）。
- `docs/arc_vs_func.md` stale 清除、roadmap 更新；README 含 DSE 能力矩阵。
- `.omo/evidence/gate-1b-dse-ttft-targets.md` 回流产物齐备（§7 四项）。
- F1、F2、F3、F4 全部以 `verdict=PASS` 和 exit 0 完成。
