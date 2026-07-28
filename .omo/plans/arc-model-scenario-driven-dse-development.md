# Arc Model 场景化 DSE 开发与验证计划

## TL;DR

> Summary:      先修复当前 Arc Model 的物理不变量与单位/兼容性基础，再建立统一内存层次、版本化 workload graph、具身智能/Physical-AI 时序执行器和 scenario-driven DSE；任何架构排名只有在校准、覆盖与复现门全部通过后才可标记为 authoritative。
> Deliverables:
> - 8 引擎共享且不可绕过的 cycle/吞吐/利用率/诊断契约
> - 明确区分 request batch、active sequences、token block、image count、action horizon、flow steps、inflight jobs 的版本化 schema
> - SRAM、3D on-chip DRAM、LPDDR、HBM 的统一 residency/spill、带宽、PPA 与能耗模型
> - LLM、CV、SmolVLA、π0、OpenVLA/OFT/FAST、Helix、Physical-AI multi-job 可执行 workload fixtures
> - 确定性的队列、优先级、抢占/分区、P50/P99、deadline、过载与恢复仿真
> - scenario 驱动的设计点生成、多目标 Pareto、coverage manifest 与 replay bundle
> - 参数来源、校准范围、误差预算、可信度等级和完整 agent-executed 验收证据
> Effort:       XL
> Risk:         High - 当前 63 项测试通过但仍可复现频率无效、M 边界反向、利用率超过 1、3D DRAM 未进入搜索空间等物理违例

## Scope

### Must have

- 把当前 README/历史 postfix 报告视为历史证据，不把 “63 passed” 当成物理正确性的证明；为本计划重新捕获 commit/config/依赖绑定的 baseline。
- 规范化运算计数：
  - `mac_count = M × K × N`，双 GEMM pair 为 `2 × M × K × N`；
  - `op_count = 2 × mac_count`；
  - `peak_macs_per_cycle = array_height × array_width`；
  - `peak_ops_per_cycle = peak_macs_per_cycle × ops_per_mac`；
  - legacy `EngineResult.ops` 暂投影为 `mac_count` 并标记 deprecated，禁止再用含混字段建立新 oracle。
- 规范化单位：
  - 配置权威带宽单位为 decimal `GB/s`；
  - `bytes_per_cycle = bandwidth_gbps × 1000 / frequency_mhz`；
  - `seconds = cycles / (frequency_mhz × 1e6)`；
  - compute-bound 的 cycle 不随频率变化、wall time 近似反比；
  - 固定 GB/s 的纯 memory-bound wall time在跨频率时保持不变，仅 cycle 数随频率变化。
- 所有 8 个 factory engine 必须满足：
  - 正整数 `M/K/N` 与合法硬件配置；
  - `total_cycles >= ceil(mac_count / peak_macs_per_cycle)`；
  - `total_cycles >= ceil(raw_transfer_bytes / effective_bytes_per_cycle)`；
  - `0 < utilization <= 1`；
  - 固定硬件/带宽下，处理更多相同工作时 total latency 不得反向下降；
  - 必需 diagnostics 缺失必须 fail，禁止 `skip`。
- `mac_engine` 为 schema v2 的规范硬件键；legacy `mxu` 仅由 adapter 接受。两者同时存在且归一化后不一致时抛字段明确的配置错误；一致时允许并记录迁移 warning。
- 版本化 declarative graph 是 workload 的权威表示；JSON 是无损序列化，ONNX 只作为 lowering adapter。首阶段不依赖 PyTorch 或 ROS。
- graph 必须包含稳定 node/tensor ID、op、依赖、shape/符号维、layout、precision、bytes、lifetime、alias、resource hint 与 provenance。
- operator registry 只能返回 `modeled`、`explicitly_free_or_fused` 或 `unsupported`；任意未注册 op 在 importer 和 executor 边界都必须 fail-closed。
- edge batch 验收：
  - `request_batch` / `active_sequences`: 标准 `{1,2,4,8}`，压力 `{16}`；
  - `token_block`: `{16,32,64,128,256}`，VLM/VLA 扩展 `{512,1024}`；
  - `image_count`: `{1,2,3,4}`；
  - `action_horizon`: `{8,10,25,50}`；
  - `flow_steps`: `{4,8,10}`；
  - `resident_models`: `{4,8}`；
  - `inflight_jobs`: `{4,8,16}`。
- typed memory modes必须区分 `sram`、`on_chip_3d_dram`、`lpddr5`、`lpddr5x`、`hbm2e`、`hbm3`，并验证 PHY/TSV/package 组件组合。
- 3D DRAM 首版为可替换参数化 macro backend，必须建模 capacity、读写带宽/延迟、容量相关面积、TSV/接口面积、静态功耗、访问能耗、活动功耗和 thermal proxy；不要求 Ramulator/DRAMSim。
- byte-accurate footprint 至少包含 persistent weights、KV、activations、scratch、queue buffers、allocator alignment 与 reserve fraction；placement 返回 full/partial/spill 明细并满足容量守恒。
- 所有 engine 必须消费同一个 `MemoryAccessPlan`，不得保留 Block-only 的 “capacity > 0 即完全驻留” 快路。
- 提供以下独立可执行 workload，而不是只写需求 YAML：
  - compact SmolVLA-class；
  - continuous π0-class；
  - OpenVLA baseline、OFT、FAST 三个独立 variant；
  - Helix S2/S1 与 optional S0 多速率图；
  - Physical-AI multi-camera/multi-job DAG；
  - 现有 LLM/CV fixtures 的 graph 化等价版本。
- 外部来源事实、工程 sweep、待校准值必须有不同 provenance class；无来源的精确值不能伪装成市场事实。
- scenario executor 必须确定性定义 periodic/trace arrival、warm-up、measurement window、nearest-rank P50/P99、arrival-to-completion deadline、FIFO/mailbox_latest、queue capacity/drop、priority tie-break、preemption point/cost、partition、resource arbitration、cancellation 与 recovery。
- 利用率定义为 measurement window 内每个资源的 busy_time/window_time；分别验证 90–95% stable offered load 与 >100% overload/recovery，禁止把 offered load 与 achieved utilization 混写。
- DSE 必须由 scenario schema 生成设计点；每个设计点拥有由 normalized config + scenario + workload digest 派生的稳定 ID，结果携带完整配置而非依赖位置关联。
- 结果 schema v2 必须包含 simulator commit、依赖锁摘要、输入/config/workload/calibration 摘要、seed、status、错误、完整 metrics/PPA/memory plan、coverage manifest link 与 trust level。
- partial run 必须标记 `non_authoritative`；任何未覆盖、失败或未校准的关键轴不得进入 release recommendation。
- 旧入口继续可用：
  - `sim/design_space_explorer.py` 保留 `--quick/--allow-partial/--output/--top/--cv-model/--model-spec/--batch-m`；
  - `--batch-m 1` 映射 legacy decode，`--batch-m 2` 映射 legacy two-token prefill；与新 batch flags 同时使用时报错；
  - `sim/npu_sim.py` 既有 flags/exit codes/legacy JSON 字段保持；
  - 新 scenario CLI 默认输出 schema v2，旧 CLI 默认 legacy projection，并提供显式 `--result-schema v2`。

### Must NOT have

- 不修改 Func Model 数值黄金模型、RTL、训练质量或机器人安全控制器逻辑。
- 不在首阶段引入 PyTorch、ROS、完整模型权重、在线下载或必须联网的测试。
- 不在首阶段强制集成 Ramulator/DRAMSim；只保留稳定 backend protocol 和将来的 adapter seam。
- 不用 `min(utilization, 1)`、强行 clamp 周期或调常数来掩盖错误公式；先修 root formula，再由契约验证。
- 不把 Systolic 与另一套可能共享同一错误公式的生产 estimator 互比称为“独立 oracle”。
- 不把 metadata/fused op 的 0 cycle 扩展到未注册算子。
- 不把 HBM 当作 on-chip 3D DRAM；HBM 必须保留独立 PHY/TSV/package 成本。
- 不以模型名称的近似参数量代替 tensor footprint；所有 residency 由图和精度计算。
- 不使用结果/配置列表的位置关联；错误、过滤后仍须靠稳定 design-point ID 关联。
- 不从未校准点发布绝对性能/PPA结论；只能标记 exploratory 并展示敏感度/不确定性。
- 不修改、移动或提交 `.omo/ultraresearch/20260723-vla-models/sources/`。

## Verification strategy

> Zero human intervention - all verification is agent-executed.

- Test decision: strict TDD with `pytest`;每个 todo 先提交可复现红色测试/fixture，再做最小实现并跑相关回归。
- Independent oracle policy:
  - oracle 只使用 closed-form seconds/bytes/MAC conservation，禁止调用 production engine/memory/scheduler estimator；
  - 小矩阵 golden 采用手算可枚举事件；
  - metamorphic tests 验证频率、带宽、容量、batch、并发、功耗单调性与边界；
  - graph adapter 采用 canonical JSON round-trip 和缩小 ONNX golden 等价。
- Blocking-test policy:
  - blocking suite 禁止 `skip`、`xfail`；
  - focused suite、完整 pytest、quick scenario DSE、full coverage dry-run 和 replay 都必须记录 collected/passed/failed/skipped/exit code；
  - 任何 blocking skip、错误计数、missing coverage 或 digest 不匹配均失败。
- Environment:
  - 新增 `pyproject.toml` 与 `uv.lock`；
  - 支持 CPython `>=3.10,<3.13`；
  - 锁定当前必要运行/测试依赖（NumPy、PyYAML、Pydantic v2、ONNX、pytest、ruff、basedpyright），测试 fixture 离线可用。
- Evidence: `.omo/evidence/task-<N>-<slug>.<ext>`；每条证据同时记录命令、exit code、git commit、依赖锁 digest、输入/config digest。
- QA policy: 每个 todo 都含 happy path 与 failure/negative path；任何一项不得以 grep 命中、worker 自述或历史 JSON 代替实际执行。
- Time budgets:
  - focused unit/property suite ≤ 60s；
  - quick scenario suite ≤ 5min；
  - full acceptance 可分片，但每片必须有 coverage manifest，聚合后 expected=visited+pruned+failed。

## Execution strategy

### Parallel execution waves

> Target 5-8 todos per wave. Wave 0 为可信基础阻断门；后续两条实现 lane 只有在共同契约通过后才并行。

```
Wave 0A.1 — baseline freeze
└── Todo 1: 可复现环境、历史 baseline 与兼容性冻结

Wave 0A.2 — contract red gate (after Todo 1, parallel)
├── Todo 2: schema v2、单位、错误与 legacy migration 契约
└── Todo 3: 全引擎独立物理 oracle 红色矩阵

Wave 0B.1 — shared engine contract (after 0A)
└── Todo 4: engine registry/result/input/diagnostic 契约

Wave 0B.2 — physical repairs (after Todo 4, parallel)
├── Todo 5: Systolic/OS/GMMA/TensorCore 物理公式修复
└── Todo 6: 频率/带宽单位端到端修复

Wave 1A — canonical contracts (after 0B, parallel)
├── Todo 7: declarative workload graph + operator registry
└── Todo 9: result v2、稳定 ID 与 legacy projection

Wave 1B — adapters (after Todo 7, parallel with late Todo 9 work)
└── Todo 8: JSON/ONNX adapters + legacy trace lowering

Wave 2A — two parallel product lanes (after Wave 1)
├── Todo 10: 统一 memory hierarchy/residency/spill
└── Todo 12: LLM/CV/VLA/Physical-AI executable fixtures

Wave 2B — 3D backend completion (after Todo 10, parallel with late Todo 12 work)
└── Todo 11: 参数化 3D DRAM PPA/energy backend

Wave 3A — temporal kernel (after all Wave 2)
└── Todo 13: 确定性资源/队列/抢占调度器

Wave 3B — temporal metrics (after Todo 13)
└── Todo 14: P50/P99/deadline/高利用率/过载恢复 metrics

Wave 4A — scenario space (after Wave 3)
└── Todo 15: scenario-to-space 枚举、约束和 coverage manifest

Wave 4B — Pareto/replay (after Todo 15)
└── Todo 16: 多目标 Pareto、replay bundle 与兼容 CLI

Wave 5A — calibration gate (after Wave 4)
└── Todo 17: 参数 provenance、校准、误差预算与 trust gate

Wave 5B — release (after Todo 17)
└── Todo 18: 全矩阵端到端验收、文档和发布证据

Final wave (after ALL todos, parallel)
├── F1: Plan compliance audit
├── F2: Code quality and model-integrity review
├── F3: Real CLI/scenario/replay QA
└── F4: Scope and evidence fidelity
```

Critical path: `1 → (2,3) → 4 → (5,6) → 7 → (8 → 12, 10 → 11) → 13 → 14 → 15 → 16 → 17 → 18 → F1-F4`; Todo 9 在 Todo 6 后与 7/8 并行，但须在 Todo 15 前完成。

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|---|---|---|---|
| 1 | — | 2, 3 | — |
| 2 | 1 | 4, 7, 9 | 3 |
| 3 | 1 | 4, 5, 6 | 2 |
| 4 | 2, 3 | 5, 6, 7 | — |
| 5 | 4 | 7, 10 | 6 |
| 6 | 4 | 7, 9, 10 | 5 |
| 7 | 5, 6 | 8, 10, 12 | 9 |
| 8 | 7 | 12 | 9 |
| 9 | 2, 6 | 15, 16 | 7, 8 |
| 10 | 7 | 11, 13 | 12 |
| 11 | 10 | 13, 15, 17 | 12 |
| 12 | 7, 8 | 13, 15 | 10, 11 |
| 13 | 10, 11, 12 | 14 | — |
| 14 | 13 | 15, 17 | — |
| 15 | 9, 11, 12, 14 | 16 | — |
| 16 | 15 | 17, 18 | — |
| 17 | 11, 14, 16 | 18 | — |
| 18 | 16, 17 | F1-F4 | — |

## Todos

> Implementation + Test = ONE todo. Never separate.

- [ ] 1. 冻结可复现环境、历史 baseline 与公共兼容面

  **What to do**:
  1. 新增 `pyproject.toml` 与 `uv.lock`，声明 CPython `>=3.10,<3.13`，锁定 NumPy、PyYAML、Pydantic v2、ONNX、pytest、ruff、basedpyright；不新增 Hypothesis，property matrix 使用 pytest 参数化。`pyproject.toml` 同时固定 ruff 与 basedpyright 的项目检查范围。
  2. 新增 `sim/tests/test_legacy_compatibility.py`，实际运行并 snapshot：
     - `python sim/npu_sim.py --engine systolic --json`
     - `python sim/npu_sim.py --json`
     - `python sim/design_space_explorer.py --quick`
     - DSE `--model-spec/--batch-m/--cv-model/--allow-partial` 的 help、exit code 与 legacy JSON 顶层字段。
  3. 新增 `sim/tests/golden/legacy_cli_contract.json`，只冻结公共命令、flags、exit codes、字段名与单位，不冻结已知错误的频率数值。
  4. 新增 `sim/tests/test_environment_repro.py` 验证 lock、离线 fixtures 和必需 standalone assets。
  5. 在 `README.md` 明确旧 63-test/postfix 证据只证明历史 bug 集合，不证明跨频率、batch、内存和高利用率正确；记录本计划 baseline commit/config/lock digest。

  **Must NOT do**:
  - 不修改任何 engine 公式或 rebaseline 已知错误行为。
  - 不把 `.omo/ultraresearch/20260723-vla-models/sources/` 加入依赖或测试。

  **Parallelization**: Can parallel NO | Wave 0A | Blocks 2,3 | Blocked by —

  **References**:
  - `README.md:313-358` — 当前把 63 passed/证据链描述为完整复现，需限定适用范围。
  - `pytest.ini:1-4` — 当前唯一测试入口配置。
  - `sim/design_space_explorer.py:541-559` — 必须冻结的 legacy DSE flags。
  - `sim/npu_sim.py:477-594` — 必须冻结的 npu_sim flags 与 JSON 输出。
  - `sim/tests/test_standalone_assets.py:13-47` — 现有 asset-presence 模式。

  **Acceptance criteria**:
  - `uv sync --frozen` 在 clean venv 退出 0，随后 `uv run pytest --collect-only -q` 退出 0。
  - `uv run pytest sim/tests/test_legacy_compatibility.py sim/tests/test_environment_repro.py -q` 无 skip/xfail。
  - golden 明确记录 legacy schema，不包含绝对路径、时间戳或未跟踪 source 文件。
  - `git status --short` 仍只显示实施产生的预期文件和原有未跟踪 ultraresearch 目录。

  **QA scenarios**:
  - **Happy**: 在临时 uv 环境执行 legacy CLI snapshots，两次运行字段集合/exit code 相同；Evidence `.omo/evidence/task-1-legacy-baseline.json`。
  - **Failure**: 从临时复制中删除 lock 或 required fixture，`test_environment_repro` 必须失败并指出缺失项；Evidence `.omo/evidence/task-1-repro-negative.txt`。
  - **Commands**:
    ```bash
    uv sync --frozen
    uv run pytest sim/tests/test_legacy_compatibility.py sim/tests/test_environment_repro.py -q -k "baseline or snapshot or clean" > .omo/evidence/task-1-legacy-baseline.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_environment_repro.py -q -k "missing_lock or missing_fixture" > .omo/evidence/task-1-repro-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `chore(repro): pin arc-model environment and freeze legacy contracts` | `pyproject.toml`, `uv.lock`, `README.md`, `sim/tests/test_environment_repro.py`, `sim/tests/test_legacy_compatibility.py`, `sim/tests/golden/legacy_cli_contract.json`

- [ ] 2. 建立 schema v2、单位、错误与 legacy migration 契约

  **What to do**:
  1. 新建 `sim/contracts/units.py`，只在此实现 GB/s↔bytes/cycle、cycles↔seconds/us、bytes/GiB 换算；所有换算使用显式 decimal GB 与 binary GiB 名称。
  2. 新建 `sim/contracts/errors.py`，定义 `ConfigError`、`SchemaVersionError`、`DimensionBindingError`、`UnsupportedOperatorError`、`CoverageError`、`NonAuthoritativeRunError`。
  3. 新建 `sim/contracts/hardware.py`，用 Pydantic v2 定义 hardware/memory schema：
     - v2 规范键为 `mac_engine`；
     - 只出现 `mxu` 时迁移；
     - 两者一致时接受并产生 structured warning；
     - 不一致时 fail-closed；
     - unknown field 默认 forbid。
  4. 新建 `sim/contracts/migrations.py`，提供 v1→v2 pure migration 和 v2→legacy projection；输入不原地修改。
  5. 更新 `sim/config/npu_config.py`，顶层/嵌套 YAML 非 mapping、非法版本、bool 充当 int、非有限/非正数必须产生字段明确的 typed error。
  6. 新增 `sim/tests/test_contract_schema.py` 与 `sim/tests/test_units.py`，覆盖 800/1000/1200 MHz、25.6–819.2 GB/s 和 malformed YAML。

  **Must NOT do**:
  - 不让 production modules继续直接解释 `bandwidth_bytes_per_cycle` 配置字段。
  - 不静默选择冲突的 `mxu`/`mac_engine`。

  **Parallelization**: Can parallel YES | Wave 0A | Blocks 4,7,9 | Blocked by 1 | Parallel with 3

  **References**:
  - `sim/config/npu_config.yaml:11-21` — legacy `mxu` 结构。
  - `sim/config/design_space.yaml:6-27` — `mac_engine` 与混合带宽字段。
  - `sim/config/npu_config.py:1-15` — 当前无 schema 的 YAML loader。
  - `sim/engine/mac_engine.py:38-60` — 当前分散、含混的 config/带宽解析。
  - `sim/design_space_explorer.py:245-249` — 当前把 GB/s 数值直接写为 bytes/cycle。

  **Acceptance criteria**:
  - `uv run pytest sim/tests/test_contract_schema.py sim/tests/test_units.py -q` 全部通过且无 skip。
  - 对固定 51.2 GB/s，800/1000/1200 MHz 分别得到 64/51.2/42.666… bytes/cycle；换算回 wall-time 误差 `<1e-12`。
  - `mxu`/`mac_engine` 冲突、NaN/Inf、0/负频率、0/负带宽、bool 数值、错误 YAML shape 均抛确定的 typed error。
  - migration round-trip 保留可表达字段；不可表达字段列入 structured loss report。

  **QA scenarios**:
  - **Happy**: 加载两个现有 YAML，迁移为 v2，序列化后重载并比较 normalized model；Evidence `.omo/evidence/task-2-schema-roundtrip.json`。
  - **Failure**: 参数化 malformed configs，断言 exact error type 与 field path；Evidence `.omo/evidence/task-2-schema-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_contract_schema.py sim/tests/test_units.py -q -k "roundtrip or conversion or legacy" > .omo/evidence/task-2-schema-roundtrip.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_contract_schema.py sim/tests/test_units.py -q -k "malformed or conflict or nonfinite or rejects" > .omo/evidence/task-2-schema-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(contracts): add versioned hardware schema and canonical units` | `sim/contracts/__init__.py`, `sim/contracts/units.py`, `sim/contracts/errors.py`, `sim/contracts/hardware.py`, `sim/contracts/migrations.py`, `sim/config/npu_config.py`, `sim/tests/test_contract_schema.py`, `sim/tests/test_units.py`

- [ ] 3. 建立全引擎独立物理 oracle 红色矩阵

  **What to do**:
  1. 新建 `sim/tests/oracles/physics.py`，用 closed-form MAC、byte、秒单位实现独立下界；禁止 import `sim.engine.*` estimator。
  2. 新建 `sim/tests/test_engine_physical_invariants.py`，对 factory 的 8 个 engine × `M={1,2,3,4,8,16,64,256,1024}` × 小/非整除 K,N × LPDDR/HBM/high-BW 做参数化。
  3. 强制断言：
     - 正确的 `mac_count/op_count`；
     - peak MAC lower bound；
     - raw DMA ceil lower bound；
     - utilization `(0,1]`；
     - M 增加时 total work latency不下降；
     - bandwidth 单调且在 compute floor 饱和；
     - diagnostics key 集合完整。
  4. 新建 `sim/tests/test_engine_invalid_inputs.py`，覆盖所有 `estimate` 和 `estimate_weight_cache_pair` 的 0/负数/bool/float/string 形状以及非法 array/precision/bandwidth。
  5. 先记录当前 red manifest，必须至少捕获 Systolic M=2→3、OS M scaling、GMMA peak floor 和 TensorCore partial tile 风险。

  **Must NOT do**:
  - 本 todo 不改 production engine。
  - 不用已有 MXUModel 作为唯一 oracle。

  **Parallelization**: Can parallel YES | Wave 0A | Blocks 4,5,6 | Blocked by 1 | Parallel with 2

  **References**:
  - `sim/engine/mac_engine.py:8-29,98-115` — 当前 EngineResult 与含混 peak 定义。
  - `sim/engine/systolic_engine.py:21-145` — M=2/3 分支。
  - `sim/engine/os_systolic_engine.py:44-109` — 当前 compute 未正确扩展 M。
  - `sim/engine/gmma_engine.py:77-139` — pipeline scale 与 raw DMA floor。
  - `sim/engine/tensor_core_engine.py:72-158` — partial M tile 风险。
  - `sim/tests/test_engine_result_contract.py:33-117` — 当前 narrow shape 与 skip diagnostics。

  **Acceptance criteria**:
  - red 阶段输出精确 failing node IDs 与反例参数，不允许 collection error。
  - oracle 模块经 AST/rg 验证不 import production engine estimator。
  - invalid-input tests 覆盖 8 engine direct estimate 以及有 override 的 cache-pair path。
  - red manifest 中每个失败都映射到 Todo 4/5/6，不得通过删断言解决。

  **QA scenarios**:
  - **Happy/collect**: `uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_invalid_inputs.py --collect-only -q`；Evidence `.omo/evidence/task-3-physical-collect.txt`。
  - **Failure/reproduction**: 执行 red suite，输出 counterexample JSON；Evidence `.omo/evidence/task-3-physical-red.json`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_invalid_inputs.py --collect-only -q > .omo/evidence/task-3-physical-collect.txt 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_invalid_inputs.py -q > .omo/evidence/task-3-physical-red.json 2>&1
    test $? -ne 0
    ```

  **Commit**: YES | `test(engine): add independent physical invariant matrix` | `sim/tests/oracles/__init__.py`, `sim/tests/oracles/physics.py`, `sim/tests/test_engine_physical_invariants.py`, `sim/tests/test_engine_invalid_inputs.py`

- [ ] 4. 统一 engine registry、result、输入和 diagnostic 契约

  **What to do**:
  1. 新建 `sim/engine/registry.py`，以单一 registry 注册全部 8 engine；factory、DSE、npu_sim choices、report iteration 全部从 registry 派生。
  2. 重构 `sim/engine/mac_engine.py`：
     - 接受 normalized `HardwareConfig`；
     - 校验正整数 shape/array/precision；
     - `EngineResult` 新增 `mac_count/op_count/ideal_compute_cycles/raw_dma_cycles`；
     - legacy `ops` 保留只读 alias；
     - result construction 校验数值有限、cycle 非负、必需 diagnostics 存在。
  3. 将 `test_engine_result_contract.py` 的 diagnostics `pytest.skip` 改为 hard failure；每个 engine 注册 required diagnostic schema。
  4. 更新 `sim/npu_sim.py` 和 `sim/design_space_explorer.py` 的 engine 列表、help 与 “best per engine” 逻辑，不再靠 label 截断匹配。
  5. 保持 legacy engine 名称；未知 engine 抛 typed `ConfigError`。

  **Must NOT do**:
  - 不在 result validator 中 clamp 利用率或修改 engine 返回周期。
  - 不保留第二份手写 engine choices。

  **Parallelization**: Can parallel NO | Wave 0B | Blocks 5,6,7 | Blocked by 2,3

  **References**:
  - `sim/engine/mac_engine.py:127-157` — 当前 factory。
  - `sim/npu_sim.py:485-515` — 当前遗漏 FSA 的 CLI choices。
  - `sim/design_space_explorer.py:170-175,596-600,698-706` — 当前 8/7/2 engine 列表分叉与 label 匹配。
  - `sim/tests/test_dse_coverage.py:19-63` — registry coverage 的现有起点。
  - `sim/tests/test_engine_result_contract.py:52-113` — 需移除 skip。

  **Acceptance criteria**:
  - registry、factory、DSE full、npu_sim list 的 engine set 完全等于 8 个规范 ID。
  - 每个 engine result 都包含并通过 required diagnostics；人为删除字段测试必须红而非 skip。
  - `uv run pytest sim/tests/test_engine_instantiate.py sim/tests/test_engine_result_contract.py sim/tests/test_dse_coverage.py -q -rs` 输出 0 skipped。
  - legacy CLI 每个 engine 名能实例化实际请求 class。

  **QA scenarios**:
  - **Happy**: registry-driven factory/CLI/DSE coverage；Evidence `.omo/evidence/task-4-engine-registry.json`。
  - **Failure**: 注入未知 engine 和缺失 diagnostic，验证 typed error/hard failure；Evidence `.omo/evidence/task-4-engine-contract-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_engine_instantiate.py sim/tests/test_engine_result_contract.py sim/tests/test_dse_coverage.py -q -k "registry or instantiate or coverage or diagnostics" > .omo/evidence/task-4-engine-registry.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_engine_result_contract.py sim/tests/test_engine_instantiate.py -q -k "unknown or missing or rejects" > .omo/evidence/task-4-engine-contract-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `refactor(engine): centralize registry and result contracts` | `sim/engine/registry.py`, `sim/engine/mac_engine.py`, `sim/npu_sim.py`, `sim/design_space_explorer.py`, `sim/tests/test_engine_result_contract.py`, `sim/tests/test_engine_instantiate.py`, `sim/tests/test_dse_coverage.py`

- [ ] 5. 修复 Systolic、OS、GMMA、TensorCore 的物理公式

  **What to do**:
  1. `systolic_engine.py` 统一 M mapping，使 M=2→3 不再出现 total latency 下降；decode/prefill 只作为 workload 语义，不作为两个不连续硬件公式的隐藏阈值。
  2. `os_systolic_engine.py` 加入 M tiles/passes、尾 tile 有效行和正确 K reduction；area penalty 只由 PPA 参数处理，不混入 timing。
  3. `gmma_engine.py`：
     - pipeline compute 不得低于 ideal MAC floor；
     - raw DMA 用 `ceil`；
     - direct/cache-pair 都满足 physical floor；
     - pipeline scale 保留为校准参数但不能产生超峰值。
  4. `tensor_core_engine.py` 对最后一个 partial M/K/N tile 使用实际有效尺寸计算 payload、descriptor 和 compute。
  5. 审计 Block、WMMA、InputStationary、FSA 的同类边界；只有红色 invariant 暴露问题时才做最小修复。
  6. 使 Todo 3 red suite 全绿，同时保留现有 63-test 回归。

  **Must NOT do**:
  - 不以特定 Qwen 排名为目标调 descriptor/pipeline 常数。
  - 不用 `max(ideal, buggy_total)` 作为唯一修复；result 必须由正确 tile/work 公式自然满足 lower bound，公共 validator 只负责拒绝。

  **Parallelization**: Can parallel YES | Wave 0B | Blocks 7,10 | Blocked by 4 | Parallel with 6

  **References**:
  - `sim/engine/systolic_engine.py:21-145` — decode/prefill 分支。
  - `sim/engine/os_systolic_engine.py:44-109` — M-independent compute。
  - `sim/engine/gmma_engine.py:77-139,141-214` — direct/cache-pair floors。
  - `sim/engine/tensor_core_engine.py:72-158` — partial M payload。
  - `sim/engine/ppa_model.py:43-84` — OS 与 Block area baseline 分叉需在后续校准说明。

  **Acceptance criteria**:
  - Todo 3 全矩阵通过，0 skipped/xfail。
  - 固定 64×64/high-BW 下所有 engine 的 `M={1,2,3,4,8,16,64,256,1024}` 均 `utilization<=1` 且 total latency 不下降。
  - GMMA `M=64,K=N=64` 不低于独立 ideal floor；OS `M=1024` 不再返回与 M=1 相同 compute。
  - TensorCore M=17 payload 等于 full first tile + one-row tail 的独立 byte oracle。

  **QA scenarios**:
  - **Happy**: 全 engine physical matrix；Evidence `.omo/evidence/task-5-engine-physical-green.json`。
  - **Failure**: 对 production result 注入超峰值/向下取整 DMA，validator 必须拒绝；Evidence `.omo/evidence/task-5-physical-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_result_contract.py sim/tests/test_engines.py -q > .omo/evidence/task-5-engine-physical-green.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_engine_physical_invariants.py -q -k "rejects_injected_invalid" > .omo/evidence/task-5-physical-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `fix(engine): enforce physical timing bounds across batch shapes` | `sim/engine/systolic_engine.py`, `sim/engine/os_systolic_engine.py`, `sim/engine/gmma_engine.py`, `sim/engine/tensor_core_engine.py`, 必要时其他被 invariant 证实有问题的 engine 文件

- [ ] 6. 修复跨频率与跨带宽的端到端单位传播

  **What to do**:
  1. 删除 DSE `tok_s_from_layer` 的 1000 MHz 常数和 CV `1e9/cycles` 常数，统一使用 `units.py`。
  2. `generate_configs` 只写 `bandwidth_gbps`；engine construction 时按每个 design point 频率计算 bytes/cycle。
  3. 修复 `npu_sim.py --freq`：override 必须在模型创建前归一化，或显式刷新所有消费 frequency 的组件。
  4. 审计 KV、DMA、DRAM、NoC、PPA 和 report 路径，禁止用 GB/s 数值冒充 bytes/cycle。
  5. 新增 `sim/tests/test_frequency_bandwidth_scaling.py`：
     - compute-only closed-form at 800/1000/1200 MHz wall time按 `1000/f` 比例，容差 0.1%；
     - fixed 51.2 GB/s memory-only wall time差异 ≤0.1%，cycle 按频率比例；
     - LPDDR5→HBM3 单调并在 compute floor 饱和；
     - CLI/DSE output 与底层报告一致。

  **Must NOT do**:
  - 不修改 physical bandwidth 随 core frequency 一起缩放。
  - 不把功耗随频率变化误当成 throughput 已正确传播。

  **Parallelization**: Can parallel YES | Wave 0B | Blocks 7,9,10 | Blocked by 4 | Parallel with 5

  **References**:
  - `sim/design_space_explorer.py:148-151,245-249,265-276` — 三处已确认单位错误。
  - `sim/npu_sim.py:76-97,489-572` — component 初始化与 CLI override。
  - `sim/engine/mac_engine.py:45-60` — engine bandwidth consumption。
  - `sim/models/dram.py:13-58` — DRAM bandwidth/refresh。
  - `sim/engine/ppa_model.py:146-180` — frequency/power coupling。

  **Acceptance criteria**:
  - `uv run pytest sim/tests/test_units.py sim/tests/test_frequency_bandwidth_scaling.py -q` 全绿无 skip。
  - DSE 相同 Block config 在 800/1000/1200 MHz 的 compute-bound tok/s 严格递增且比例误差 ≤0.1%。
  - memory-bound wall time跨频率误差 ≤0.1%，HBM/LPDDR byte conservation exact。
  - `npu_sim --freq 800/1000/1200 --json` 输出不再相同，并与独立换算 oracle 一致。

  **QA scenarios**:
  - **Happy**: 三频点 × LPDDR/HBM × compute/memory-bound matrix；Evidence `.omo/evidence/task-6-frequency-bandwidth.json`。
  - **Failure**: 注入 legacy `bandwidth_bytes_per_cycle` 与冲突 GB/s，schema 必须拒绝；Evidence `.omo/evidence/task-6-unit-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_units.py sim/tests/test_frequency_bandwidth_scaling.py -q -k "compute_bound or memory_bound or bandwidth or cli" > .omo/evidence/task-6-frequency-bandwidth.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_contract_schema.py sim/tests/test_frequency_bandwidth_scaling.py -q -k "conflict or legacy_bytes_per_cycle or rejects" > .omo/evidence/task-6-unit-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `fix(units): propagate frequency and bandwidth consistently` | `sim/design_space_explorer.py`, `sim/npu_sim.py`, `sim/engine/mac_engine.py`, `sim/models/dram.py`, `sim/models/dma.py`, `sim/models/kv_cache.py`, `sim/engine/ppa_model.py`, `sim/tests/test_frequency_bandwidth_scaling.py`

- [ ] 7. 实现版本化 declarative workload graph 与 fail-closed operator registry

  **What to do**:
  1. 新建 `sim/workloads/schema.py`，定义 `WorkloadGraphV1`、`TensorSpec`、`NodeSpec`、`SymbolicDim`、`Provenance`：
     - stable IDs、inputs/outputs/dependencies；
     - shape 支持正整数或命名符号；
     - layout/precision/bytes；
     - lifetime/alias 只能引用存在 tensor；
     - graph 必须 DAG；
     - unknown fields/version fail。
  2. 新建 `sim/workloads/dimensions.py`，规范字段：
     - `request_batch`、`active_sequences`、`token_block`、`image_count`、`action_horizon`、`flow_steps`、`resident_models`、`inflight_jobs`；
     - 每个字段绑定到命名 symbolic axis 或 temporal expansion；
     - 禁止用一个维度隐式替代另一个。
  3. 新建 `sim/workloads/operators.py`，registry entry 明确 `modeled`、`explicitly_free_or_fused`、`unsupported`；free/fused 必须带 fused_into/provenance，不能只因没有实现而为 0 cycle。
  4. 新建 `sim/workloads/validate.py`，在 lowering/execution 前完成 graph、dimension、operator、tensor lifetime 验证。
  5. 新增 `sim/tests/test_workload_schema.py`、`test_dimension_semantics.py`、`test_operator_registry.py`。

  **Must NOT do**:
  - 不把 `profile_required/unknown` 当成默认值。
  - 不接受未绑定 symbolic dimension 或 cycle 为 0 的 arbitrary op。

  **Parallelization**: Can parallel YES | Wave 1 | Blocks 8,10,12 | Blocked by 5,6 | Parallel with 9

  **References**:
  - `sim/design_space_explorer.py:34-72` — 当前 `batch_m` 混合 graph 与 temporal 语义。
  - `sim/cv/cv_trace.py:7-22,210-261` — 当前 dict trace schema/unknown lowering。
  - `sim/cv/cv_sim.py:26-33,178-191` — 当前 unknown op 落到 0-cycle metadata。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:48-284` — graph/temporal/batch 字段来源。

  **Acceptance criteria**:
  - JSON round-trip 后 normalized graph 完全相等，stable ID 不变。
  - 构造 cycle、悬空 tensor、重复 ID、非法 alias、未绑定符号、未知 op、unsupported schema version 均 typed fail。
  - `request_batch` 改变不能自动改变 `active_sequences/token_block`；所有维度有 orthogonality test。
  - 明确 free/fused node 才允许 cost=0，且结果记录 fused target。

  **QA scenarios**:
  - **Happy**: 构造含 GEMM/SFU/vector/fused reshape 的最小 DAG，绑定维度并 round-trip；Evidence `.omo/evidence/task-7-workload-graph.json`。
  - **Failure**: 参数化 malformed graph 和未知 op，全部 fail-closed；Evidence `.omo/evidence/task-7-workload-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_workload_schema.py sim/tests/test_dimension_semantics.py sim/tests/test_operator_registry.py -q -k "not negative" > .omo/evidence/task-7-workload-graph.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_workload_schema.py sim/tests/test_dimension_semantics.py sim/tests/test_operator_registry.py -q -k "negative or rejects" > .omo/evidence/task-7-workload-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(workload): add versioned graph and operator contracts` | `sim/workloads/__init__.py`, `sim/workloads/schema.py`, `sim/workloads/dimensions.py`, `sim/workloads/operators.py`, `sim/workloads/validate.py`, `sim/tests/test_workload_schema.py`, `sim/tests/test_dimension_semantics.py`, `sim/tests/test_operator_registry.py`

- [ ] 8. 实现 JSON/ONNX adapters 与 legacy trace lowering

  **What to do**:
  1. 新建 `sim/workloads/json_adapter.py`，canonical JSON 采用排序键、稳定 float/enum 表示、schema version 和 digest。
  2. 新建 `sim/workloads/onnx_adapter.py`：
     - 保留 symbolic dimension 名，不再转 0；
     - 只 lowering registry 中 modeled/free/fused ops；
     - 动态 shape 必须由 `DimensionBindings` 完整绑定后才能 cost；
     - 未知/unsupported op 抛 `UnsupportedOperatorError`，包含 node/opset/path。
  3. 新建 `sim/workloads/legacy_adapter.py`：
     - 将现有 LLM tuple trace 与 CV dict trace 转成 graph；
     - `--batch-m 1`→legacy decode (`active_sequences=1`)；
     - `--batch-m 2`→legacy two-token prefill (`token_block=2`)；
     - legacy batch 与新 batch flags 同时出现 fail。
  4. 改造 `sim/cv/onnx_importer.py`、`cv_trace.py`、`cv_sim.py` 通过统一 adapter/registry；删除 unmatched→metadata 0 path。
  5. 新增缩小离线 ONNX golden `sim/tests/fixtures/tiny_mixed_ops.onnx` 及 equivalence tests。

  **Must NOT do**:
  - 不在线下载 ONNX 模型。
  - 不改变 legacy trace 的已有 modeled op 数值，除非是已确认的 unknown-op 0-cycle 缺口。

  **Parallelization**: Can parallel YES | Wave 1 | Blocks 12 | Blocked by 7 | Parallel with 9

  **References**:
  - `sim/cv/onnx_importer.py:45-72,171-213` — symbolic dim→0 与 whitelist。
  - `sim/cv/cv_trace.py:175-261` — importer→trace mapping。
  - `sim/cv/cv_sim.py:82-191` — execution dispatch/fail-open。
  - `sim/design_space_explorer.py:34-53,557-588` — legacy LLM trace/CLI。

  **Acceptance criteria**:
  - tiny ONNX→graph 与手写 canonical JSON 在 node/tensor/shape/op/precision 上等价，digest 相同。
  - symbolic batch 未绑定时 fail，绑定 1/2/4/8 后产生不同但稳定 graph instance ID。
  - LayerNorm/Softmax/GELU/MaxPool/Upsample 必须明确 modeled 或 unsupported；任何一个都不得再以 unknown 0 cycles 通过。
  - legacy LLM/CV adapter 的代表性 modeled op cycle 与旧路径一致；冲突 batch flags exit 2。

  **QA scenarios**:
  - **Happy**: tiny ONNX 与 JSON golden 双向规范化比较；Evidence `.omo/evidence/task-8-adapter-equivalence.json`。
  - **Failure**: 含未知 op 与未绑定 symbolic batch 的 ONNX/JSON，验证 typed errors；Evidence `.omo/evidence/task-8-adapter-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_workload_adapters.py sim/tests/test_legacy_compatibility.py -q -k "roundtrip or equivalence or legacy" > .omo/evidence/task-8-adapter-equivalence.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_workload_adapters.py -q -k "unsupported or unbound or conflicting" > .omo/evidence/task-8-adapter-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(workload): add json onnx and legacy adapters` | `sim/workloads/json_adapter.py`, `sim/workloads/onnx_adapter.py`, `sim/workloads/legacy_adapter.py`, `sim/cv/onnx_importer.py`, `sim/cv/cv_trace.py`, `sim/cv/cv_sim.py`, `sim/tests/fixtures/tiny_mixed_ops.onnx`, `sim/tests/test_workload_adapters.py`

- [ ] 9. 实现 result schema v2、稳定 design-point ID 与 legacy projection

  **What to do**:
  1. 新建 `sim/contracts/result.py`，定义：
     - `RunStatus={complete,partial,failed,filtered}`；
     - `TrustLevel={authoritative,calibrated_estimate,exploratory,non_authoritative}`；
     - full normalized hardware/scenario/workload/calibration references；
     - average/P50/P99/max、throughput、deadline miss/drop/replacement、resource utilization、memory/PPA/energy；
     - generated/evaluated/pruned/failed/error summary。
  2. 新建 `sim/contracts/identity.py`，以 canonical normalized JSON SHA-256 生成 design_point_id、input/config/workload/calibration/lock digest；禁止用 label 或列表索引。
  3. 新建 `sim/contracts/legacy_result.py`，保留当前 LLM/CV JSON 顶层字段与单位；loss report 标出 v2-only 数据。
  4. `--allow-partial` 结果强制 `partial + non_authoritative`；如果 release-required axis 有 error/missing，不得生成 recommendation。
  5. 新增 deterministic serialization、filtered/error association 与 legacy snapshot tests。

  **Must NOT do**:
  - 不在 ID 中加入时间戳、绝对路径或 iteration order。
  - 不把 error text 作为唯一错误分类；保存 sanitized typed code + bounded details。

  **Parallelization**: Can parallel YES | Wave 1 | Blocks 15,16 | Blocked by 2,6 | Parallel with 7,8

  **References**:
  - `sim/design_space_explorer.py:604-659` — 当前 error/filter 流。
  - `sim/design_space_explorer.py:716-781` — 当前 positional config/result 与输出 metadata。
  - `sim/engine/ppa_model.py:15-31` — 当前 PPA result。
  - `sim/tests/test_dse_strict.py:14-67` — partial behavior 起点。

  **Acceptance criteria**:
  - 相同 normalized input 两次生成相同 ID/JSON digest；任一设计轴变化 ID 必变。
  - error/filter 后 result 始终携带正确完整 config，不依赖位置。
  - partial run 必为 non_authoritative，release recommendation API 对其抛 `NonAuthoritativeRunError`。
  - legacy snapshots 保持 Todo 1 冻结字段；v2 round-trip lossless。

  **QA scenarios**:
  - **Happy**: 打乱 config 枚举顺序，两次 v2 results 按 ID 比较完全相等；Evidence `.omo/evidence/task-9-result-determinism.json`。
  - **Failure**: 注入一项 error/一项 filtered，验证关联、状态和 recommendation 拒绝；Evidence `.omo/evidence/task-9-result-negative.json`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_result_schema.py sim/tests/test_result_identity.py -q -k "deterministic or roundtrip or reordered" > .omo/evidence/task-9-result-determinism.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_result_schema.py sim/tests/test_result_identity.py -q -k "partial or filtered or rejects or mismatch" > .omo/evidence/task-9-result-negative.json 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(result): add stable v2 identities and legacy projection` | `sim/contracts/result.py`, `sim/contracts/identity.py`, `sim/contracts/legacy_result.py`, `sim/tests/test_result_schema.py`, `sim/tests/test_result_identity.py`

- [ ] 10. 实现统一 memory hierarchy、residency 与 spill

  **What to do**:
  1. 新建 `sim/models/memory_hierarchy.py`，定义 typed tier、capacity、read/write bandwidth、latency、alignment、reserve fraction。
  2. 新建 `sim/models/residency.py`，输入 graph tensor lifetimes + resident models + queue buffers，输出 immutable `MemoryAccessPlan`：
     - persistent weights → KV → live activations/scratch → queues 的显式优先级；
     - 256-byte 默认 alignment（可配置）；
     - full/partial/spill bytes 与目的 tier；
     - eviction/reload/access bytes；
     - capacity conservation：每 tier allocated+reserved≤capacity。
  3. 更新所有 engine、KV、DMA、CV/LLM simulator 接收同一个 plan；移除 `MACEngine.weight_resident` 正数 predicate 和 Block-only 全驻留特殊判断。
  4. 对外部 LPDDR/HBM 与 3D tier 的带宽竞争都通过计划中的 access streams 计算。
  5. 新增 `sim/tests/oracles/memory.py` 和 capacity boundary tests：`footprint=capacity-1/aligned equal/+1`、多 resident model、KV/activation queue。

  **Must NOT do**:
  - 不按 model alias/参数量猜 footprint。
  - 不让不同 engine 对相同 graph/config 得到不同 residency。

  **Parallelization**: Can parallel YES | Wave 2 | Blocks 11,13 | Blocked by 7 | Parallel with 12

  **References**:
  - `sim/engine/mac_engine.py:68-76` — 当前 capacity>0 predicate。
  - `sim/engine/block_engine.py:55-97` — 当前 Block-only on-chip path。
  - `sim/cv/cv_sim.py:193-227` — 当前只做 aggregate activation spill。
  - `sim/models/kv_cache.py:18-144` — KV footprint/access 模型。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:91-95,149-153,207-211,264-268` — reservable/reserve requirements。

  **Acceptance criteria**:
  - 0.001/0.1/5/16 GB 对同一模型不再都判 full resident；full/partial/spill 随 capacity 单调。
  - placement byte conservation exact，所有 tier overflow 为 0；非法 reserve/alignment/capacity fail。
  - 8 engine 在同一 graph/config 的 `MemoryAccessPlan.digest` 相同。
  - spill 增加不能使 memory traffic/latency 下降。

  **QA scenarios**:
  - **Happy**: sweep capacity 覆盖 full/partial/spill 三态；Evidence `.omo/evidence/task-10-memory-residency.json`。
  - **Failure**: capacity 小于 reserved bytes、非法 alias lifetime、无 spill tier 时 fail；Evidence `.omo/evidence/task-10-memory-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_memory_residency.py -q -k "full or partial or spill or conservation" > .omo/evidence/task-10-memory-residency.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_memory_residency.py -q -k "rejects or overflow or invalid" > .omo/evidence/task-10-memory-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(memory): unify hierarchy residency and spill planning` | `sim/models/memory_hierarchy.py`, `sim/models/residency.py`, `sim/engine/mac_engine.py`, `sim/engine/block_engine.py`, 其他 engine integration files, `sim/models/kv_cache.py`, `sim/cv/cv_sim.py`, `sim/npu_sim.py`, `sim/tests/oracles/memory.py`, `sim/tests/test_memory_residency.py`

- [ ] 11. 实现可替换参数化 3D DRAM PPA/energy backend

  **What to do**:
  1. 新建 `sim/models/memory_backend.py` protocol，输入 topology/capacity/bandwidth/access/read-write mix/activity，输出 latency/area/static power/dynamic energy/thermal proxy/validity envelope。
  2. 新建 `sim/models/onchip_dram.py` 的 `Parametric3DMemoryBackend`：
     - memory die area 随 capacity 单调；
     - TSV/interface area 随 bandwidth/lane 单调；
     - leakage 随 capacity；
     - dynamic energy 随 read/write bytes；
     - active power=energy/time；
     - 超出参数校准范围标 exploratory，不外推 authoritative。
  3. 更新 `ppa_model.py`，区分 on-chip DRAM、HBM、LPDDR 组件；修复 OS 专用 area baseline 未消费问题；HBM 保留 PHY+TSV/package，on-chip 去外部 PHY。
  4. 配置 `sim/config/memory_macros.yaml`，每个参数包含 value/unit/provenance/source/status/range；未校准默认 `engineering_assumption`。
  5. 新增独立 closed-form PPA/energy oracle 和 backend conformance tests，为未来 Ramulator/DRAMSim adapter 固定 protocol。

  **Must NOT do**:
  - 不用固定 10% TSV 作为全部容量/带宽成本。
  - 不把工程假设标为 calibrated 或 authoritative。

  **Parallelization**: Can parallel YES | Wave 2 | Blocks 13,15,17 | Blocked by 10 | Parallel with 12

  **References**:
  - `sim/engine/ppa_model.py:34-128` — 当前 area/固定 TSV 路径。
  - `sim/engine/ppa_model.py:132-180` — 当前 power 仍读 external memory bandwidth。
  - `sim/config/design_space.yaml:88-124` — 当前 area constants/TSV。
  - `sim/config/scenarios.yaml:84-105` — memory component taxonomy。

  **Acceptance criteria**:
  - capacity 增加时 memory area/leakage 不下降；bandwidth 增加时 interface area/active power 不下降；access bytes 增加时 energy 不下降。
  - 0.1/5/16 GB 与 100/500/1000 GB/s 产生不同且可由 oracle 复算的 PPA。
  - on-chip、HBM、LPDDR component manifest 分别满足规则；非法 PHY/TSV 组合 fail。
  - backend protocol fake implementation 可替换运行同一 conformance suite。

  **QA scenarios**:
  - **Happy**: capacity×bandwidth×access sweep 输出 PPA/energy/validity；Evidence `.omo/evidence/task-11-3d-dram-macro.json`。
  - **Failure**: 非法/超范围参数分别 typed fail 或 exploratory，绝不 authoritative；Evidence `.omo/evidence/task-11-3d-dram-negative.json`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_memory_backend.py sim/tests/test_memory_ppa.py -q -k "monotonic or energy or substitute" > .omo/evidence/task-11-3d-dram-macro.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_memory_backend.py sim/tests/test_memory_ppa.py -q -k "invalid or extrapolat or rejects" > .omo/evidence/task-11-3d-dram-negative.json 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(ppa): add parametric 3d memory backend` | `sim/models/memory_backend.py`, `sim/models/onchip_dram.py`, `sim/engine/ppa_model.py`, `sim/config/memory_macros.yaml`, `sim/tests/oracles/ppa.py`, `sim/tests/test_memory_backend.py`, `sim/tests/test_memory_ppa.py`

- [ ] 12. 建立 LLM、CV、VLA 与 Physical-AI 可执行 fixtures

  **What to do**:
  1. 创建 `sim/config/workloads/` 下 source-pinned、schema-valid 的 fixtures：
     - `llm-qwen25-3b.yaml`、`cv-yolov8n.yaml`、`cv-vit-b16.yaml`；
     - `smolvla-class.yaml`；
     - `pi0-class.yaml`；
     - `openvla-baseline.yaml`、`openvla-oft.yaml`、`openvla-fast.yaml`；
     - `helix-multirate.yaml`；
     - `physical-ai-multijob.yaml`。
  2. 将 source facts 与 engineering sweep 分开：
     - SmolVLA: `<0.5B` 与 async chunk 为 source facts；具体 component split 若无直接来源则 assumption；
     - π0: multi-image、H=50、10 flow steps；
     - OFT chunk 8/25，FAST token compression 作为 variant，不把论文 speedup 当本地模型常数；
     - Helix S2 7–9Hz、S1 200Hz、optional S0 1kHz；未知 action horizon 显式 optional，不填假值；
     - physical multi-job 用现有 CV graph 组合成 periods 33.333/50/83.333/100ms。
  3. 明确 batch axis sweeps 与合法组合；每个 fixture 解析为完整 graph+scenario，不保留验收关键字段的 `profile_required/target_measurement/unknown`。
  4. 新增 `sim/workloads/catalog.py` 与 coverage tests，确保每个 named workload 独立可发现、可实例化、可 footprint。
  5. 旧 `embodied-physical-ai-requirements.example.yaml` 保留为需求来源，新增迁移文档指向 executable fixtures。

  **Must NOT do**:
  - 不下载或提交完整模型权重。
  - 不把 OpenVLA/OFT/FAST 合并成一个 fixture。

  **Parallelization**: Can parallel YES | Wave 2 | Blocks 13,15 | Blocked by 7,8 | Parallel with 10,11

  **References**:
  - `sim/model_specs.py:26-42` — 当前只有 LLM/CV alias。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:48-284` — 需求基线与未决字段。
  - `sim/cv/traces/yolov8n_trace.py`, `vit_trace.py`, `qwen_vl_vit_trace.py` — 可复用视觉图。
  - External provenance: SmolVLA `https://arxiv.org/html/2506.01844v1`; π0 `https://arxiv.org/html/2410.24164v1`; OFT `https://arxiv.org/html/2502.19645v1`; Helix `https://www.figure.ai/news/helix`; Helix 02 `https://www.figure.ai/news/helix-02`.

  **Acceptance criteria**:
  - catalog 精确包含 10 个上述 fixture，全部 schema-valid、无 unresolved required 字段。
  - batch/dimension coverage manifest 包含 active 1/2/4/8+16 stress、token blocks、images、horizons、flow steps、resident/inflight axes。
  - 每个 fixture 有 deterministic node/tensor counts、footprint digest 和 provenance summary golden。
  - source fact 与 engineering assumption 分栏；不存在无 source 的 market_source 数值。

  **QA scenarios**:
  - **Happy**: load/instantiate 全 catalog 并输出 coverage/provenance；Evidence `.omo/evidence/task-12-workload-catalog.json`。
  - **Failure**: 删除 dimension binding、source tag 或 required graph stage，catalog validation 必须失败；Evidence `.omo/evidence/task-12-workload-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_workload_catalog.py -q -k "catalog or coverage or provenance" > .omo/evidence/task-12-workload-catalog.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_workload_catalog.py -q -k "missing or unresolved or rejects" > .omo/evidence/task-12-workload-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(workload): add embodied and physical-ai fixture catalog` | `sim/workloads/catalog.py`, `sim/config/workloads/*.yaml`, `sim/config/embodied-physical-ai-requirements.example.yaml`, `sim/tests/golden/workload_catalog.json`, `sim/tests/test_workload_catalog.py`

- [ ] 13. 实现确定性事件、资源竞争、队列与抢占执行器

  **What to do**:
  1. 新建 `sim/scheduler/events.py`、`kernel.py`，内部时间统一 integer picoseconds；cycle→ps 使用向上取整。事件顺序固定 `(time_ps, phase, insertion_sequence)`，同一时刻依次 release、arrival/timer、dispatch。
  2. 新建 `sim/scheduler/resources.py`：
     - compute engine/DMA channel 为 capacity resource；
     - SRAM ports/FIFO 有界；
     - LPDDR/HBM/3D/NoC 为 work-conserving byte server；
     - 同权共享为默认，strict-priority bandwidth QoS 显式选择；
     - membership 变化时确定性重算完成时间。
  3. 新建 `queues.py`：
     - bounded FIFO 满时 terminal `queue_full`，不静默丢；
     - mailbox_latest 每 stream/context 只保留一个 pending，新到达只替换 pending，不替换 running，记录 observation age/replacement。
  4. 新建 `policies.py`、`admission.py`：
     - service-class priority，类内 EDF，再 release time、stable job ID；
     - deadline=`release+relative_deadline`，恰好完成算 pass；
     - preemption 只在 graph node/flow iteration boundary，保存 continuation；
     - admission 检查 memory、context/inflight、peak bandwidth fraction 与 lower-priority blocking。
  5. `sim/engine/timeline.py`、`multicore.py` 作为 legacy adapter；不得用固定 70% overlap 作 scheduler oracle。

  **Must NOT do**:
  - 不使用 wall-clock、随机 set iteration 或 float event timestamps。
  - 不让零时间事件 livelock；每个 event 必须推进时间、消耗 work 或终止 job。

  **Parallelization**: Can parallel NO | Wave 3 | Blocks 14 | Blocked by 10,11,12

  **References**:
  - `sim/engine/timeline.py:55-196` — 当前单 core watermark overlap。
  - `sim/engine/multicore.py:1-190` — 当前 multicore/FIFO 与启发式 overlap。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:74-90,145-148,202-206,236-263` — queue/preemption/admission requirements。

  **Acceptance criteria**:
  - 独立手算：单 100B/10B-per-us transfer=10us；两个同权=20us；strict priority 高优先 10us、低优先 20us。
  - stable job IDs 保证输入顺序打乱后 canonical metrics 相同。
  - zero capacity、负 work、未知 resource、cyclic dependency、zero-time livelock typed fail。
  - FIFO/mailbox/priority/EDF/preemption/admission counterexamples 与 hand oracle 一致。

  **QA scenarios**:
  - **Happy**: kernel/resource/queue/policy hand-audited cases；Evidence `.omo/evidence/task-13-scheduler-kernel.json`。
  - **Failure**: invalid resource、queue overflow、nonpreemptible violation、livelock；Evidence `.omo/evidence/task-13-scheduler-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_scheduler_kernel.py sim/tests/test_scheduler_resources.py sim/tests/test_scheduler_queues.py sim/tests/test_scheduler_policies.py sim/tests/test_scheduler_admission.py -q -k "not negative" > .omo/evidence/task-13-scheduler-kernel.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_scheduler_kernel.py sim/tests/test_scheduler_resources.py sim/tests/test_scheduler_queues.py sim/tests/test_scheduler_policies.py sim/tests/test_scheduler_admission.py -q -k "negative or rejects or overflow or livelock" > .omo/evidence/task-13-scheduler-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(scheduler): add deterministic contention and policy kernel` | `sim/scheduler/__init__.py`, `events.py`, `kernel.py`, `resources.py`, `queues.py`, `policies.py`, `admission.py`, `sim/engine/timeline.py`, `sim/engine/multicore.py`, `sim/tests/test_scheduler_*.py`

- [ ] 14. 实现 P50/P99、deadline、高利用率、过载与恢复验收

  **What to do**:
  1. 新建 `sim/scheduler/metrics.py`：
     - arrival-to-start/complete、observation age；
     - nearest-rank P50/P99/max；
     - completed throughput、deadline/timeout miss、drop/replacement/admission reject/underflow；
     - per-resource busy_time/window_time utilization、peak queue、bytes、energy。
  2. 新建 `sim/scenarios/schema.py`、`compiler.py`：
     - standard acceptance 使用 periodic/explicit trace arrivals，不引入随机分布；
     - warm-up 为每 class 前 10 个 release；
     - measurement 为随后每 class 1000 个 release，并 drain 所有 admitted jobs，避免 right-censor；
     - deterministic seed 仍写入结果，供未来 stochastic adapter。
  3. 实现 stable/overload/recovery fixtures：
     - hand case service 4ms/period 10ms：P50=P99=4ms、util=0.4、0 miss；
     - service 6ms/period 5ms：FIFO backlog/latency/miss 增长；
     - admission 版本显式 reject 且 queue bounded；
     - mailbox 版本 pending depth≤1 且 replacements>0；
     - 90–95% stable offered load；
     - 110% overload 后切换 70% offered load并证明 backlog drains。
  4. 编译 SmolVLA async refill、π0 flow iterations、Helix multi-rate mailbox/partition、Physical-AI multi-job periods。
  5. 新增 `sim/scenario_runner.py` 和端到端 tests。

  **Must NOT do**:
  - 不把 200Hz action output 自动解释为 Helix 全网络 200Hz。
  - 不把 dropped/replaced jobs从分母移除以美化 P99/miss。

  **Parallelization**: Can parallel NO | Wave 3 | Blocks 15,17 | Blocked by 13

  **References**:
  - `sim/config/embodied-physical-ai-requirements.example.yaml:74-80` — SmolVLA refill。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:128-138` — π0 rates/deadlines。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:190-206` — Helix multi-rate。
  - `sim/config/embodied-physical-ai-requirements.example.yaml:236-263` — Physical-AI periods/queue/admission。

  **Acceptance criteria**:
  - hand cases exact match oracle；P50≤P99≤max。
  - 90–95% stable case queue bounded、drain 完成；110% case不能报告 stable。
  - overload recovery case在输入降到70%后 backlog 最终为0并报告 recovery time。
  - 四类具身/Physical-AI scenario 都产生终态完整 metrics，不存在 unresolved required field。

  **QA scenarios**:
  - **Happy**: stable、mailbox、priority、四类 workload 运行；Evidence `.omo/evidence/task-14-temporal-metrics.json`。
  - **Failure**: overload、deadline miss、nonpreemptible、invalid unresolved profile；Evidence `.omo/evidence/task-14-temporal-negative.json`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_scheduler_metrics.py sim/tests/test_scheduler_stability.py sim/tests/test_temporal_scenarios.py -q -k "stable or mailbox or priority or profile" > .omo/evidence/task-14-temporal-metrics.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_scheduler_stability.py sim/tests/test_temporal_scenarios.py -q -k "overload or deadline or unresolved or nonpreemptible" > .omo/evidence/task-14-temporal-negative.json 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(scenarios): add deterministic tail-latency and overload metrics` | `sim/scheduler/metrics.py`, `sim/scenarios/__init__.py`, `sim/scenarios/schema.py`, `sim/scenarios/compiler.py`, `sim/scenario_runner.py`, `sim/config/temporal_scenarios.yaml`, `sim/tests/test_scheduler_metrics.py`, `sim/tests/test_scheduler_stability.py`, `sim/tests/test_temporal_scenarios.py`

- [ ] 15. 实现 scenario-to-space 枚举、约束与 coverage manifest

  **What to do**:
  1. 新建 `sim/dse/space.py` 和 `sim/config/dse_axes.yaml`，从 scenario 选择/约束轴而不是硬编码 Cartesian loops。
  2. 正交轴至少包括 engine、array、frequency、precision、batch dimensions、LPDDR/HBM、3D capacity/bandwidth/PPA、image/horizon/flow、resident/inflight、partition、queue policy、nonpreemptible quantum。
  3. 新建 `sim/dse/manifest.py`：
     - 每轴 requested/generated/evaluated/successful/pruned/failed/missing 值与计数；
     - conditional exclusion 需要 reason code；
     - quick mode 记录缩减后的 requested set，而非假装覆盖 full；
     - invariant `generated=evaluated+pre_eval_pruned`、`evaluated=successful+post_eval_filtered+failed`。
  4. full default 必须实际包含 on-chip 3D points；ci-all-axes 用 pairwise/hand-selected 小集合触达每个轴值，避免组合爆炸。
  5. 更新 `dse_scenario.py`：preflight 与真实 space 使用同一 scenario model；删除只预检不驱动搜索的分叉。

  **Must NOT do**:
  - 不用 grep/label 判断某轴已覆盖。
  - 不用无 reason 的 pruning。

  **Parallelization**: Can parallel NO | Wave 4 | Blocks 16 | Blocked by 9,11,12,14

  **References**:
  - `sim/design_space_explorer.py:163-257` — 当前硬编码 space。
  - `sim/dse_scenario.py:83-203,269-456` — 当前 preflight/validation。
  - `sim/config/scenarios.yaml:5-105` — 当前场景但未驱动 DSE。
  - `sim/tests/test_dse_coverage.py:19-63` — 当前只覆盖 engine set。

  **Acceptance criteria**:
  - full/ci-all-axes manifest `missing=[]`，且 3D on-chip generated>0。
  - batch/profile/memory/frequency 的每个 required value 至少出现一个 generated point或结构化 exclusion。
  - 故意从 generator 删除某值，manifest test 必须失败。
  - space 输入顺序变化不改变 design-point ID set。

  **QA scenarios**:
  - **Happy**: 运行每个 scenario 的 ci-all-axes dry-run并聚合 coverage；Evidence `.omo/evidence/task-15-dse-coverage.json`。
  - **Failure**: omission/prune-without-reason/duplicate ID 注入必须失败；Evidence `.omo/evidence/task-15-dse-coverage-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_scenario_space.py sim/tests/test_dse_manifest.py -q -k "coverage or all_axes or deterministic" > .omo/evidence/task-15-dse-coverage.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_scenario_space.py sim/tests/test_dse_manifest.py -q -k "omission or duplicate or missing_reason or rejects" > .omo/evidence/task-15-dse-coverage-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(dse): generate design points from scenario contracts` | `sim/dse/__init__.py`, `sim/dse/space.py`, `sim/dse/manifest.py`, `sim/config/dse_axes.yaml`, `sim/dse_scenario.py`, `sim/tests/test_scenario_space.py`, `sim/tests/test_dse_manifest.py`

- [ ] 16. 实现多目标 Pareto、replay bundle 与兼容 CLI

  **What to do**:
  1. 新建 `sim/dse/runner.py`、`pareto.py`、`serialization.py`；`design_space_explorer.py` 变为 thin legacy/scenario CLI wrapper。
  2. Pareto 先应用 hard gates：complete graph、无 CPU fallback、capacity fit、quality gate placeholder resolved、P99/deadline、power/thermal、terminal completion；infeasible points保留 reason但不入 frontier。
  3. scenario 声明 objective direction；默认 maximize admitted completed throughput，minimize P99、miss rate、area、power、energy；tie 最终用 design_point_id。
  4. replay bundle 写入 normalized inputs、result、coverage、manifest、SHA256SUMS；timestamp 放非确定 metadata，canonical payload byte-identical。
  5. CLI：
     - 旧 flags/默认输出保持 Todo 1 snapshots；
     - 新增 `--scenario/--space/--seed/--result-schema {legacy,v2}/--replay`；
     - old `generate_configs/evaluate_config/find_pareto` import surface 通过 `sim/dse/legacy_adapter.py`。
  6. `eval_model_zoo.py`、`model_zoo_report.py` 无 caller 改动可运行。

  **Must NOT do**:
  - 不让 partial/non-authoritative point进入推荐。
  - 不覆盖已有 release/replay bundle。

  **Parallelization**: Can parallel NO | Wave 4 | Blocks 17,18 | Blocked by 15

  **References**:
  - `sim/design_space_explorer.py:305-327` — 当前仅 throughput/area Pareto。
  - `sim/design_space_explorer.py:541-792` — 当前 CLI、runner、serialization 混合。
  - `sim/eval_model_zoo.py:19-57`, `sim/model_zoo_report.py:79-212` — legacy consumers。
  - `sim/tests/test_dse_strict.py:14-67` — fail-closed/partial compatibility。

  **Acceptance criteria**:
  - 同 commit/input/seed 两次 canonical JSON 和 run ID byte-identical；修改任一 nested config 字段 digest 必变。
  - deliberate overload/infeasible/partial point不进入 Pareto。
  - replay 在 clean temporary checkout 重建相同 result/coverage digest。
  - 所有 legacy CLI snapshots、model-zoo callers 与 import surfaces 通过。

  **QA scenarios**:
  - **Happy**: 两次 compact-VLA ci-all-axes `cmp`，随后 replay；Evidence `.omo/evidence/task-16-dse-replay.json`。
  - **Failure**: partial/missing coverage/hash mismatch/overwrite attempt 全部拒绝推荐或 replay；Evidence `.omo/evidence/task-16-dse-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_scenario_pareto.py sim/tests/test_dse_reproducibility.py sim/tests/test_dse_legacy_compat.py -q > .omo/evidence/task-16-dse-replay.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_scenario_pareto.py sim/tests/test_dse_reproducibility.py -q -k "partial or missing or mismatch or overwrite or rejects" > .omo/evidence/task-16-dse-negative.txt 2>&1
    test $? -eq 0
    uv run python sim/design_space_explorer.py --scenario embodied_compact_vla --space ci-all-axes --seed 17 --result-schema v2 --output /tmp/arc-dse-a.json
    uv run python sim/design_space_explorer.py --scenario embodied_compact_vla --space ci-all-axes --seed 17 --result-schema v2 --output /tmp/arc-dse-b.json
    cmp /tmp/arc-dse-a.json /tmp/arc-dse-b.json
    uv run python sim/design_space_explorer.py --replay /tmp/arc-dse-a.json
    ```

  **Commit**: YES | `feat(dse): add reproducible multi-objective scenario search` | `sim/dse/runner.py`, `sim/dse/pareto.py`, `sim/dse/serialization.py`, `sim/dse/legacy_adapter.py`, `sim/design_space_explorer.py`, `sim/eval_model_zoo.py`, `sim/model_zoo_report.py`, `sim/tests/test_scenario_pareto.py`, `sim/tests/test_dse_reproducibility.py`, `sim/tests/test_dse_legacy_compat.py`

- [ ] 17. 建立参数 provenance、校准、误差预算与 trust gate

  **What to do**:
  1. 新建 `sim/calibration/schema.py`、`registry.py`、`evaluate.py` 与 `references/calibration/parameters.yaml`。
  2. trust 定义：
     - T0 assumption：只允许 exploratory sensitivity；
     - T1 analytic/published proxy：只允许范围内 feasibility/bound；
     - T2 reproduced RTL/reference + held-out validation：允许 measured domain 内相对决策并附 residual interval；
     - T3 signed-off representative RTL/silicon：允许范围内 numeric prediction，仍附 interval/limitations。
  3. 每个 decision-driving 参数保存 stable calibration_id、source URI/path/hash、tool/hardware、domain、training/held-out IDs、fit method、residual metrics、interval/status。
  4. GMMA `pipeline_scale=0.05`、TensorCore descriptor=5、3D macro assumptions 初始为 T0；不得因测试通过升级。
  5. 修改 `scripts/calibrate_mxu_model.py`：删除 raw RTL 缺失时用 analytic expected value 的 fallback；缺失/重复/checksum mismatch fail。
  6. DSE `exploratory` 可运行 T0/T1 并标记；`decision-grade` 要求所有 Pareto-driving 参数 T2+ 且点在 calibration domain 内。

  **Must NOT do**:
  - 不从 production model output 生成“measured” calibration。
  - 不虚构 confidence interval。

  **Parallelization**: Can parallel NO | Wave 5 | Blocks 18 | Blocked by 11,14,16

  **References**:
  - `sim/config/npu_config.yaml:80-82`, `sim/config/design_space.yaml:85-87` — GMMA assumption。
  - `sim/engine/tensor_core_engine.py:38-52` — descriptor cost。
  - `scripts/calibrate_mxu_model.py:148-152` — 当前 analytic fallback。
  - `reports/dse-engine-model-bugs-postfix-2026-07-27.md:75-121` — 当前 ranking/uncalibrated 不一致。

  **Acceptance criteria**:
  - missing/checksum-mismatch raw data fail-closed；valid train/held-out fixture产生 deterministic metrics。
  - T0 参数下 `decision-grade` 非零退出且列出 calibration IDs；`exploratory` 成功但所有受影响 point 标 exploratory。
  - held-out IDs 不参与 fitting；registry change 使 calibration/result digest 变化。
  - 超 calibration range 点不得 authoritative。

  **QA scenarios**:
  - **Happy**: checksum-bound tiny calibration train/held-out 流程；Evidence `.omo/evidence/task-17-calibration.json`。
  - **Failure**: 缺 raw、篡改 byte、T0 decision-grade、extrapolation；Evidence `.omo/evidence/task-17-calibration-negative.txt`。
  - **Commands**:
    ```bash
    uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q -k "valid or heldout or deterministic" > .omo/evidence/task-17-calibration.json 2>&1
    test $? -eq 0
    uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q -k "missing or checksum or t0 or extrapolat or rejects" > .omo/evidence/task-17-calibration-negative.txt 2>&1
    test $? -eq 0
    ```

  **Commit**: YES | `feat(calibration): bind model parameters to evidence and trust levels` | `sim/calibration/__init__.py`, `sim/calibration/schema.py`, `sim/calibration/registry.py`, `sim/calibration/evaluate.py`, `references/calibration/parameters.yaml`, `references/calibration/raw/README.md`, `references/calibration/raw/SHA256SUMS`, `scripts/calibrate_mxu_model.py`, `sim/tests/test_calibration_registry.py`, `sim/tests/test_calibration_evaluate.py`

- [ ] 18. 完成全矩阵验收、文档迁移和发布证据

  **What to do**:
  1. 新建 `sim/validation/scenario_matrix.py` 与 `scripts/release_gate.py`，聚合：
     - 8 engines；
     - M boundary `{1,2,3,15,16,17,63,64,65,128,1024}`；
     - frequency 800/1000/1200；
     - active sequences/token blocks；
     - LPDDR5/LPDDR5X/HBM2e/HBM3；
     - 3D full/partial/spill；
     - image/action/flow；
     - resident/inflight/periods；
     - 50/90/95/110% load；
     - invalid schema/op/hash/calibration。
  2. 新增 mutation tests，至少证明以下退化会被捕获：
     - frequency 强制1000；
     - GMMA floor移除；
     - descriptor忽略；
     - spill强制0；
     - unknown CV op返回0；
     - design-point positional association恢复；
     - partial point进入Pareto。
  3. 新增 `docs/model-trust-and-release.md`、`docs/publication-manifest.yaml`、`reports/README.md`；历史 bug/架构报告不改原文，只标 historical/superseded/exploratory/release_candidate。
  4. 更新 README、design methodology、arc-vs-func、engine guide、calibration docs，删除无证据的 “精确/<15%/完整证明” 表述。
  5. release profiles：
     - experimental：允许 T0/T1 但必须显式标签，不发布 promoted rankings；
     - decision-grade：所有 ranking parameter T2+、held-out interval、无 extrapolated winner、coverage missing=[]。
  6. 从 clean detached checkout 运行 locked QA，生成 content-addressed `artifacts/releases/<run-id>/manifest.json` 与 `SHA256SUMS`，不得覆盖旧 artifacts。
  7. 新增 `scripts/verify_evidence_ledger.py`：读取本计划 Todo/F1-F4、commit/evidence manifest schema，验证每项命令、exit code、digest、collected/passed/failed/skipped 和 artifact 存在性，输出结构化 JSON 并以非零退出表示缺口。
  8. 新增 `scripts/verify_scope.py`：以 baseline commit 和 publication manifest 为输入，机器检查改动路径、禁止依赖/目录、historical report 未改、ultraresearch 未 stage、current recommendation 均绑定 release manifest；输出结构化 JSON。
  9. 新增 `scripts/verify_model_integrity.py`：基于 AST、registry introspection 和 focused counterexample execution 检查 utilization clamp、未注册 calibration constant、重复 engine/operator registry、unknown-op zero-cycle fallback、直接读取 legacy bytes/cycle 和 diagnostics skip；输出结构化 JSON，任一违规非零退出。

  **Must NOT do**:
  - 不修改 dated historical report 内容来伪装当前有效。
  - 不在 dirty worktree 生成 decision-grade artifact。

  **Parallelization**: Can parallel NO | Wave 5 | Blocks F1-F4 | Blocked by 16,17

  **References**:
  - `sim/tests/test_engines.py:288-316,363-398` — 现有 self-consistency/baseline。
  - `.omo/evidence/task-10-verification.json` — 当前缺少 commit/config/lock hashes。
  - `docs/NPU系统级模拟器方案v0.1.md:1-20` — 精度 claim。
  - `docs/mxu-perf-calibration.md:35-42` — 误差证据冲突。
  - `README.md:313-358` — 当前发布/证据表述。

  **Acceptance criteria**:
  - `uv sync --frozen`、full pytest、blocking matrix、quick/full coverage、replay、publication audit 全部退出0，blocking skip/xfail=0。
  - experimental gate 只在所有 exploratory 标签/coverage/hash 完整时通过。
  - decision-grade 在 T0 参数存在时必须失败；只有提供 T2+ evidence 或移除这些 engine 的 ranking role 后才可通过。
  - release manifest hash 全部可校验，两次 clean replay result digest 相同。
  - publication manifest 不含无 run manifest 的 current recommendation。

  **QA scenarios**:
  - **Happy**: clean detached checkout experimental release + replay；Evidence `.omo/evidence/task-18-release-gate.json` 与 `artifacts/releases/<run-id>/`。
  - **Failure**: dirty tree、missing cell、T0 decision-grade、tampered artifact、mutation suite；Evidence `.omo/evidence/task-18-release-negative.json`。
  - **Commands**:
    ```bash
    uv sync --frozen
    uv run pytest sim/tests/test_scenario_acceptance.py sim/tests/test_evidence_ledger.py sim/tests/test_scope_gate.py -q > .omo/evidence/task-18-release-gate.json 2>&1
    test $? -eq 0
    uv run python scripts/release_gate.py --profile experimental
    uv run pytest sim/tests/test_scenario_acceptance.py -q -k "mutation or missing or tampered or dirty" > .omo/evidence/task-18-release-negative.json 2>&1
    test $? -eq 0
    test "$(uv run python scripts/release_gate.py --profile decision-grade >/tmp/decision-grade.txt 2>&1; echo $?)" -ne 0
    ```

  **Commit**: YES, two commits | `test(validation): add full arc-model acceptance matrix`; `docs(release): bind architecture claims to reproducible evidence` | `sim/validation/*`, `scripts/release_gate.py`, `scripts/verify_evidence_ledger.py`, `scripts/verify_model_integrity.py`, `scripts/verify_scope.py`, `sim/tests/test_scenario_acceptance.py`, `sim/tests/test_evidence_ledger.py`, `sim/tests/test_model_integrity_gate.py`, `sim/tests/test_scope_gate.py`, mutation tests, `docs/*`, `README.md`, `reports/README.md`, `artifacts/releases/<run-id>/*`

## Final verification wave (after ALL todos)

> Runs in parallel. ALL must return machine-readable `verdict=APPROVE` with exit code 0. Results are surfaced to the user after completion; user acknowledgement is not a technical acceptance dependency.

- [ ] F1. Plan compliance audit

  Verify every Todo 1–18 acceptance criterion against actual files/evidence/commits; reject missing evidence, skipped blocking tests, stale artifacts, untracked implementation files, or scope substitutions.

  ```bash
  uv run python scripts/verify_evidence_ledger.py \
    --plan .omo/plans/arc-model-scenario-driven-dse-development.md \
    --evidence-root .omo/evidence \
    --output .omo/evidence/final-f1-plan-compliance.json
  ```

  **APPROVE iff** exit code 0、Todos/F1-F4 schema 可解析、Todo 1–18 每项有匹配 commit/evidence、所有 recorded commands exit 0、blocking skipped/xfail=0、artifact/config/lock digests 校验通过。

- [ ] F2. Code quality and model-integrity review

  Review typed boundaries, module size/cohesion, error handling, unit consistency, all engine formulas, residency/PPA conservation, scheduler determinism, security/path handling and legacy adapters；运行：

  ```bash
  uv run ruff format --check . > .omo/evidence/final-f2-code-quality.txt 2>&1
  uv run ruff check . >> .omo/evidence/final-f2-code-quality.txt 2>&1
  uv run basedpyright >> .omo/evidence/final-f2-code-quality.txt 2>&1
  uv run pytest sim/tests/test_engine_physical_invariants.py \
    sim/tests/test_memory_residency.py sim/tests/test_memory_ppa.py \
    sim/tests/test_scheduler_stability.py sim/tests/test_operator_registry.py -q \
    >> .omo/evidence/final-f2-code-quality.txt 2>&1
  uv run python scripts/verify_model_integrity.py \
    --output .omo/evidence/final-f2-code-quality.json
  ```

  **APPROVE iff** 全部命令 exit 0、blocking skip/xfail=0，且 verifier 输出 `.omo/evidence/final-f2-code-quality.json` 中 `verdict=APPROVE`、`clamps/unregistered_constants/duplicate_registries/fail_open_paths/legacy_unit_reads/diagnostic_skips=[]`。

- [ ] F3. Real CLI/scenario/replay QA

  In a clean detached checkout run legacy npu_sim/DSE/model-zoo, every new workload family, ci-all-axes, full shard aggregation, experimental release and replay：

  ```bash
  uv run python scripts/release_gate.py \
    --profile experimental \
    --clean-checkout \
    --exercise-legacy \
    --exercise-all-workloads \
    --space ci-all-axes \
    --output .omo/evidence/final-f3-manual-qa.json
  ```

  **APPROVE iff** exit 0，结构化结果中 `legacy_failures=[]`、`workload_failures=[]`、`coverage.missing=[]`、`errors=0`、`replay_digest_match=true`、`experimental_gate=pass`，并实际产生可校验 release bundle。

- [ ] F4. Scope and evidence fidelity

  Audit diff against Must have/Must NOT have：

  ```bash
  uv run python scripts/verify_scope.py \
    --plan .omo/plans/arc-model-scenario-driven-dse-development.md \
    --baseline-commit "$(git merge-base HEAD origin/main)" \
    --publication-manifest docs/publication-manifest.yaml \
    --output .omo/evidence/final-f4-scope-fidelity.json
  ```

  **APPROVE iff** exit 0，且 JSON 中 `forbidden_dependencies=[]`、`ultraresearch_changes=[]`、`historical_report_changes=[]`、`unbound_current_claims=[]`、`out_of_scope_paths=[]`；PyTorch/ROS/Ramulator/DRAMSim 未进入 phase-one dependencies。

## Commit strategy

- 使用 conventional commits；每个 todo 的 implementation+test 为一个原子 commit，Todo 18 的 validation code 与 content-addressed documentation/release artifact 分成两个 commit。
- 不 amend、不 squash 已发布历史；修复审查问题使用独立 `fix(...)` commit。
- 每个 commit 只包含该 todo 的 Files 列表与其证据；`.omo/ultraresearch/20260723-vla-models/sources/` 永不 stage。
- 每个 wave 完成后创建 evidence manifest，记录 commit range、commands、exit codes、collected/passed/failed/skipped、config/workload/calibration/lock digests。
- 只有 Wave 0B 物理门通过后才允许合入后续 feature commits；只有 Wave 5 gate 完成后才允许发布架构推荐。

## Success criteria

- 当前已复现的频率无效、M=2→3 反向、OS/GMMA 超峰值、3D config=0、capacity-insensitive residency、3D PPA 二元化和 CV unknown-op 0-cycle 全部由独立测试先红后绿。
- 8 engine 在完整 shape/bandwidth/frequency matrix 上满足 MAC、DMA、utilization、diagnostic 与非法输入契约。
- batch 维度彼此正交且覆盖已批准的 edge/VLA/Physical-AI 范围。
- 3D on-chip DRAM 进入真实 DSE，full/partial/spill、capacity/bandwidth/area/power/energy 均可解释并满足守恒/单调性。
- 所有 named workload 都有独立 executable graph、provenance、footprint 和 temporal scenario。
- stable/90–95%/overload/recovery 的 P50/P99/deadline/queue/utilization 指标由确定性手算 oracle 锁定。
- scenario DSE coverage manifest 无 missing，结果可在 clean checkout byte-identical replay，partial/uncalibrated points不会被提升为 recommendation。
- legacy CLI/JSON/model-zoo 兼容测试全部通过；新 schema/result 有明确迁移与回滚路径。
- experimental release 可在完整标签与证据下通过；decision-grade 在所有 ranking-driving 参数达到 T2+ 之前保持诚实阻断。
- F1、F2、F3、F4 全部以结构化 `verdict=APPROVE` 和 exit code 0 完成；随后向用户报告结果，用户确认不作为技术 gate。
