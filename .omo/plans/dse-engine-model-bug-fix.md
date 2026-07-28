# dse-engine-model-bug-fix — Work Plan

## TL;DR (For humans)

> **Summary**: 在独立 Arc Model 仓库 `main` 上修复 8 个 DSE 引擎模型 bug（BUG-DSE-001 ~ 008），使 pytest 回归 8 红 → 8 绿，DSE 严格 fail-closed，证据 commit-bound。
>
> **Deliverables**:
> - 4 个引擎的公式级修复（SystolicEngine、OS-Systolic、TensorCore、GMMA），每条只有 1-3 行改动
> - 仓库根目录一键 `pytest` 通过，无需手动 PYTHONPATH
> - fail-closed DSE（异常 → 非零退出码，--allow-partial 保留有效结果）
> - post-fix 验证报告 + 更新后的架构文档
>
> **Effort**: Standard（11 todos，3 waves，预估 2-3 小时纯执行时间）
>
> **Risk**: Medium — 修复后 CLI 基准值和文档性能数字会变化，必须同步更新，否则造成新的版本漂移
>
> **架构决策**:
> - SystolicEngine **自己持有公式**，不引入共享 helper、不委托 MXUModel（和 BlockEngine/GMMAEngine 对等）
> - GMMA pipeline_scale=0.05 保留为类常量，config YAML 可覆盖，标注"未校准"
> - TensorCore descriptor 成本复用 `config: dma.descriptor_overhead_cycles=5`，不加新魔术常数
> - MXUModel 降级为 legacy 参考模型，npu_sim.py 不再直接实例化它

## Scope

### Must have

| # | 文件 | 改动 | BUG |
|---|------|------|-----|
| 1 | `systolic_engine.py:31` | decode `per_tile_compute` 改为 `self.H * (M + 1) + self.W` | DSE-002 |
| 2 | `systolic_engine.py:83` | prefill `pipeline_drain` 改为 `M if M < self.H else self.H` | DSE-003 |
| 3 | `os_systolic_engine.py:57` | `per_tile_compute` 加入 `self.H` | DSE-001 |
| 4 | `tensor_core_engine.py:71-84` | 每 wave 加入 `active_tcs * descriptor_overhead_cycles` | DSE-004 |
| 5 | `gmma_engine.py:59-61` | per-tile compute 启用 pipeline scaling：`max(1, ceil((H+M+W)*scale))` | DSE-005/006 |
| 6 | `design_space_explorer.py:602-610` | 异常捕获改为结构化：记录 `errors` 计数，非零退出 | — |
| 7 | `test_engines.py:347,366` | CLI 基准重新测量并更新 `pytest.approx` | DSE-007/008 |
| 8 | 仓库根 | 新建 `pytest.ini`，使 `pytest` 根目录可发现 `sim/tests` | — |
| 9 | `npu_config.yaml`, `design_space.yaml` | 新增 `gmma.pipeline_scale: 0.05` 字段 | — |

### Must NOT have

- 不修改 Golden Executor、Func Model、RTL、CaduceusCore 仓库任何文件
- 不修改 `.omo/ultraresearch/`、不覆盖用户未跟踪文件
- 不为通过排名断言而反复调整常数（TensorCore 的 descriptor 用已有 config 值 5，不 tune）
- 不引入无校准来源的 barrier/setup 常数
- 不允许 GMMA 的重叠后 DMA 时间替代物理 raw-DMA floor
- 不把历史 dated 报告中的旧数字原地重写
- 不把 quick DSE 的 exit 0、grep 命中或代理自述当作完成证据
- 不顺带重构 FSA、Input-Stationary、WMMA、PPA 模型
- 不引入共享 `systolic_timing.py` helper

## Verification strategy

> Zero human intervention — all verification is agent-executed.

| 维度 | 机制 |
|------|------|
| **测试框架** | `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q` 从仓库根目录执行 |
| **TDD 策略** | 每个 model fix 先让针对性测试稳定复现红，再改最小实现，最后跑跨引擎回归 |
| **QA per todo** | 每条 todo 包含 happy path + failure 两个场景，输出写入 `.omo/evidence/task-<N>-<slug>.<ext>` |
| **误导性成功防护** | pytest 证据同时记录 collected/passed/failed/exit code；DSE 证据同时记录 generated/evaluated/filtered/errors/valid |
| **Source authority** | 实施目标：当前仓库 `HEAD`（`main` 分支，fresh pull）。只读来源：`../CaduceusCore` 提交历史 |
| **Shipment check** | 最终 scope audit 证明所有测试、配置、报告均存在于 standalone Arc Model，不依赖来源仓 |

## Execution strategy

### Parallel execution waves

> Target 5-8 todos per wave. Wave 1 = no deps (5 lanes parallel). Wave 2 = after Wave 1 (5 lanes parallel). Wave 3 = integration-constrained (2 lanes). F1-F4 = final review-only.

```
Wave 1 (no deps, 5 lanes)
├── Todo 1: 建立测试基础设施（pytest.ini + conftest + 修正矛盾断言）
├── Todo 2: 暴露校准参数到 config（GMMA pipeline_scale + TensorCore descriptor）
├── Todo 3: DSE fail-closed 错误处理
├── Todo 4: 跨引擎结果契约测试
└── Todo 5: 修正 SystolicEngine decode + prefill 公式

Wave 2 (after W1, 5 lanes)
├── Todo 6: 修正 OS-Systolic K-reduction + Block-equivalent DMA
├── Todo 7: TensorCore descriptor fragmentation 建模
├── Todo 8: 恢复 GMMA pipeline scale + raw-DMA floor
├── Todo 9: 独立仓库验证入口（pytest.ini 终版 + all-engine DSE smoke）
└── Todo 10: 端到端 CLI 基准重测 + 完整回归

Wave 3 (after W2, integration-constrained)
└── Todo 11: 发布 post-fix 证据 + 更新架构文档

Final wave (after ALL todos, parallel review)
├── F1: Plan compliance audit
├── F2: Code quality review
├── F3: Real manual QA
└── F4: Scope fidelity
```

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|------|-----------|--------|---------------------|
| 1 | — | 2, 3, 4, 5, 9, 10 | — |
| 2 | — | 7, 8 | 1, 3, 4, 5 |
| 3 | — | 9, 10 | 1, 2, 4, 5 |
| 4 | — | 6, 7, 8, 10 | 1, 2, 3, 5 |
| 5 | — | 10 | 1, 2, 3, 4 |
| 6 | 1, 4 | 10 | 7, 8, 9 |
| 7 | 1, 2, 4 | 10 | 6, 8, 9 |
| 8 | 1, 2, 4 | 10 | 6, 7, 9 |
| 9 | 1, 3 | 10 | 6, 7, 8 |
| 10 | 1–9 | 11 | — |
| 11 | 10 | F1-F4 | — |

## Todos

> Implementation + Test = ONE todo. Never separate.
> Every todo has: exact file paths + line numbers in References, agent-executable Acceptance criteria, happy+failure QA with evidence paths, and a Commit line.

---

- [x] 1. 建立测试基础设施（pytest.ini + conftest + 修正矛盾断言）

  **What to do**:
  1. 在仓库根目录创建 `pytest.ini`，内容：
     ```ini
     [pytest]
     testpaths = sim/tests
     pythonpath = sim
     addopts = -p no:cacheprovider -q
     ```
  2. 创建 `sim/tests/conftest.py`，包含共享 fixture `engine_config()`：返回 `load_config()` 的深拷贝，array=64×64, freq=1000MHz, precision=INT4, DRAM=LPDDR5-64b(51.2GB/s)。
     **注意**：`sim/config/` 目录当前仅有 YAML 文件，无 `__init__.py`。需同步创建：
     - `sim/config/__init__.py`（空文件）
     - `sim/config/npu_config.py`，包含 `load_config()` 函数：读取 `sim/config/npu_config.yaml` 并返回 dict
  3. 修正 `sim/tests/test_engines.py:213` 矛盾注释（说 128×128 但实际构造 64×64）：改为 "OS-Systolic 64×64 decode contract"。
  4. 将 `sim/tests/test_engines.py:274` 的 Systolic/MXU 对比循环改为 `@pytest.mark.parametrize`，按 `(mode, M, op_name, K, N)` 展开，使单次失败不隐藏后续形状。
  5. 保留所有现有测试断言（包括当前红色值 `11.17`、`29.6`），**不在此 todo 修改任何引擎代码**。

  **References**:
  - `sim/config/npu_config.yaml:13-15` — array_height=64, array_width=64, frequency_mhz=1000
  - `sim/engine/mac_engine.py:127-157` — create_engine 工厂支持的所有引擎类型
  - `sim/tests/test_engines.py:15` — 现有 `_engine_config()` 辅助函数
  - `sim/tests/test_engines.py:213` — 矛盾注释："construct 128×128" 实际 64×64
  - `sim/tests/test_engines.py:274-286` — fail-fast 循环，需改为 parametrize
  - `sim/tests/test_engines.py:347,366` — 红色基线值（保留不修改）

  **Acceptance criteria**:
  - `git ls-files pytest.ini sim/config/__init__.py sim/config/npu_config.py sim/tests/conftest.py sim/tests/test_engines.py sim/tests/test_engine_instantiate.py` 打印所有 6 个路径
  - `cd <repo-root> && PYTHONDONTWRITEBYTECODE=1 python -m pytest --collect-only` 退出 0
  - 每个 Qwen2.5-3B GEMM（7 个 op_name）× M=1,2 展示为独立 node ID
  - 红色阶段 `pytest` 只失败在已记录的 8 个 BUG-DSE 节点上（无其他失败/报错）

  **QA scenarios**:
  - **Happy**: 收集所有节点，输出到 `.omo/evidence/task-1-collect.txt`
    ```bash
    cd /home/prj/zhengs/caduceuscore/npu_arc_model
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --collect-only -q > .omo/evidence/task-1-collect.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-1-collect.txt
    ```
  - **Failure 验证**: 运行红色套件，确认所有失败节点属于 BUG-DSE-001~008
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q --tb=line > .omo/evidence/task-1-red.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-1-red.txt
    ```

  **Commit**: `test(infra): establish standalone pytest entrypoint and contract fixtures`
  > Files: `pytest.ini`, `sim/config/__init__.py`, `sim/config/npu_config.py`, `sim/tests/conftest.py`, `sim/tests/test_engines.py`

---

- [x] 2. 暴露校准参数到 config（GMMA pipeline_scale + TensorCore descriptor）

  **What to do**:
  1. 在 `sim/config/npu_config.yaml` 的 `dma:` 块结束位置（`arbitration: round_robin` 行之后、`memory:` 行之前，约第 79 行）新增**顶级** `gmma:` 块（注意缩进：顶层 0 空格缩进）：
     ```yaml
     gmma:
       pipeline_scale: 0.05  # uncalibrated — H100 GMMA architectural assumption, not signed off
     ```
     验证：`python -c "from sim.config.npu_config import load_config; assert load_config().get('gmma',{}).get('pipeline_scale')==0.05"` 必须通过。
  2. 在 `sim/config/design_space.yaml` 的 `interconnect:` 块结束位置（`port_bandwidth_gbps: 500` 行之后、`area_model:` 行之前，约第 84 行）新增**顶级** `gmma:` 块（同样顶层缩进）：
     ```yaml
     gmma:
       pipeline_scale: 0.05  # can be swept for sensitivity; default 0.05 is uncalibrated
     ```
     **注意**：两个 YAML 文件的 `gmma.pipeline_scale` 默认值必须一致（均为 0.05）。若一方修改另一方未同步，会导致 `npu_sim.py` 与 DSE 走不同默认值。建议在 post-fix 报告中标注此同步约束。
  3. 在 `sim/engine/gmma_engine.py:50` 确认 `GMMA_PIPELINE_SCALE = 0.05` 保留为类常量，在 `__init__` 或 `_parse_config` 中增加从 config 读取并覆盖的逻辑：
     ```python
     gmma_cfg = config.get("gmma", {})
     self.pipeline_scale = gmma_cfg.get("pipeline_scale", GMMA_PIPELINE_SCALE)
     ```
  4. 在 `sim/engine/tensor_core_engine.py` 中确认 `descriptor_overhead_cycles` 已从 config 的 `dma.descriptor_overhead_cycles` 读取（当前在 `mac_engine.py:48` 的 `_parse_config` 未解析此字段，需在 TensorCoreEngine 的 `__init__` 中显式读取）。
  5. 为两个参数各加边界校验：`pipeline_scale` 必须 `0 < scale <= 1`，`descriptor_overhead_cycles` 必须 `>= 0`（整数）。

  **References**:
  - `sim/config/npu_config.yaml:68-78` — 现有 DMA 配置块末尾（插入点）
  - `sim/config/design_space.yaml:71-83` — 现有 interconnect 块末尾（插入点）
  - `sim/engine/gmma_engine.py:44-53` — 类常量定义区域
  - `sim/engine/gmma_engine.py:34-42` — `_parse_config` 方法（需在此读取 gmma 配置）
  - `sim/engine/tensor_core_engine.py:31-42` — 类常量 + `__init__` 区域（需在此读取 descriptor）
  - `sim/config/npu_config.yaml:70` — `dma.descriptor_overhead_cycles: 5`（已有值，只需在 TensorCoreEngine 中消费）

  **Acceptance criteria**:
  - 默认 config 解析出 `pipeline_scale=0.05`，`descriptor_overhead_cycles=5`
  - 通过 `npu_config.yaml` override 可改变解析值
  - `pipeline_scale=0`、`1.01` 或 `descriptor=-1` 时构造抛 `ValueError`（含字段名）

  **QA scenarios**:
  - **Happy**: 写聚焦测试 `sim/tests/test_calibration_config.py`
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_calibration_config.py -q > .omo/evidence/task-2-calibration.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-2-calibration.txt
    ```
  - **Failure**: 参数化边界值（scale=0, scale=1.01, descriptor=-1）均抛 ValueError
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_calibration_config.py -q -k "invalid" > .omo/evidence/task-2-invalid.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-2-invalid.txt
    ```

  **Commit**: `feat(config): expose gmma pipeline_scale and tensor_core descriptor knobs`
  > Files: `sim/config/npu_config.yaml`, `sim/config/design_space.yaml`, `sim/engine/gmma_engine.py`, `sim/engine/tensor_core_engine.py`, `sim/tests/test_calibration_config.py`

---

- [x] 3. DSE fail-closed 错误处理

  **What to do**:
  1. 替换 `sim/design_space_explorer.py:602-610` 的 `try/except/pass` 循环为结构化统计：
     ```python
     results: List[PPA] = []
     generated = len(configs)
     evaluated = 0
     filtered_by_area = 0
     errors = 0
     error_details: List[str] = []

     for cfg in configs:
         try:
             ppa = evaluate_config(cfg, area_model, power_model)
             if ppa.area_mm2 <= 200:
                 results.append(ppa)
                 evaluated += 1
             else:
                 filtered_by_area += 1
         except Exception as e:
             errors += 1
             engine_type = cfg.get("mac_engine", {}).get("type", "unknown")
             dims = f"H={cfg.get('mac_engine',{}).get('array_height','?')}xW={cfg.get('mac_engine',{}).get('array_width','?')}"
             mem_mode = "on_chip" if cfg.get("on_chip_memory", {}).get("capacity_gb", 0) > 0 else "external"
             error_details.append(f"[{engine_type}] {dims} {mem_mode}: {e}")
             print(f"ERROR [{engine_type}] {dims} {mem_mode}: {e}", file=sys.stderr)
     ```
  2. 在 `sim/design_space_explorer.py:544` CLI parser 中新增 `--allow-partial` flag（`store_true`）。
  3. 默认模式（无 `--allow-partial`）：`errors > 0` 时 `sys.exit(1)`。
  4. `--allow-partial` 模式：仍打印每个错误到 stderr，输出 JSON 的 metadata 包含 `errors` 计数，`sys.exit(0)`。
  5. `evaluated == 0`（无有效结果）时永远 `sys.exit(1)`，无论 `--allow-partial`。
  6. 在 `sim/design_space_explorer.py:686-730` 输出 JSON 的 metadata 中加入 `generated`、`evaluated`、`filtered_by_area`、`errors`、`error_details` 字段。

  **References**:
  - `sim/design_space_explorer.py:541-557` — CLI parser（插入 `--allow-partial`）
  - `sim/design_space_explorer.py:602-610` — 当前静默 try/except（替换为目标代码）
  - `sim/design_space_explorer.py:163-175` — `generate_configs` 函数签名（理解 config 结构）
  - `sim/design_space_explorer.py:686-730` — JSON 输出格式（metadata 插入点）
  - `sim/engine/mac_engine.py:129-130` — config 中 engine type 的读取路径

  **Acceptance criteria**:
  - 注入一个 evaluator 异常：默认模式 exit 1，stderr 包含 `ERROR [engine_type]` + 维度
  - 同样注入 + `--allow-partial`：保留有效结果，metadata 中 `errors=1`，exit 0
  - 正常 quick DSE：exit 0，metadata 中 `errors=0`
  - 无有效结果（全部异常或全被面积过滤）：exit 1

  **QA scenarios**:
  - **Happy**: `python sim/design_space_explorer.py --quick` → exit 0，errors=0
    ```bash
    cd /home/prj/zhengs/caduceuscore/npu_arc_model
    python sim/design_space_explorer.py --quick --output .omo/evidence/task-3-happy.json 2>.omo/evidence/task-3-happy-stderr.txt
    echo "EXIT: $?" > .omo/evidence/task-3-happy-exit.txt
    ```
  - **Failure**: 写 pytest 测试 `sim/tests/test_dse_strict.py`，monkeypatch `evaluate_config` 抛异常
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_dse_strict.py -q > .omo/evidence/task-3-failure.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-3-failure.txt
    ```

  **Commit**: `fix(dse): fail closed on configuration errors with structured accounting`
  > Files: `sim/design_space_explorer.py`, `sim/tests/test_dse_strict.py`

---

- [x] 4. 跨引擎结果契约测试

  **What to do**:
  1. 新建 `sim/tests/test_engine_result_contract.py`，测试所有 `create_engine` 返回的引擎类型（8 种）。
  2. 每个引擎的 `estimate(M=1, K=11008, N=4096)` 结果必须满足：
     - `total_cycles > 0`，`compute_cycles > 0`，`dma_cycles >= 0`
     - `utilization` 在 `(0, 1]` 区间
     - `ops > 0`，`ops >= M * K * N`
     - `bottleneck` 为 `"compute"` 或 `"dma"`
     - `weight_bytes > 0`（所有权重从 DRAM 加载时）
     - `details` dict 非空
  3. 对 OS-Systolic：要求 `details` 包含 `raw_dma_cycles`、`k_reduction_cycles`、`total_compute_cycles`、`bottleneck_reason`（由 Todo 6 输出）。
  4. 对 TensorCore：要求 `details` 包含 `active_tcs`、`num_waves`、`per_wave_payload_cycles`、`descriptor_cycles_per_wave`、`total_descriptor_cycles`（由 Todo 7 输出）。
  5. 对 GMMA：要求 `details` 包含 `raw_dma_cycles`、`tma_hidden_dma`、`tma_exposed_dma`、`per_tile_compute`、`pipeline_scale`（由 Todo 8 输出），且 `total_cycles >= details["raw_dma_cycles"]`（raw-DMA floor 约束，使用含 weight_dram_eff 的完整 raw-DMA = `total_weight_bytes / (eff_bw * weight_dram_eff) + act_bytes / eff_bw`）。
  6. 本 todo **不修改任何引擎代码**，新增的诊断字段缺失时测试应为红色（等待后续 todo 修复）。

  **References**:
  - `sim/engine/mac_engine.py:15-35` — `EngineResult` dataclass 完整 schema
  - `sim/engine/mac_engine.py:127-157` — `create_engine` 工厂（引擎类型全集）
  - `sim/engine/os_systolic_engine.py:34-89` — 当前 estimate 的 details 输出
  - `sim/engine/gmma_engine.py:63-108` — 当前 estimate 的 details（缺少 raw_dma_cycles）
  - `sim/engine/tensor_core_engine.py:51-84` — 当前 estimate 的 details（缺少 descriptor 字段）
  - `sim/tests/test_engine_instantiate.py:36-43` — 现有实例化测试（只测工厂）

  **Acceptance criteria**:
  - 测试覆盖全部 8 种引擎的 `create_engine` 返回类型
  - 每个引擎结果通过基本字段验证
  - OS-Systolic、TensorCore、GMMA 缺少诊断字段时为红色（预期，待 Todo 6/7/8 修复）
  - 其他 5 个引擎的契约测试为绿色

  **QA scenarios**:
  - **Happy**: 契约测试套件（预期部分红）
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engine_result_contract.py -q > .omo/evidence/task-4-contract.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-4-contract.txt
    ```
  - **Failure**: 构造畸形 `EngineResult` fixture（负周期、空 details、非法 bottleneck）→ 全部断言拒绝
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engine_result_contract.py -q -k "invalid" > .omo/evidence/task-4-invalid.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-4-invalid.txt
    ```

  **Commit**: `test(engine): define inspectable result contracts for all engine types`
  > Files: `sim/tests/test_engine_result_contract.py`

---

- [x] 5. 修正 SystolicEngine decode + prefill 公式

  **What to do**:
  1. 修改 `sim/engine/systolic_engine.py:31`：
     ```python
     # BEFORE:
     per_tile_compute = pipeline_fill + pipeline_drain  # pipeline_drain = M + self.H (line 30)
     # AFTER:
     per_tile_compute = self.H * (M + 1) + self.W
     ```
     同时删除第 29-30 行不再需要的 `pipeline_fill` 和 `pipeline_drain` 中间变量。
  2. 修改 `sim/engine/systolic_engine.py:82-84`：
     ```python
     # BEFORE:
     pipeline_drain = self.H + self.H            # line 83: always 2H (WRONG)
     per_m_tile_compute = pipeline_fill + pipeline_drain  # line 84

     # AFTER:
     if M_tiles == 1 and M < self.H:
         per_m_tile_compute = pipeline_fill + M   # partial tile
     else:
         per_m_tile_compute = pipeline_fill + self.H  # full tile
     ```
     `pipeline_fill = self.H + self.W`（line 82）保留不变。
  3. `sim/engine/systolic_engine.py:137-167` 的 `estimate_weight_cache_pair`：检查 `drain = M + self.H`（line 151）。MXUModel 自己的 pair 路径（`sim/models/mxu.py:238-243`）同样使用 `drain = M + self.H`，未使用 `H*(M+1)+W`。因此 pair 路径的 `drain` 公式**保持不变**。仅确认 `per_matm_drain = M + self.W` 和 `dual_compute = 2 * per_matm_drain + 1`（line 147-148）与 MXUModel pair 一致。
  4. 删除 `sim/engine/systolic_engine.py:21` 和 `:70` 的 "byte-identical to MXUModel" 注释，替换为 "self-contained systolic timing — validated against MXUModel reference at sim/models/mxu.py:91,163-170"。
  5. 不修改 `sim/models/mxu.py` 任何行。
  6. **Edge case guard**：`estimate()` 内部 `utilization = ideal_cycles / total_compute_cycles` 当 `M=0` 时 `ideal_cycles=0`，`utilization=0`，不会除零（因为分母 `total_compute_cycles` 为 DMA cycles，非零）。但应在函数入口处对 `M <= 0` 加显式 `raise ValueError` 以防御未来调用。

  **References**:
  - `sim/engine/systolic_engine.py:20-40` — decode 路径（BEFORE 公式）
  - `sim/models/mxu.py:87-91` — MXUModel 正确的 decode 公式（参考，不修改）
  - `sim/engine/systolic_engine.py:69-97` — prefill 路径（BEFORE 公式）
  - `sim/models/mxu.py:156-170` — MXUModel 正确的 prefill 条件分支（参考，不修改）
  - `sim/engine/systolic_engine.py:137-167` — weight-cache-pair 中的重复公式
  - `sim/engine/systolic_engine.py:21,70` — 误导性注释（删除 + 替换）

  **Acceptance criteria**:
  - `test_systolic_vs_mxumodel_decode`（test_engines.py:274）对 M=1,2 × 7 Qwen GEMM 全部通过（精确值相等）
  - `test_systolic_vs_mxumodel_prefill`（test_engines.py:289）对 M=128 × 7 Qwen GEMM 全部通过
  - SystolicEngine 内部无重复公式（decode、prefill、pair 路径公式一致）
  - 无 "byte-identical to MXUModel" 字符串残留

  **QA scenarios**:
  - **Happy**: 聚焦 parity 测试
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engines.py -q -k "systolic_vs_mxumodel" > .omo/evidence/task-5-parity.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-5-parity.txt
    ```
  - **Failure 验证**: 用不同 H/W 配置验证公式仍正确
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engines.py -q -k "systolic_vs_mxumodel" > .omo/evidence/task-5-varied-array.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-5-varied-array.txt
    ```

  **Commit**: `fix(systolic): correct decode and prefill per-tile-compute formulas`
  > Files: `sim/engine/systolic_engine.py`

---

- [ ] 6. 修正 OS-Systolic K-reduction + Block-equivalent DMA 核算

  **What to do**:
  1. 修改 `sim/engine/os_systolic_engine.py:57`：
     ```python
     # BEFORE:
     self.per_tile_compute = BROADCAST_SYNC_CYCLES + _accumulate_cycles(self.w_bits, self.a_bits)
     # AFTER:
     self.per_tile_compute = self.H + BROADCAST_SYNC_CYCLES + _accumulate_cycles(self.w_bits, self.a_bits)
     ```
     （OS-Systolic 每个 K 维度 tile 仍需做 H 次 reduction，当前完全遗漏了）
  2. 修改 `sim/engine/os_systolic_engine.py:51-54` 的 DMA 核算：不要用原始的 `(tile_weight + tile_act) / eff_bw`，改用和 BlockEngine（`block_engine.py:99-136`）完全相同的聚合外部 DRAM 核算逻辑：
     ```python
     # 完整公式（复刻 block_engine.py:115-136）：
     total_weight_bytes = K * N * self.w_bits // 8
     weight_dram_eff = self._dram_eff_for_bytes(total_weight_bytes)  # 注意：函数内部自行做 bytes→MB 转换，不要额外除 1024*1024
     weight_load_cycles = total_weight_bytes / (self.eff_bw * weight_dram_eff)
     act_bytes_per_token = K * self.a_bits // 8
     act_load_cycles = M * act_bytes_per_token / self.eff_bw
     total_dma_cycles = M_tiles * weight_load_cycles + act_load_cycles
     ```
     （M_tiles = ceil(M / self.H)，对 M=1 为 1。per_tile_dma 按流水线模型从 total_dma_cycles 反算。）
  3. 在 `estimate()` 的 `details` dict 中新增字段：`k_reduction_cycles`（值 = `self.H`）、`raw_dma_cycles`、`total_compute_cycles`、`bottleneck_reason`。
  4. 对 `estimate_weight_cache_pair()`（line 92-148）应用同样的 compute + DMA 修正。
  5. 不在 timing engine 中添加面积惩罚（面积模型在 `sim/engine/ppa_model.py:76` 独立处理）。

  **References**:
  - `sim/engine/os_systolic_engine.py:47-67` — 当前 single estimate 路径
  - `sim/engine/os_systolic_engine.py:57` — 缺失 `self.H` 的一行
  - `sim/engine/os_systolic_engine.py:51-54` — 当前原始 DMA 路径
  - `sim/engine/block_engine.py:99-136` — BlockEngine 外部 DRAM 聚合核算（参考实现）
  - `sim/engine/block_engine.py:23` — `BROADCAST_SYNC_CYCLES = 2`（OS 已正确 import）
  - `sim/engine/os_systolic_engine.py:92-148` — weight-cache-pair 路径（也需修正）

  **Acceptance criteria**:
  - 64×64 M=1 FFN_down：OS `bottleneck == "dma"` 且 OS `total_cycles >= Block total_cycles`（即有 `os_tok_s <= block_tok_s` 满足 `test_os_systolic_decode` 原始断言）
  - OS 和 Block 的 total_cycles 偏差 ≤ 10%
  - OS `details["k_reduction_cycles"] == self.H`
  - OS single 和 pair 路径报告的 `transferred_bytes` 与激活复用行为一致
  - pair 路径有限且暴露 activation-reuse 节省，无凭空面积惩罚

  **QA scenarios**:
  - **Happy**: OS/Block same-band 测试
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engines.py -q -k "os_systolic" > .omo/evidence/task-6-os.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-6-os.txt
    ```
  - **Failure**: 带宽 sweep（低/高 BW）验证瓶颈转换时不违反 raw bytes
    ```bash
    # 由测试代码中的 parametrize 覆盖，手动记录 JSON 输出
    cp /tmp/os_bw_sweep.json .omo/evidence/task-6-bw-sweep.json
    ```

  **Commit**: `fix(os-systolic): account for K-reduction depth and physical DMA aggregation`
  > Files: `sim/engine/os_systolic_engine.py`

---

- [ ] 7. TensorCore descriptor fragmentation 建模

  **What to do**:
  1. 在 `sim/engine/tensor_core_engine.py` 的 `__init__` 中解析 `DMA_DESCRIPTOR_CYCLES`（从 config 的 `dma.descriptor_overhead_cycles` 读取，默认 5）：
     ```python
     dma_cfg = config.get("dma", {})
     self.descriptor_overhead = dma_cfg.get("descriptor_overhead_cycles", 5)
     ```
  2. 修改 `sim/engine/tensor_core_engine.py:71-84` 的 per-wave DMA 核算：
     ```python
     # 每 wave 的 payload DMA（不变）
     per_wave_dma_cycles = per_wave_payload / eff_bw

     # 新增：descriptor 开销 = 活跃 TC 数 × 每个 descriptor 的周期数
     active_tcs = min(num_tcs, total_invocations - (wave_idx * num_tcs))  # 最后一波可能不满
     descriptor_cycles = active_tcs * self.descriptor_overhead

     # 每 wave 总 DMA = payload + descriptor
     per_wave_dma = per_wave_dma_cycles + descriptor_cycles
     ```
  3. 在 `details` dict 中新增字段：`active_tcs`、`num_waves`、`per_wave_payload_cycles`、`descriptor_cycles_per_wave`、`total_descriptor_cycles`。
  4. 对 `estimate_weight_cache_pair()`（line 117-189）应用相同修正。

  **References**:
  - `sim/engine/tensor_core_engine.py:28-42` — `__init__` 构造参数区域
  - `sim/engine/tensor_core_engine.py:54-84` — wave 循环和 per-wave DMA 核算
  - `sim/engine/tensor_core_engine.py:68-70` — `num_tcs` 和 `invocations_per_tc` 计算
  - `sim/engine/tensor_core_engine.py:71-73` — 当前 per-wave DMA（缺 descriptor）
  - `sim/config/npu_config.yaml:70` — `dma.descriptor_overhead_cycles: 5`
  - `sim/config/design_space.yaml:42-45` — 同名参数

  **Acceptance criteria**:
  - 默认 64×64 FFN_down：TensorCore total_cycles > BlockEngine total_cycles（即 TC 比 Block 慢）
  - 设置 `descriptor_overhead_cycles=0` 时恢复 pre-fix payload-only 行为（误差在 rounding 内）
  - 最后一波部分 wave 只对活跃 TC 收费（不是全部 `num_tcs`）
  - `details` 包含 `active_tcs`、`num_waves`、`total_descriptor_cycles`

  **QA scenarios**:
  - **Happy**: 默认 vs descriptor=0 对比
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engines.py -q -k "tensor_core" > .omo/evidence/task-7-tc.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-7-tc.txt
    ```
  - **Failure**: descriptor < 0 构造时拒绝
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_calibration_config.py -q -k "descriptor" > .omo/evidence/task-7-invalid.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-7-invalid.txt
    ```

  **Commit**: `fix(tensor-core): model per-wave descriptor fragmentation overhead`
  > Files: `sim/engine/tensor_core_engine.py`

---

- [ ] 8. 恢复 GMMA pipeline scale + raw-DMA floor

  **What to do**:
  1. 修改 `sim/engine/gmma_engine.py:59-61`：
     ```python
     # BEFORE (line 61):
     per_tile_compute = self.H + M + self.W

     # AFTER:
     per_tile_compute = max(1, math.ceil((self.H + M + self.W) * self.pipeline_scale))
     ```
     其中 `self.pipeline_scale` 从 Todo 2 的 config 读取逻辑获取。
  2. 修改 `sim/engine/gmma_engine.py:82-89` 的 DMA/overlap 核算：
     ```python
     # 当前代码 (line 82-89)：
     #   total_dma = weight_dma_cycles + act_dma_cycles
     #   tma_dma = total_dma * (1 - TMA_OVERLAP)
     #   total_cycles = max(total_compute, tma_dma)   ← TMA overlap 绕过了物理上限
     #
     # 修复后：
     #   total_dma = weight_dma_cycles + act_dma_cycles       # 不变：含 weight_dram_eff
     #   tma_hidden = total_dma * TMA_OVERLAP                 # 新增：TMA 隐藏部分
     #   tma_exposed = total_dma * (1 - TMA_OVERLAP)          # 新增：TMA 暴露部分
     #   total_cycles = max(total_compute, total_dma)         # floor = 完整 DMA（无 overlap 削减）
     ```
     **关键**：`total_dma` 已经包含 `weight_dram_eff`（在 `weight_dma_cycles` 的计算中），raw-DMA floor 不额外除以 `eff_bw`。`total_compute` 按步骤 1 的 pipeline_scale 缩放后大幅降低，因此 `max(total_compute, total_dma)` 返回 `total_dma`，GMMA 回到 DMA-bound。
  3. 在 `details` dict 中新增字段：`raw_dma_cycles`、`tma_hidden_dma`、`tma_exposed_dma`、`per_tile_compute`、`pipeline_scale`。
  4. 对 `estimate_weight_cache_pair()`（line 116-174）应用相同修正。
  5. 验证带宽单调性：LPDDR5 → HBM2e 吞吐应有提升（因为 raw-DMA floor 随带宽降低）。

  **References**:
  - `sim/engine/gmma_engine.py:39-42` — TMA_OVERLAP 注释（现有意图）
  - `sim/engine/gmma_engine.py:50` — `GMMA_PIPELINE_SCALE = 0.05`（本次启用）
  - `sim/engine/gmma_engine.py:59-61` — 当前未 scal 的 per_tile_compute
  - `sim/engine/gmma_engine.py:82-89` — 当前 total_cycles = max(total_compute, tma_dma)（允许 TMA 低于物理上限）
  - `sim/engine/gmma_engine.py:108-114` — single estimate details 输出
  - `sim/engine/gmma_engine.py:116-174` — weight-cache-pair 路径
  - `docs/NPU_Engines_Architecture_Guide.md:217` — LPDDR/GMMA contract 参考

  **Acceptance criteria**:
  - LPDDR5 FFN_down：GMMA 回到 `bottleneck="dma"`，total_cycles 有限
  - HBM2e 吞吐 > 2× LPDDR5（bandwidth monotonicity）
  - 对每种带宽：`total_cycles >= ceil((weight_bytes + act_bytes) / eff_bw)`（raw-DMA floor）
  - single 和 pair 的 details 都包含 `raw_dma_cycles`、`tma_hidden_dma`、`tma_exposed_dma`、`pipeline_scale`

  **QA scenarios**:
  - **Happy**: 带宽 sweep（LPDDR5-64b, LPDDR5-128b, HBM2e）
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engines.py -q -k "gmma" > .omo/evidence/task-8-gmma.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-8-gmma.txt
    ```
  - **Failure**: 极端 overlap 值不压低 total 到 raw DMA 以下
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_engine_result_contract.py -q -k "gmma_floor" > .omo/evidence/task-8-floor.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-8-floor.txt
    ```

  **Commit**: `fix(gmma): enable pipeline scaling and enforce physical raw-dma floor`
  > Files: `sim/engine/gmma_engine.py`

---

- [ ] 9. 独立仓库验证入口（pytest.ini 终版 + all-engine DSE smoke）

  **What to do**:
  1. 确认 `pytest.ini`（Todo 1 创建）正确配置，使根目录 `python -m pytest` 发现所有测试。
  2. 新增 `sim/tests/test_dse_coverage.py`：
     - 测试 DSE full mode 引擎列表（`generate_configs(quick=False)` 的 engine types）与 `create_engine` 工厂支持的类型完全一致
     - 测试 DSE quick mode 引擎列表为 `["systolic", "block", "gmma"]`
     - 测试根目录 `pytest.ini` 存在且包含 `testpaths = sim/tests`
  3. 新增 `sim/tests/test_standalone_assets.py`：
     - 测试所有必需的验证资产文件在独立仓库中存在：`pytest.ini`、`sim/tests/test_engines.py`、`sim/tests/test_engine_instantiate.py`、`sim/tests/test_engine_result_contract.py`、`sim/tests/test_dse_strict.py`、`sim/tests/test_calibration_config.py`、`sim/tests/test_dse_coverage.py`
     - 删除任一资产后测试失败
  4. 在 `README.md:251` 的 quick-DSE 命令旁新增 all-engine smoke 命令文档。

  **References**:
  - `sim/design_space_explorer.py:163-175` — `generate_configs` 中的引擎列表
  - `sim/design_space_explorer.py:170-175` — quick vs full 引擎列表差异
  - `sim/engine/mac_engine.py:127-157` — 工厂支持的引擎类型全集
  - `pytest.ini`（Todo 1 创建）— 验证其存在和内容
  - `README.md:244-260` — 快速开始命令文档区域

  **Acceptance criteria**:
  - 根目录 `PYTHONDONTWRITEBYTECODE=1 python -m pytest --collect-only -q` 退出 0
  - DSE/full-engine 集合对比为精确匹配
  - 所有必需资产文件被 `test_standalone_assets` 追踪
  - 删除任一必需文件后 `test_standalone_assets` 失败

  **QA scenarios**:
  - **Happy**: 根级别 collection + all-engine smoke
    ```bash
    cd /home/prj/zhengs/caduceuscore/npu_arc_model
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --collect-only -q > .omo/evidence/task-9-collect.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-9-collect.txt
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_dse_coverage.py sim/tests/test_standalone_assets.py -q > .omo/evidence/task-9-validation.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-9-validation.txt
    ```
  - **Failure**: pytest tmp-path fixture 模拟缺失资产
    ```bash
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider sim/tests/test_standalone_assets.py -q > .omo/evidence/task-9-gap.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-9-gap.txt
    ```

  **Commit**: `test(dse): enforce standalone all-engine coverage and asset validation`
  > Files: `sim/tests/test_dse_coverage.py`, `sim/tests/test_standalone_assets.py`, `pytest.ini`（如需要调整）, `README.md`

---

- [ ] 10. 端到端 CLI 基准重测 + 完整回归

  **What to do**:
  1. 运行 `npu_sim.py --json`（Block 引擎）和 `--engine systolic --json`（Systolic 引擎），记录新的 `tok_per_s` 值。
  2. 修改 `sim/tests/test_engines.py:347`：
     ```python
     # BEFORE:
     assert report["decode"]["tok_per_s"] == pytest.approx(11.17, rel=0.01)
     # AFTER:
     assert report["decode"]["tok_per_s"] == pytest.approx(<actual_value>, rel=0.01)
     ```
     其中 `<actual_value>` 为步骤 1 实际测量的 Systolic tok/s。
  3. 修改 `sim/tests/test_engines.py:366`：同样操作，Block 引擎的新实测值。
  4. 运行完整 pytest 套件：`PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q`，确认 exit 0，0 failures。
  5. 运行 strict quick DSE：`python sim/design_space_explorer.py --quick`，确认 exit 0，`errors=0`。
  6. 运行 full DSE：`python sim/design_space_explorer.py`，记录 metadata 统计（generated/evaluated/filtered/errors/valid）。
  7. 对所有 7 个引擎（不含 input_stationary 因为 DSE best-per-engine 也 skip 它）跑相同的 FFN_down shape，记录 total/compute/raw-DMA/bottleneck 到 JSON。

  **References**:
  - `sim/tests/test_engines.py:347` — Systolic CLI 基准（BEFORE: 11.17）
  - `sim/tests/test_engines.py:366` — Block CLI 基准（BEFORE: 29.6）
  - `sim/npu_sim.py:475-503` — CLI parser（确认 --json + --engine 用法）
  - `sim/npu_sim.py:595-616` — JSON 输出路径
  - `sim/design_space_explorer.py:541-557` — DSE CLI

  **Acceptance criteria**:
  - `pytest -q` 退出 0，0 failures
  - 原先 8 个红色 test node 全部绿色
  - strict quick DSE 退出 0，`errors=0`
  - CLI 基准 expected values = 同一 commit 的实测 JSON 值
  - Full DSE 输出包含全部 7 个目标引擎

  **QA scenarios**:
  - **Happy**: 完整命令清单 + 输出
    ```bash
    cd /home/prj/zhengs/caduceuscore/npu_arc_model
    PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q > .omo/evidence/task-10-pytest.txt 2>&1
    echo "PYTEST_EXIT: $?" >> .omo/evidence/task-10-verification.json
    python sim/design_space_explorer.py --quick --output .omo/evidence/task-10-quick-dse.json 2>.omo/evidence/task-10-quick-dse-stderr.txt
    echo "QUICK_DSE_EXIT: $?" >> .omo/evidence/task-10-verification.json
    python sim/design_space_explorer.py --output .omo/evidence/task-10-full-dse.json 2>.omo/evidence/task-10-full-dse-stderr.txt
    echo "FULL_DSE_EXIT: $?" >> .omo/evidence/task-10-verification.json
    python sim/npu_sim.py --json > .omo/evidence/task-10-block-cli.json 2>&1
    echo "BLOCK_CLI_EXIT: $?" >> .omo/evidence/task-10-verification.json
    python sim/npu_sim.py --engine systolic --json > .omo/evidence/task-10-systolic-cli.json 2>&1
    echo "SYSTOLIC_CLI_EXIT: $?" >> .omo/evidence/task-10-verification.json
    ```
  - **Failure**: manifest 验证器拒绝缺失命令、非零退出、dirty commit
    ```bash
    python -c "
    import json
    with open('.omo/evidence/task-10-verification.json') as f:
        for line in f:
            if 'EXIT:' in line and line.split(':')[1].strip() != '0':
                raise SystemExit(f'Non-zero exit: {line.strip()}')
    print('ALL EXITS ZERO')
    " > .omo/evidence/task-10-manifest-negative.txt 2>&1
    ```

  **Commit**: `test(cli): rebaseline repaired engine model CLI benchmarks`
  > Files: `sim/tests/test_engines.py`

---

- [ ] 11. 发布 post-fix 证据 + 更新架构文档

  **What to do**:
  1. 创建 `reports/dse-engine-model-bugs-postfix-2026-07-27.md`，包含：
     - 8 个 BUG-DSE 条目的 before/after 对照表（数值来自 Todo 10 的实测证据）
     - 每个 bug：修复的 commit hash、config hash、测试结果路径
     - 三场景推荐方案比对（修复前后的排名变化）
     - 未校准参数声明：GMMA pipeline_scale=0.05 标注"未计入签核"
     - 修复总结：Systolic→自持公式、OS→加 H、TC→加 descriptor cost、GMMA→恢复 scale
  2. 更新 `docs/NPU_Engines_Architecture_Guide.md`：
     - 第 14 行排名表 → 用 Todo 10 的实测值更新
     - 第 63 行 "193 cycles/tile" → 更新为修复后的正确值
     - 第 92 行 "OS-Systolic 零开销" → 改为 "H cycles K-reduction"
     - 第 133 行 TensorCore 章节 → 标注 descriptor cost 的影响
     - 第 191 行 GMMA 章节 → 更新为 pipeline-scaled + raw-DMA 模型
     - 第 5 行 "所有性能数据来自自研 Python simulator" → 加注 commit hash
  3. 更新 `docs/NPU硬件详细架构设计v0.1.md`：
     - 第 693-702 行 PPA 表 → 用 Todo 10 的实测值更新
     - 第 97 行 area 表 → 确认 OS PE 面积标注（OS PE ~2× Block PE area）
     - 保留版本历史，标注"recalibrated at commit <hash>"
  4. **不修改** `reports/dse-engine-model-bugs-2026-07-27.md`（原始 dated 报告）

  **References**:
  - `reports/dse-engine-model-bugs-2026-07-27.md` — 原始 bug report（只读，sha256 保持不变）
  - `.omo/evidence/task-10-*.json` — Todo 10 的实测数据（新基准来源）
  - `docs/NPU_Engines_Architecture_Guide.md:14,63,92,133,191,217` — 待更新的章节
  - `docs/NPU硬件详细架构设计v0.1.md:97,693-702` — 待更新的性能表
  - `docs/NPU硬件详细架构设计v0.1.md:5` — 版本历史（保留）

  **Acceptance criteria**:
  - 每个 BUG-DSE-001~008 在 post-fix 表中恰好出现一次，status 为 FIXED
  - 文档中每个更新过的数值 = Todo 10 的 JSON 证据中的值（± 文档 rounding）
  - `sha256sum reports/dse-engine-model-bugs-2026-07-27.md` == `61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3`
  - post-fix 报告显式声明 `GMMA pipeline_scale=0.05` 为未校准
  - 无 "210/210" 声明残留（除非独立仓库 suite 实际产生该数字并标注 commit）

  **QA scenarios**:
  - **Happy**: 报告一致性 checker 交叉验证 JSON 数值与 Markdown 表格
    ```bash
    python -c "
    import json, re, sys
    with open('reports/dse-engine-model-bugs-postfix-2026-07-27.md') as f:
        content = f.read()
    bugs = re.findall(r'BUG-DSE-\d+', content)
    assert len(set(bugs)) == 8, f'Expected 8 unique bug IDs, got {len(set(bugs))}'
    assert 'FIXED' in content
    assert 'uncalibrated' in content.lower() or '未校准' in content
    print('POST-FIX REPORT CONSISTENCY: PASS')
    " > .omo/evidence/task-11-consistency.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-11-consistency.txt
    ```
  - **Failure**: stale-number 扫描（扫描文档中所有数字声明，对照 Todo 10 evidence）
    ```bash
    grep -n 'tok/s\|cycles\|mm²\|mm2' docs/NPU_Engines_Architecture_Guide.md > .omo/evidence/task-11-stale-audit.txt 2>&1
    echo "EXIT: $?" >> .omo/evidence/task-11-stale-audit.txt
    ```

  **Commit**: `docs(dse): publish repaired engine-model verification and update architecture docs`
  > Files: `reports/dse-engine-model-bugs-postfix-2026-07-27.md`（新建）, `docs/NPU_Engines_Architecture_Guide.md`, `docs/NPU硬件详细架构设计v0.1.md`

## Final verification wave (after ALL todos)

> Runs in parallel. ALL must APPROVE. Surface results and wait for user's explicit okay.

- [ ] F1. Plan compliance audit
  - 验证 11 个 todo 和 8 个 BUG-DSE 条目都有具体证据
  - 拒绝缺失命令、自述成功、未执行的 full DSE、未绑定 commit 的 post-fix 值
  - 证据：`.omo/evidence/final-f1-plan-compliance.md`

- [ ] F2. Code quality review
  - 检查公式所有权（SystolicEngine 无共享依赖）、rounding、result schema、配置校验、错误路径
  - 运行 Python compile check + 聚焦测试
  - 拒绝：重复公式、未使用的校准常量、静默异常、魔术调参
  - 证据：`.omo/evidence/final-f2-code-quality.md`

- [ ] F3. Real manual QA
  - 实际执行 7-engine FFN_down 矩阵、Block/Systolic `npu_sim --json`、strict quick DSE、strict full DSE
  - 解析输出，验证带宽单调性、物理 DMA floor、预期排名、DSE errors=0
  - 证据：`.omo/evidence/final-f3-manual-qa.json`

- [ ] F4. Scope fidelity
  - 对比 final diff 与 scope allowlist（8 个文件改动）
  - 验证 `.omo/ultraresearch/20260723-vla-models/sources/` 未触碰
  - 验证原始用户报告 sha256 = `61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3`
  - 验证无 CaduceusCore 文件被修改
  - 验证独立 Arc Model 包含所有用于声称成功的测试/配置
  - 证据：`.omo/evidence/final-f4-scope-fidelity.md`

## Commit strategy

- 每个 todo 一个原子 commit，不合并、不 amend 跨引擎修复
- 每个 commit 前仅 stage 显式文件路径；绝不用 `git add .`
- 原始 dated 报告（`reports/dse-engine-model-bugs-2026-07-27.md`）保持未修改、未 stage
- 生成的 `.omo/evidence/` 保留用于审查，不 commit（除非仓库 convention 明确要求跟踪 evidence）
- 建议 commit 顺序：1 → 2/3/4/5 (parallel) → 6/7/8/9 (parallel) → 10 → 11

## Success criteria

- 仓库根 `pytest` 退出 0，0 failures，原先 8 个红色节点全部绿色
- SystolicEngine decode + prefill 公式自持且正确（7 Qwen GEMM × M=1,2,128 全部 parity with reference）
- OS-Systolic 包含 K-reduction（`k_reduction_cycles == H`），与 Block 同配置偏差 ≤ 10%
- TensorCore 比 Block 慢（用现有 descriptor=5，无调参）
- GMMA 带宽单调，永远不低过 raw transferred-byte time
- Strict quick/full DSE 均报 `errors=0`，full coverage 包含全部目标引擎
- CLI 基准和当前文档匹配同一 repaired commit/config 的证据
- 原始 dated 报告 sha256 不变，无脏工作树文件被覆盖
- F1-F4 全部 APPROVE
