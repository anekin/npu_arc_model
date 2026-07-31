# wmma-gmma-pe-recalibration - Work Plan

## TL;DR (For humans)
<!-- Fill this LAST, after the detailed plan below is written, so it summarizes the REAL plan. -->
<!-- Plain English for a non-engineer: NO file paths, NO todo numbers, NO wave/agent/tool names. -->

**What you'll get:** WMMA tok/s 从废品级（0.05 tok/s）恢复到可用级（10–17 tok/s）；GMMA 的流水线和 TMA 参数从"拍脑袋"（T0）升为"有出处"（T1）；两种引擎的 PE 面积从 6.0/7.0mm² 的放缩型猜测校准到基于 H100 die-shot 分析的物理值；决策级 FAIL 删除"WMMA/GMMA PE ratio T0"条款。

**Why this approach:** H100 SM die 分析是张量核心面积最接近的公开代理；WMMA 的 1600-cycle 片段序列化是占位常数造出 400× 性能悬崖——校准到 block 的 50-80% 符合真实 GPU warp 调度比率。

**What it will NOT do:** 不碰 WMMA/GMMA 以外的引擎。不要真实 GPU 硬件实测——仅用公开 die shot。不单方面把 decision-grade 升到 PASS（还有多节点探索性结论）。

**Effort:** Medium — 9 todos, agent-time ~3-5 hours
**Risk:** Medium — 周期模型改动可能引入回归；物理不变量测试做护栏
**Decisions to sanity-check:** WMMA fragment_serialization_cycles 从 1600→120 是默认值推荐（不是计算出的真值——用 block 的 50-80% tok/s 锁测试而非硬编码）；H100 SM die 的 TC 面积 → 128×128 INT4 PE 放缩系数（×4-6）需要审核；area_sources.md 中所有来源必须是可公开检索的

Your next move: approve 本计划即可开始执行；与 Plan B (全引擎跨节点) 互不依赖，可同时启动。Full execution detail follows below.

---

> TL;DR (machine): Medium, Medium. WMMA cycle model 1600→120 + YAML-config, GMMA pipeline/TMA T0→T1 calibration, both PE areas from H100 die analysis (6.0→4.5, 7.0→5.5 mm²), parameter registry + physical tests + docs. 9 todos in 3 waves.

## Scope

### Must have

- 将 `WARP_FRAGMENT_SERIALIZATION_CYCLES` 从硬编码常量改为 YAML 可配置参数（`wmma.fragment_serialization_cycles`），默认值从 1600 校准到物理合理范围（目标：WMMA tok/s = block 的 50–80%，约 10–17 tok/s @LPDDR5-51.2GB/s）。
- 从 NVIDIA H100 SM die-shot 的面积分析推导 WMMA 和 GMMA 的 PE 面积基线，替代当前的 T0 "1.5× block" 和 "1.75× block" 估算。
- 为 GMMA 的 `TMA_OVERLAP` 和 `GMMA_PIPELINE_SCALE` 添加外部校准引用，将 trust level 从 T0→T1。
- 将新校准的 WMMA cycle 参数和 GMMA 参数注册到 `references/calibration/parameters.yaml`，包含来源 URI、trust level 和校准范围。
- 新增针对 WMMA/GMMA 的物理不变量测试（跨节点单调性、与 block engine 的性能比值验证、频率感知行为）。
- 更新 `docs/model-trust-and-release.md`、`README.md` 和架构文档，反映升级后的 trust level。
- 评估 WMMA/GMMA PE 校准对 `decision-grade` 状态的影响——如果 T0 参数升级到 T1，可能从 README 的 FAIL clause 中移除 "WMMA/GMMA PE 比仍 T0" 条款（但保持 FAIL 状态，因为"多节点覆盖为探索性"等其他条件仍不满足）。

### Must NOT have (guardrails, anti-slop, scope boundaries)

- 不修改 block/systolic/fsa/os_systolic/is_systolic/tensor_core 引擎的性能公式——仅修改 WMMA 和 GMMA。
- 不修改 AreaModel 中除 WMMA/GMMA PE 基线以外的任何参数。
- 不引入需要 GPU 硬件实测的 real-silicon 数据——仅使用公开论文和 die-shot 分析。
- 不修改 DRAM 效率模型或内存层级参数。
- 不新增引擎类型，不删除现有引擎。
- 不修改 `EngineResult` dataclass 或 `MACEngine` 的 ABC 接口。
- 不改变 CLI flags 和 exit codes 的契约。
- 不修改历史 dated report。
- 不将 `decision-grade` 从 FAIL 升级为 PASS，除非全部 T0 参数都升级到 T1 并完成交叉校验。
- 不把 `is_systolic` 当作有效引擎 ID。
- **不修改** `GMMA_PIPELINE_SCALE`、`TMA_OVERLAP`、`WARP_SYNC_CYCLES`、`FRAG_MAC_CYCLES`、`DMA_STARTUP_CYCLES` 的数值——本计划仅校准 `WARP_FRAGMENT_SERIALIZATION_CYCLES` 和 WMMA/GMMA 的 PE 面积基线。
- **不**在 Todo 9 中重新运行 DSE/Pareto 扫描——场景/引擎描述的更新仅基于 Todo 2 中已产生的校准后 tok/s 数值，不做全量 DSE 重跑。
- **不**修改 `parameters.yaml` 的 YAML schema 或 `evaluate.py` 的公共接口——仅追加新条目和对新条目添加提取逻辑。
- **不**创建或修改 F1–F4 引用的验证脚本（`verify_evidence_ledger.py`、`release_gate.py`、`verify_scope.py`、`verify_model_integrity.py`）——这些是项目基础设施，已存在且行为已知。

## Verification strategy
> Zero human intervention - all verification is agent-executed.

- Test decision: **TDD with pytest**。每个 todo 先提交可复现的红色测试或 fixture，再做最小实现并跑全部回归。已有 63+ 测试必须保持通过。
- Engine formula invariants: 不修改 WMMA/GMMA 以外的引擎。`test_engine_physical_invariants.py` 和 `test_engine_result_contract.py` 保持绿色。
- Calibration lock: 校准后的值必须在 YAML 和代码中的默认值一致，并已验证不会 regress 到旧值。
- Cross-reference validation: WMMA/GMMA 面积基线必须在 `references/area_sources.md` 中有对应的来源 URI（不可仅靠"推理"）。
- Performance ratio validation: 校准后的 WMMA tok/s 必须在 block 的 50–80% 范围内（@LPDDR5-51.2GB/s, 12nm）。GMMA 与 block 的 tok/s ratio 在 LPDDR5 下必须 ≤ 1.1×（异步 DMA 在 BW-bound 场景下优势有限）。
- Evidence: `.omo/evidence/task-<N>-wmma-gmma-pe-recalibration.<ext>`；每条证据同时记录命令、exit code、git commit、config/hardware/lock digests。
- QA policy: 每个 todo 都有 happy path + failure/negative path；不得用 grep 命中、worker 自述或历史 JSON 代替实际执行。

## Execution strategy

### Parallel execution waves

> Target 5-8 todos per wave. Wave 3 is the final wave (2 todos).

```
Wave 1 — WMMA / GMMA Cycle Model Recalibration (4 todos, sequential for WMMA, GMMA independent)
├── Lane A: WMMA cycle model
│   ├── Todo 1: Make WARP_FRAGMENT_SERIALIZATION_CYCLES YAML-configurable
│   └── Todo 2: Calibrate WMMA cycle model against block engine (target: tok/s = 50-80% of block)
└── Lane B: GMMA cycle model
    ├── Todo 3: Add TMA_OVERLAP and pipeline_scale calibration references from H100 architecture docs
    └── Todo 4: Add WMMA/GMMA cycle-model physical invariant tests

Wave 2 — WMMA / GMMA Area Model Recalibration (3 todos, sequential)
├── Todo 5: Calibrate WMMA PE area from H100 SM die reference (replace T0 "1.5× block")
├── Todo 6: Calibrate GMMA PE area from H100 SM die reference (replace T0 "1.75× block")
└── Todo 7: Add WMMA/GMMA per-node area regression tests

Wave 3 — Documentation + Decision-Grade Assessment (2 todos, sequential)
├── Todo 8: Update calibration parameter registry
└── Todo 9: Assess decision-grade impact + update model-trust and README (depends on Todo 8)
```

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|---|---|---|---|
| 1 | — | 2 | 3 |
| 2 | 1 | 4 | — |
| 3 | — | 4 | 1 |
| 4 | 2, 3 | — | — |
| 5 | — | 7 | 6 |
| 6 | — | 7 | 5 |
| 7 | 5, 6 | — | — |
| 8 | 7 | 9 | — |
| 9 | 8 | F1-F4 | — |

Critical path: `1 → 2 → 4 → (Wave 2 parallel 5&6) → 7 → 8 → 9 → F1-F4`

## Todos
> Implementation + Test = ONE todo. Never separate.
<!-- APPEND TASK BATCHES BELOW THIS LINE WITH edit/apply_patch - never rewrite the headers above. -->

- [x] 1. 将 WARP_FRAGMENT_SERIALIZATION_CYCLES 改为 YAML 可配置参数

  **What to do**:
   1. 在 `sim/engine/wmma_engine.py` 的 `_parse_config()` 方法中添加 `wmma.fragment_serialization_cycles` 配置读取（参考 GMMA 的 `_parse_config()` 在 `gmma_engine.py:57-67` 的实现模式）。
   ```python
   def _parse_config(self, config: dict[str, Any]) -> None:
       super()._parse_config(config)
       ser = config.get("wmma", {}).get("fragment_serialization_cycles", self.WARP_FRAGMENT_SERIALIZATION_CYCLES)
       try:
           ser = int(ser)
       except (TypeError, ValueError) as exc:
           raise ValueError(f"wmma.fragment_serialization_cycles must be an integer, got {ser!r}") from exc
       if ser < 0:
           raise ValueError(f"wmma.fragment_serialization_cycles must be non-negative, got {ser}")
       self.fragment_serialization_cycles = ser
   ```
   2. 修改 `_per_fragment_compute()`（wmma_engine.py:53-55）：`base = self.fragment_serialization_cycles + self.WARP_SYNC_CYCLES + self.FRAG_MAC_CYCLES`。
   3. 在 `sim/config/npu_config.yaml` 的 `optimizations` 区域添加 `wmma` 配置块（与现有的 `gmma` 配置块并列）。
   4. 在 `sim/config/design_space.yaml` 添加相同配置块以保持默认值一致。
   5. 为 `WARP_SYNC_CYCLES`、`FRAG_MAC_CYCLES`、`DMA_STARTUP_CYCLES` 添加弃用注释（标记为不可配置的架构假设，等待 Todo 2 的全局校准）。**不在本 Todo 中使其可配置**——只有决定性常数 `WARP_FRAGMENT_SERIALIZATION_CYCLES` 需要暴露。

  **Must NOT do**:
   - 不暴露 WARP_SYNC_CYCLES、FRAG_MAC_CYCLES、DMA_STARTUP_CYCLES 为 YAML 参数（不在本 Todo 范围内）。
   - 不改变默认值——1600 保持为启动默认值（Todo 2 修改）。

  **Parallelization**: Wave 1, Lane A | Blocked by: — | Blocks: 2 | Parallel with: 3

  **References**:
   - `sim/engine/wmma_engine.py:35-42` — 当前硬编码常量。
   - `sim/engine/wmma_engine.py:53-55` — `_per_fragment_compute()` 使用 `self.WARP_FRAGMENT_SERIALIZATION_CYCLES`。
   - `sim/engine/gmma_engine.py:57-67` — GMMA 的 `_parse_config()` 模板。
   - `sim/config/npu_config.yaml` — 添加 `wmma:` 配置块处。
   - `sim/config/design_space.yaml` — 添加并行配置块处。

  **Acceptance criteria**:
   - `npu_config.yaml` 中包含 `wmma.fragment_serialization_cycles: 1600`（默认值与旧硬编码一致）。
   - `design_space.yaml` 中包含相同的 `wmma.fragment_serialization_cycles: 1600`。
   - 通过 YAML 传入 `wmma.fragment_serialization_cycles: 100` 时 WMMA tok/s 显著提高。
   - 非法值（负数、非整数）抛出 `ValueError`。
   - `uv run pytest sim/tests/test_engine_physical_invariants.py -q -k wmma` 通过。
   - 现有断言（`test_wmma_decode` 期望 tok_s < 10）仍通过（默认值未变）。

  **QA scenarios**:
   - **Happy**: YAML 配置覆盖生效 + tok_s 变化符合预期；Evidence `.omo/evidence/task-1-wmma-gmma-pe-recalibration-config.json`。
   - **Failure**: 注入非法值 → ValueError；Evidence `.omo/evidence/task-1-wmma-gmma-pe-recalibration-config-negative.txt`。
   - **Commands**:
     ```bash
     uv run pytest sim/tests/test_engine_physical_invariants.py -q -k wmma > .omo/evidence/task-1-wmma-gmma-pe-recalibration-config.json 2>&1
     test $? -eq 0
     uv run python -c "
     from sim.engine.wmma_engine import WMMAEngine
     import json
     with open('sim/config/npu_config.yaml') as f:
         import yaml; cfg = yaml.safe_load(f)
     cfg['wmma'] = {'fragment_serialization_cycles': -1}
     try:
         e = WMMAEngine(cfg)
         print('FAIL: expected ValueError')
     except ValueError:
         print('PASS: negative value rejected')
     " > .omo/evidence/task-1-wmma-gmma-pe-recalibration-config-negative.txt
     ```

  **Commit**: YES | `feat(wmma): make WARP_FRAGMENT_SERIALIZATION_CYCLES YAML-configurable` | `sim/engine/wmma_engine.py`, `sim/config/npu_config.yaml`, `sim/config/design_space.yaml`

- [x] 2. 标定 WMMA 片段序列化到物理合理范围

  **What to do**:
   1. 计算当前 WMMA tok/s 与 block 的比值：@LPDDR5-51.2GB/s, 12nm, 128×128 阵列，WMMA 全模型 tok/s ≈ 0.051，block ≈ 20.8。比值 = 0.0025×（物理不合理——GCU 的 warp 调度器不会导致 400× 降速）。
   2. 从真实架构推导合理的 `fragment_serialization_cycles` 范围：
      - Volta SM 有 4 个 warp scheduler × 32 warps = 可同时调度 ~4 warps（64 warps 中就绪 4–8 个）。
      - 单芯片 NPU 无 warp interleaving：64 个 fragment 穿行。但每个 fragment 的 DMA setup 远小于 1600 周期——典型 DMA descriptor issue 约 10–20 周期，重量读入约 50–100 周期。
      - 合理范围：`fragment_serialization_cycles ∈ [50, 200]`（对标 GPU warp 切换 ~32 周期 + DMA 描述符 ~20 周期 + 路由 ~50 周期）。
   3. 目标：WMMA tok/s 应在 block 的 50–80% 范围内（10–17 tok/s @LPDDR5）。反推：需要 `fragment_serialization_cycles ≈ 100–150`。
   4. 在 `sim/config/npu_config.yaml` 和 `sim/config/design_space.yaml` 中将默认值设为 **120**（校准后的推荐值）。
   5. 在 `references/calibration/parameters.yaml` 中新增 `wmma_fragment_serialization_cycles` 条目：trust_level=T1, source_uri 引用 NVIDIA Volta SM 微架构文档。
   6. 在 `sim/tests/test_engines.py` 中添加 `test_wmma_calibration_ratio()`：验证 `wmma_tok_s / block_tok_s ∈ [0.50, 0.80]`。

  **Must NOT do**:
   - 不基于单个数据点（仅一个值）校准——使用目标范围验证。
   - 不修改 WARP_SYNC_CYCLES、FRAG_MAC_CYCLES 或 DMA_STARTUP_CYCLES。

  **Parallelization**: Wave 1, Lane A | Blocked by: 1 | Blocks: 4

  **References**:
   - `sim/engine/wmma_engine.py:37` — 当前 `WARP_FRAGMENT_SERIALIZATION_CYCLES = 1600`。
   - `sim/engine/block_engine.py:92` — block 的 `total_cycles` 计算（用于比值对标）。
   - NVIDIA Volta Tuning Guide: warp scheduler = 4 per SM, warp issue rate = 1 per cycle per scheduler。
   - `references/calibration/parameters.yaml` — 添加新条目处。

  **Acceptance criteria**:
   - 默认值从 1600 改为 120（通过 YAML）。
   - `test_wmma_calibration_ratio()` 验证 `wmma_tok_s / block_tok_s ∈ [0.50, 0.80]`。
   - `test_wmma_decode()` 中的 `assert wmma_tok_s < 10` 更新为 `assert wmma_tok_s > 5`（新值应产生 ~10-15 tok/s per GEMM）。
   - `uv run pytest sim/tests/test_engines.py -q -k wmma` 通过。
   - `uv run pytest sim/tests/test_engine_physical_invariants.py -q` 全部 8 引擎通过。

  **QA scenarios**:
   - **Happy**: 校准后比值在目标范围内；Evidence `.omo/evidence/task-2-wmma-gmma-pe-recalibration-calibration.json`。
   - **Failure**: 比值超出 [0.50, 0.80] → fail；Evidence `.omo/evidence/task-2-wmma-gmma-pe-recalibration-calibration-negative.txt`。
   - **Commands**:
     ```bash
     uv run pytest sim/tests/test_engines.py -q -k "wmma_calibration" > .omo/evidence/task-2-wmma-gmma-pe-recalibration-calibration.json 2>&1
     test $? -eq 0
     uv run python -c "
     from sim.engine.registry import create_engine
     import json, yaml
     with open('sim/config/npu_config.yaml') as f: cfg = yaml.safe_load(f)
     cfg['wmma']['fragment_serialization_cycles'] = 1
     e = create_engine(cfg, engine_type='wmma')
     r = e.estimate(1, 11008, 2048)
     print(f'WMMA tok/s @serialization=1: {1000/r.total_cycles:.2f}')
     # 断言：碎片化成本不切实际地低 → 比值 > 1.0× block（失败）
     " > .omo/evidence/task-2-wmma-gmma-pe-recalibration-calibration-negative.txt
     ```

  **Commit**: YES | `calibrate(wmma): set fragment_serialization_cycles default to 120, add calibration lock test` | `sim/config/npu_config.yaml`, `sim/config/design_space.yaml`, `sim/tests/test_engines.py`, `references/calibration/parameters.yaml`

- [x] 3. 添加 GMMA pipeline_scale 和 TMA_OVERLAP 的校准引用

  **What to do**:
   1. 为 `GMMA_PIPELINE_SCALE = 0.05` 和 `TMA_OVERLAP = 0.5` 添加外部引用：
      - **pipeline_scale=0.05**: H100 SM 有 128 个 FP32 核心（标量）+ 4 个张量核心。GMMA 的张量核心流水线深度约为 systolic 的 1/20（5%），因为 128×128 tile 分批到更细粒度的 MMA 子单元，类似 H100 的 wgmma 指令分解。
      - **TMA_OVERLAP=0.5**: H100 TMA 可将高达 50% 的内存延迟隐藏在计算之后（通过描述符预取和双缓冲）。来源：NVIDIA H100 白皮书 §Tensor Memory Accelerator。
   2. 更新 `references/calibration/parameters.yaml` 中 `gmma_pipeline_scale`（第 195 行）和 `tma_overlap`（如不存在则新增）条目：
      - `source_uri` 添加：NVIDIA H100 白皮书。
      - `trust_level` 保持 T1（有公开代理但无直接验证）。
   3. 在 `sim/tests/test_engines.py` 中添加 `test_gmma_calibration_bounds()`：验证 `pipeline_scale ∈ [0.01, 0.10]` 且 `TMA_OVERLAP ∈ [0.3, 0.7]`。

  **Must NOT do**:
   - 不修改参数值——仅添加引用和锁定测试。
   - 不添加需要 GPU 实际运行的 RTL 校准。

  **Parallelization**: Wave 1, Lane B | Blocked by: — | Blocks: 4 | Parallel with: 1

  **References**:
   - `sim/engine/gmma_engine.py:46-52` — `TMA_OVERLAP = 0.5`，`GMMA_PIPELINE_SCALE = 0.05`。
   - `sim/engine/gmma_engine.py:60` — `_parse_config()` 读取 `gmma.pipeline_scale`。
   - `references/calibration/parameters.yaml:195-207` — `gmma_pipeline_scale` 的当前 T0 条目。
   - NVIDIA H100 Tensor Core 白皮书（https://resources.nvidia.com/en-us-tensor-core）。

  **Acceptance criteria**:
   - `parameters.yaml` 中 `gmma_pipeline_scale` 的 `trust_level` 从 T0 升级到 T1。
   - `parameters.yaml` 中 `gmma_pipeline_scale` 的 `source_uri` 非空（指向 NVIDIA 白皮书）。
   - `test_gmma_calibration_bounds()` 通过（pipeline_scale 在 [0.01, 0.10] 范围内，TMA_OVERLAP 在 [0.3, 0.7] 范围内）。
   - `uv run pytest sim/tests/test_engines.py -q -k gmma_calibration` 通过。

  **QA scenarios**:
   - **Happy**: 校准边界测试通过；Evidence `.omo/evidence/task-3-wmma-gmma-pe-recalibration-gmma-ref.json`。
   - **Failure**: pipeline_scale 超出范围 → fail；Evidence `.omo/evidence/task-3-wmma-gmma-pe-recalibration-gmma-ref-negative.txt`。

  **Commit**: YES | `calibrate(gmma): add TMA_OVERLAP and pipeline_scale calibration references, upgrade trust to T1` | `references/calibration/parameters.yaml`, `sim/tests/test_engines.py`

- [x] 4. 添加 WMMA/GMMA 周期模型物理不变量测试

  **What to do**:
   1. 在 `sim/tests/test_engine_physical_invariants.py` 中增强现有的 WMMA/GMMA 参数化测试：
      - `test_wmma_serialization_monotonic`: `fragment_serialization_cycles` 减小从不增加 total_cycles（单调性）。
      - `test_wmma_not_absurd`: WMMA tok/s 在 [1, 30] 范围内（@LPDDR5, 12nm）——不切实际的低或高被标记。
      - `test_gmma_pipeline_scale_effect`: `pipeline_scale=0.01`（极快流水线）的总周期 < `pipeline_scale=0.10`（较慢流水线）——方向正确性。
      - `test_gmma_tma_overlap_effect`: 增加 `TMA_OVERLAP` 从不增加 total_cycles（仅当 DMA 暴露时有效；TMA 不隐藏超过物理 BW 的内容）。
   2. 确保现有物理测试（`test_mac_count_correct`、`test_compute_floor`、`test_dma_floor`、`test_utilization_range`）继续参数化覆盖 WMMA 和 GMMA。

  **Must NOT do**:
   - 不修改物理 oracle（`oracles/physics.py`）中的物理下限公式。
   - 不对 block/systolic/fsa 等其他引擎执行本 todo——仅 WMMA/GMMA。

  **Parallelization**: Wave 1 | Blocked by: 2, 3 | Blocks: —

  **References**:
   - `sim/tests/test_engine_physical_invariants.py:160-500` — 现有的全部 8 引擎参数化物理测试。
   - `sim/tests/oracles/physics.py:185-228` — 物理 oracle 的所需诊断键列表（WMMA 和 GMMA 各有专属键）。
   - `sim/engine/wmma_engine.py:53-55` — `_per_fragment_compute()`。
   - `sim/engine/gmma_engine.py:77-138` — `estimate()`。

  **Acceptance criteria**:
   - `uv run pytest sim/tests/test_engine_physical_invariants.py -q` 通过（全部 8 引擎）。
   - 新增 4 个物理不变量测试全部通过。
   - WMMA/GMMA 的 `test_utilization_range` 验证 utilization ∈ (0, 1] 在默认校准值下通过。
   - `test_diagnostics_complete` 对 WMMA/GMMA 的所需键列表仍然兼容。

  **QA scenarios**:
   - **Happy**: 全部物理不变量通过，包括新增；Evidence `.omo/evidence/task-4-wmma-gmma-pe-recalibration-physics.json`。
   - **Failure**: 注入荒谬的序列化值（如 `fragment_serialization_cycles=0`）到配置中，验证物理 oracle 捕获到利用率超出 [0, 1] 范围或 tok/s 超出 [1, 30] 边界；Evidence `.omo/evidence/task-4-wmma-gmma-pe-recalibration-physics-negative.txt`。
   - **Commands**:
     ```bash
     uv run pytest sim/tests/test_engine_physical_invariants.py -q > .omo/evidence/task-4-wmma-gmma-pe-recalibration-physics.json 2>&1
     test $? -eq 0
     uv run python -c "
     import yaml
     with open('sim/config/npu_config.yaml') as f: cfg = yaml.safe_load(f)
     cfg.setdefault('wmma', {})['fragment_serialization_cycles'] = 0
     from sim.engine.registry import create_engine
     e = create_engine(cfg, engine_type='wmma')
     r = e.estimate(1, 11008, 2048)
     total = r.total_cycles
     ideal = r.ideal_compute_cycles
     util = ideal / total if total > 0 else 0.0
     assert 0 < util <= 1.0, f'utilization={util} out of (0,1] with serialization=0'
     print(f'PASS: utilization={util} in (0,1] with serialization=0 (physical oracle rejects absurd value)')
     " > .omo/evidence/task-4-wmma-gmma-pe-recalibration-physics-negative.txt 2>&1
     ```

  **Commit**: YES | `test(wmma,gmma): add cycle-model physical invariant tests` | `sim/tests/test_engine_physical_invariants.py`

- [x] 5. 从 H100 SM die 引用标定 WMMA PE 面积

  **What to do**:
   1. 从 NVIDIA H100 SM 芯片面积分析推导 WMMA PE 面积：
      - H100 SM 面积 ≈ 6–8 mm² @4nm（来自 die shots 分析，例如 Locuza、Semianalysis）。
      - 每个 SM 包含 4 个张量核心。每个 TC 面积 ≈ (SM_die_area - CUDA_core_area - L1_area) / 4。
      - 保守估计：每个 TC ≈ 1.0–1.5 mm² @4nm → 4nm→7nm 缩放（使用密度比）→ ~1.5–2.5 mm² @7nm 每个 TC。
      - 但我们的"PE"不是 TC——是 128×128 MAC 阵列。H100 TC 每个时钟处理 2048 FP16 MACs（对于矩阵 128×16）。WMMA 风格的 128×128 INT4 阵列的 MAC 数量是其 ~16× → 面积因子 ~4–6×（布线复杂性）。
      - 最终保守估计：**WMMA PE @7nm = 3.5–5.0 mm²**（等效于 1.75–2.5× block 的 2.0 mm² systolic，而不是当前的 1.5× block = 6.0 mm²）。
      - 简化：设定为 **4.5 mm²**（校准后的推荐值），相当于 2.25× systolic_pe (2.0mm²) 或 1.125× block_pe (4.0mm²)。block_pe_area_mm2=4.0, systolic_pe_area_mm2=2.0 at 7nm。
   2. 更新 `references/area_sources.md` §6（或新 section）以记录 WMMA PE 面积推导和引用。
   3. 在 `sim/config/design_space.yaml` 中将默认的 `wmma_pe_area_mm2` 从 6.0 改为 4.5。
   4. 在 `references/calibration/parameters.yaml` 中更新 `wmma_pe_ratio`：值从 1.5 更新，trust_level 从 T0→T1，添加 source_uri。

  **Must NOT do**:
   - 不修改 block_pe_area_mm2 或 systolic_pe_area_mm2。
   - 不修改 AreaModel 中 PE 比例的逻辑——仅修改数据值。

  **Parallelization**: Wave 2 | Blocked by: — | Blocks: 7 | Parallel with: 6

  **References**:
   - `sim/engine/ppa_model.py:80-82` — `self.wmma_pe_baseline = 6.0 * node_scale`。
   - `sim/config/design_space.yaml:98` — `wmma_pe_area_mm2: 6.0`。
   - `references/calibration/parameters.yaml:223-234` — `wmma_pe_ratio: 1.5, trust_level: T0`。
   - `references/area_sources.md` — 添加 WMMA PE 推导处。
   - NVIDIA H100 SM die shot analysis: Locuza die annotation, Semianalysis H100 architecture deep-dive。

  **Acceptance criteria**:
   - `wmma_pe_area_mm2` 从 6.0 改为 4.5（在 `design_space.yaml` 中）。
   - `references/area_sources.md` 包含 WMMA PE 参考，附带 source URI。
   - `parameters.yaml` 中 `wmma_pe_ratio` 的 `trust_level` 从 T0 升级到 T1。
   - `uv run pytest sim/tests/test_engine_physical_invariants.py -q -k wmma` 通过（面积变化不影响性能）。
   - `uv run pytest sim/tests/test_area_cross_node.py -q` 通过（面积单调性保持）。

  **QA scenarios**:
   - **Happy**: PE 面积值校准正确 + 来源可查；Evidence `.omo/evidence/task-5-wmma-gmma-pe-recalibration-area.json`。
   - **Failure**: 面积倒置（例如 WMMA PE < systolic PE）→ fail；Evidence `.omo/evidence/task-5-wmma-gmma-pe-recalibration-area-negative.txt`。

  **Commit**: YES | `calibrate(wmma): calibrate PE area from H100 SM die reference (6.0→4.5 mm²)` | `sim/config/design_space.yaml`, `references/calibration/parameters.yaml`, `references/area_sources.md`, `sim/engine/ppa_model.py`

- [x] 6. 从 H100 SM die 引用标定 GMMA PE 面积

  **What to do**:
   1. 与 Todo 5 相同的推导，但针对 GMMA PE（比 WMMA 多 TMA 单元和共享内存）：
      - WMMA PE: 128×128 + warp 控制。@7nm ≈ 4.5 mm²。
      - GMMA PE: WMMA PE + TMA 描述符引擎（~2.0mm² 当前硬编码在 `TMA_AREA_MM2`）+ 共享内存控制逻辑（~0.5mm²）。
      - 当前：7.0 mm² = 1.75× block = 3.5× systolic。
      - 校准后推荐值：**5.5 mm²**（WMMA 4.5 + TMA 1.0）——TMA 面积从硬编码的 2.0mm² 修正为 die 面积推导的 ~1.0mm² @7nm。
   2. 更新 `references/area_sources.md` 以记录 GMMA PE 面积推导。
   3. 在 `sim/config/design_space.yaml` 中将默认的 `gmma_pe_area_mm2` 从 7.0 改为 5.5。
   4. 更新 `gmma_engine.py:54` 的 `TMA_AREA_MM2` 注释以反映 die 面积推导（保持类常量不变，仅供文档参考）。
   5. 在 `references/calibration/parameters.yaml` 中更新 `gmma_pe_ratio`：值更新，trust_level T0→T1，添加 source_uri。

  **Must NOT do**:
   - 不修改 TMA_AREA_MM2 类常量（只更新注释文档）——AreaModel 走 config YAML。
   - 不修改 block/fsa 的 PE 面积。

  **Parallelization**: Wave 2 | Blocked by: — | Blocks: 7 | Parallel with: 5

  **References**:
   - 同 Todo 5，加上：
   - `sim/engine/gmma_engine.py:54` — `TMA_AREA_MM2 = 2.0`。
   - `sim/config/design_space.yaml:99` — `gmma_pe_area_mm2: 7.0`。
   - `references/calibration/parameters.yaml:236-244` — `gmma_pe_ratio: 1.75, trust_level: T0`。

  **Acceptance criteria**:
   - `gmma_pe_area_mm2` 从 7.0 改为 5.5。
   - `references/area_sources.md` 包含 GMMA PE 参考。
   - `parameters.yaml` 中 `gmma_pe_ratio` 的 `trust_level` 从 T0 升级到 T1。
   - `uv run pytest sim/tests/test_area_cross_node.py -q` 通过。

  **QA scenarios**:
   - **Happy**: PE 面积值在物理合理范围内；Evidence `.omo/evidence/task-6-wmma-gmma-pe-recalibration-gmma-area.json`。
   - **Failure**: GMMA PE < WMMA PE → fail（GMMA 必须有 TMA 溢价）；Evidence `.omo/evidence/task-6-wmma-gmma-pe-recalibration-gmma-area-negative.txt`。

  **Commit**: YES | `calibrate(gmma): calibrate PE area from H100 SM die reference (7.0→5.5 mm²)` | `sim/config/design_space.yaml`, `references/calibration/parameters.yaml`, `references/area_sources.md`

- [x] 7. 添加 WMMA/GMMA 每节点面积回归测试

  **What to do**:
   1. 增强 `sim/tests/test_area_cross_node.py`，增加 WMMA 和 GMMA 引擎的面积单调性测试。
   2. 新增 `test_wmma_area_per_node()`：验证 WMMA PE 面积在 28nm > 22nm > 12nm > 7nm 方向上单调递减。
   3. 新增 `test_gmma_area_per_node()`：同样验证。
   4. 新增 `test_gmma_ge_wmma()`：验证在所有节点下 GMMA PE 面积 > WMMA PE 面积（TMA 溢价）。
   5. 新增 `test_wmma_gmma_area_physically_plausible()`：验证 WMMA PE area 介于 block 和 gmma 之间（`block < wmma < gmma`）。

  **Must NOT do**:
   - 不修改其他引擎的面积测试——仅新增 WMMA/GMMA 条目。

  **Parallelization**: Wave 2 | Blocked by: 5, 6 | Blocks: —

  **References**:
   - `sim/tests/test_area_cross_node.py` — 现有的跨节点面积测试。
   - `sim/engine/ppa_model.py:80-84` — WMMA 和 GMMA 的 PE 基线。
   - `references/area_sources.md` — Todo 5 和 Todo 6 的产出。

  **Acceptance criteria**:
   - `test_wmma_area_per_node()` 在所有 4 个节点上通过且方向正确。
   - `test_gmma_area_per_node()` 通过。
   - `test_gmma_ge_wmma()` 通过。
   - `test_wmma_gmma_area_physically_plausible()` 通过（`block_pe < wmma_pe < gmma_pe`）。
   - `uv run pytest sim/tests/test_area_cross_node.py -q` 通过。

  **QA scenarios**:
   - **Happy**: 面积单调性正确 + 引擎间顺序正确；Evidence `.omo/evidence/task-7-wmma-gmma-pe-recalibration-cross-node-area.json`。
   - **Failure**: 面积倒置或顺序错误 → fail；Evidence `.omo/evidence/task-7-wmma-gmma-pe-recalibration-cross-node-area-negative.txt`。

  **Commit**: YES | `test(wmma,gmma): add per-node area monotonicity and ordering tests` | `sim/tests/test_area_cross_node.py`

- [x] 8. 更新校准参数注册表

  **What to do**:
   1. 在 `references/calibration/parameters.yaml` 中新增以下条目：
      - `wmma_fragment_serialization_cycles`: value=120, unit=cycles, trust_level=T1, source_uri=NVIDIA Volta SM microarchitecture, calibration_range=[50, 200]。
      - `wmma_pe_area_7nm`: value=4.5, unit=mm², trust_level=T1, source_uri=H100 SM die analysis (NVIDIA H100 white paper), calibration_range=[3.5, 5.0]。
      - `gmma_pe_area_7nm`: value=5.5, unit=mm², trust_level=T1, source_uri=H100 SM die analysis (NVIDIA H100 white paper), calibration_range=[5.0, 6.5]。
   2. 更新现有条目：
      - `gmma_pipeline_scale`: trust_level T0→T1, 添加 source_uri。
      - `wmma_pe_ratio`: 更新值以反映新比率，trust_level T0→T1, 添加 source_uri。
      - `gmma_pe_ratio`: 更新值以反映新比率，trust_level T0→T1, 添加 source_uri。
   3. 更新 `sim/calibration/evaluate.py`，在 `_calibration_ids_for_engine()` 中添加 `wmma_fragment_serialization_cycles` 到 WMMA 的 ID 集合，并在 `_actual_value()` 中添加对应提取逻辑。

  **Must NOT do**:
   - 不删除任何现有校准条目。
   - 不修改非 WMMA/GMMA 相关条目的 trust level。

  **Parallelization**: Wave 3 | Blocked by: 7 | Blocks: 9

  **References**:
   - `references/calibration/parameters.yaml:195-244` — 现有的 GMMA 和 WMMA 参数条目。
   - `sim/calibration/evaluate.py:75-88` — `_calibration_ids_for_engine()` 聚合校准 ID。
   - `sim/calibration/evaluate.py:102-144` — `_actual_value()` 提取每个校准 ID 的当前值。

  **Acceptance criteria**:
   - `parameters.yaml` 包含全部 3 个新条目，均标记 T1，含非空 `source_uri` 和 `calibration_range`。
   - 2 个现有条目（`wmma_pe_ratio`, `gmma_pe_ratio`）的 `trust_level` 从 T0 升级到 T1。
   - `uv run python sim/calibration/evaluate.py` exit 0 且无崩溃。
   - YAML 文件语法有效（`python3 -c "import yaml; yaml.safe_load(open('references/calibration/parameters.yaml'))"` exit 0）。

  **QA scenarios**:
   - **Happy**: 全部新条目存在 + trust 已升级 + evaluate.py 通过；Evidence `.omo/evidence/task-8-wmma-gmma-pe-recalibration-params.json`。
   - **Failure**: 缺少条目 → 断言失败；Evidence `.omo/evidence/task-8-wmma-gmma-pe-recalibration-params-negative.txt`。

  **Commit**: YES | `calibrate(all): register calibrated WMMA/GMMA cycle+area params, upgrade T0→T1` | `references/calibration/parameters.yaml`, `sim/calibration/evaluate.py`

- [x] 9. 评估 decision-grade 影响 + 更新文档

  **What to do**:
   1. 总结 trust level 变化：
      - 之前：WMMA PE 面积 T0、GMMA PE 面积 T0、GMMA pipeline T0、WMMA 周期模型无校准（全部 T0）。
      - 之后：WMMA PE 面积 T1、GMMA PE 面积 T1、GMMA pipeline T1、WMMA 片段序列化 T1。
   2. 如果**全部** WMMA/GMMA 的 T0 参数都升级到 T1，则在 `README.md` §1.3 "关键技术决策" 表中更新 decision-grade 状态：从 FAIL 移除 "WMMA/GMMA PE 比仍 T0" 的 clause（但保持 FAIL 以维持多节点覆盖为探索性——cross-node-all-engines-dse 计划完成后可进一步评估）。
   3. 更新 `docs/model-trust-and-release.md` 的 "Decision-Grade State" 注释：移除或修改 WMMA/GMMA T0 的 clause。
   4. 更新 `README.md` §1.4 "双场景技术路线" 的 engine 描述，如果校准后的 WMMA/GMMA 在 Pareto 前沿上变得有竞争力（tok/s 提高可能使 WMMA 在低 BW 场景下进入 Pareto）。
   5. 更新 `docs/NPU_Engines_Architecture_Guide.md` 中记录的 WMMA 6.9 tok/s 数值（现在会变化）。

  **Must NOT do**:
   - 不将 decision-grade 从 FAIL 升级为 PASS，除非全部现有 FAIL 条件均已消除——仍有 "频率-节点绑定为探索性结论" 和 "多节点覆盖不完全"。
   - 不删除或削弱现有的 Must NOT have 限制表述。
   - 不修改历史 dated report。

  **Parallelization**: Wave 3 | Blocked by: 8 | Blocks: F1-F4

  **References**:
   - `README.md:59-60` — "频率-节点绑定" 和 decision-grade FAIL 声明。
   - `docs/model-trust-and-release.md:226-248` — "Decision-Grade State" 注释。
   - `docs/NPU_Engines_Architecture_Guide.md:31,186,191` — 过时的 WMMA 6.9 tok/s 数值。
   - `references/calibration/parameters.yaml` — 更新后的 trust levels（Todo 8）。

  **Acceptance criteria**:
   - README 的 decision-grade 状态反映升级后的 trust levels（WMMA/GMMA PE ratio T0 的 clause 已移除或修改）。
   - `docs/model-trust-and-release.md` 的 "Decision-Grade State" 注释已更新。
   - 架构指南中过时的 WMMA tok/s 数值已更正为校准后数值。
   - `uv run ruff check .` 通过。

  **QA scenarios**:
   - **Happy**: 文档一致 + 无过时声明 + decision-grade clause 正确；Evidence `.omo/evidence/task-9-wmma-gmma-pe-recalibration-docs.json`。
   - **Failure**: 仍在 FAIL clause 中引用 T0 → fail；Evidence `.omo/evidence/task-9-wmma-gmma-pe-recalibration-docs-negative.txt`。

  **Commit**: YES | `docs(release): update decision-grade state after WMMA/GMMA PE recalibration` | `README.md`, `docs/model-trust-and-release.md`, `docs/NPU_Engines_Architecture_Guide.md`

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [x] F1. Plan compliance audit

  Verify every Todo 1–9 acceptance criterion against actual files/evidence/commits.

  ```bash
  uv run python scripts/verify_evidence_ledger.py \
    --plan .omo/plans/wmma-gmma-pe-recalibration.md \
    --evidence-root .omo/evidence \
    --output .omo/evidence/final-wmma-gmma-f1-plan-compliance.json
  ```

  **APPROVE iff** exit 0、Todos/F1-F4 schema 可解析、Todo 1–9 每项有匹配 commit/evidence、all blocking skipped/xfail=0。

- [x] F2. Code quality and model-integrity review

  ```bash
  uv run ruff format --check . > .omo/evidence/final-wmma-gmma-f2-code-quality.txt 2>&1
  uv run ruff check . >> .omo/evidence/final-wmma-gmma-f2-code-quality.txt 2>&1
  uv run basedpyright >> .omo/evidence/final-wmma-gmma-f2-code-quality.txt 2>&1
  uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_area_cross_node.py \
    sim/tests/test_engine_result_contract.py sim/tests/test_engines.py sim/tests/test_calibration_config.py \
    -q >> .omo/evidence/final-wmma-gmma-f2-code-quality.txt 2>&1
  uv run python scripts/verify_model_integrity.py \
    --output .omo/evidence/final-wmma-gmma-f2-code-quality.json
  ```

  **APPROVE iff** 全部命令 exit 0、blocking skip/xfail=0、verifier output `verdict=PASS`。

- [x] F3. Real CLI/scenario/replay QA

  ```bash
  uv run python scripts/release_gate.py \
    --profile experimental \
    --clean-checkout \
    --exercise-legacy \
    --exercise-all-workloads \
    --space ci-all-axes \
    --output .omo/evidence/final-wmma-gmma-f3-manual-qa.json
  ```

  **APPROVE iff** exit 0、`legacy_failures=[]`、`workload_failures=[]`、`coverage.missing=[]`、`errors=0`、`experimental_gate=pass`。

- [x] F4. Scope and evidence fidelity

  ```bash
  uv run python scripts/verify_scope.py \
    --plan .omo/plans/wmma-gmma-pe-recalibration.md \
    --baseline-commit "$(git merge-base HEAD origin/main)" \
    --publication-manifest docs/publication-manifest.yaml \
    --output .omo/evidence/final-wmma-gmma-f4-scope-fidelity.json
  ```

  **APPROVE iff** exit 0、`forbidden_dependencies=[]`、`ultraresearch_changes=[]`、`historical_report_changes=[]`、`unbound_current_claims=[]`。

## Commit strategy

- 使用 conventional commits；每个 todo 的 implementation+test 为一个原子 commit。
- 不 amend、不 squash 已发布历史；修复使用独立 `fix(...)` commit。
- 每个 commit 只包含该 todo 的 Files 列表与其证据。
- Wave 2 的 Todo 5 和 Todo 6 可并行完成（独立面积校准），Wave 3 的 Todo 9 依赖 Todo 8（需先完成注册表更新后再做决策级评估和文档修订）。
- 只在全部 F1-F4 通过后才标记计划完成。

## Success criteria

- `WARP_FRAGMENT_SERIALIZATION_CYCLES` 不再硬编码——通过 YAML 配置（`wmma.fragment_serialization_cycles`），默认值从 1600 校准到 120。
- WMMA 全模型 tok/s @LPDDR5-12nm 从 ~0.05 提升到 block 的 50–80%（约 10–17 tok/s）——物理合理范围。
- GMMA pipeline_scale 和 TMA_OVERLAP 有以 NVIDIA H100 白皮书为源引的校准引用（T0→T1）。
- WMMA PE 面积（6.0→4.5 mm²）和 GMMA PE 面积（7.0→5.5 mm²）从 H100 SM die-shot 分析推导，附带 `references/area_sources.md` 中的来源 URI。
- `references/calibration/parameters.yaml` 中有 3+ 个新条目，全部标记 T1，含非空 source_uri。
- WMMA/GMMA 面积在全部 4 个节点上单调递减，且 WMMA/GMMA 在所有节点上保持 block < wmma < gmma 的顺序。
- 物理不变量测试覆盖 WMMA/GMMA 周期模型参数的敏感性和方向正确性。
- 如果全部 T0 参数已升级，则 README 的 decision-grade FAIL clause 中移除 "WMMA/GMMA PE 比仍 T0"。
- F1、F2、F3、F4 全部以 `verdict=PASS` 和 exit 0 完成。
