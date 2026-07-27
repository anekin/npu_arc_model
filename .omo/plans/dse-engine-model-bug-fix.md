# DSE Engine Model Bug Fix Plan

## TL;DR
> Summary:      修复并锁定 OS-Systolic、Systolic/MXU、TensorCore、GMMA 的解析时序契约，使当前 8 个回归失败转绿，同时把独立 Arc Model 的测试、DSE 严格错误门禁和验证证据补齐。
> Deliverables:
> - 4 条 engine timing 修复路径及共享的 Systolic timing helper
> - 被 Git 跟踪、可由仓库根目录直接执行的 pytest 回归
> - fail-closed DSE、全引擎覆盖和 commit-bound 验证证据
> - 更新后的当前架构文档和一份新的 post-fix 报告
> Effort:       Large
> Risk:         High - 解析公式、相对排名和已发布性能数字会一起变化，且现有测试与文档已经发生版本漂移

## Scope

### Must have
- 在 standalone Arc Model `main` 上修复 `BUG-DSE-001` 至 `BUG-DSE-008`；相邻 `../CaduceusCore` 仅作为只读来源证据，不在那里实施修改。
- 以当前架构指南的工程语义为契约：
  - OS 与 Block 在相同 64x64、LPDDR5 条件下应处于同一性能区间，但 OS 仍需支付 K reduction。
  - TensorCore 因碎片化 descriptor 开销应略慢于 Block。
  - GMMA 在 LPDDR5 下受原始 DRAM 字节时间约束，随带宽提升而改善，TMA 只能隐藏 latency。
  - `SystolicEngine` 与 `MXUModel` 对相同配置和形状必须逐 cycle 一致。
- 采用 TDD：每个模型修复先让针对性测试稳定复现，再改最小实现，最后执行跨引擎和 CLI 回归。
- 将用户复制的 `sim/tests/test_engines.py`、`sim/tests/test_engine_instantiate.py` 纳入仓库测试面；保留测试意图，但修正已确认的测试自相矛盾和 fail-fast 遮蔽。
- DSE 默认 fail closed；只有显式 exploratory 参数才允许部分结果，且必须打印失败数量和失败配置。
- 所有新性能数字从修复后的命令输出生成，证据必须包含 Git commit、dirty 状态、配置哈希、完整命令和退出码。
- 原始 dated bug report 保持字节不变；另建 post-fix 报告。
- 原始报告的计划基线 SHA-256 为 `61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3`。

### Must NOT have (guardrails, anti-slop, scope boundaries)
- 不修改 Golden Executor、Func Model 数值路径、RTL 或相邻 CaduceusCore 仓库。
- 不修改 `.omo/ultraresearch/20260723-vla-models/sources/`，不覆盖任何用户未跟踪文件，不执行全目录 stage。
- 不为通过相对排名断言而反复调大常数。
- 不新增无校准来源的 barrier/setup 常数：
  - TensorCore 复用 `sim/config/*:dma.descriptor_overhead_cycles=5`；
  - 如果需要 barrier 字段，默认必须是 0，并在 post-fix 报告标为“未校准、未计入签核”。
- 不允许 GMMA 的重叠后 DMA 时间替代物理 raw-DMA floor。
- 不把历史 dated 报告中的旧数字原地重写；只更新当前指南和新报告。
- 不把 quick DSE 的 exit 0、grep 命中或代理自述当作完成证据。
- 不顺带重构 FSA、Input-Stationary、WMMA、PPA 模型，除非统一结果契约测试暴露了阻断本任务的直接回归。

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Test decision: TDD + pytest 9，仓库根目录统一使用 `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider`.
- QA policy: every todo has agent-executed scenarios；测试输出、CLI JSON 和 DSE 日志均写入独立 evidence 文件。
- Evidence: `.omo/evidence/task-<N>-<slug>.<ext>`
- Source authority:
  - 实施目标：当前仓库 `HEAD`。
  - 只读来源：`../CaduceusCore` 的提交历史和同名测试。
  - shipment check：最终 scope audit 必须证明所有测试、配置和报告均存在于 standalone Arc Model，而不是只存在于来源仓。
- Misleading-success policy:
  - pytest 证据同时记录 collected、passed、failed、exit code。
  - DSE 证据同时记录 generated、evaluated、filtered、errors、valid 和 exit code。
  - 有 unexpected skip/error 时不得签核。

## Execution strategy

### Parallel execution waves
> Target 5-8 todos per wave. Wave 3 is integration-constrained; the final verification wave is intentionally review-only.

Wave 1 (no deps, five lanes):
1. Track and normalize the engine regression contract.
2. Introduce and unit-test a pure shared Systolic timing helper.
3. Expose and validate existing TensorCore/GMMA calibration knobs.
4. Make DSE error handling fail closed.
5. Add cross-engine result-contract tests.

Wave 2 (after relevant Wave 1 foundations, five lanes):
6. Align `SystolicEngine` and `MXUModel`.
7. Correct OS-Systolic K-depth and Block-equivalent DMA accounting.
8. Model TensorCore descriptor fragmentation.
9. Restore physical GMMA pipeline and raw-DMA semantics.
10. Add strict all-engine DSE coverage and repository-owned validation entrypoint.

Wave 3 (after all model lanes, two integration lanes):
11. Rebaseline end-to-end CLI and run complete regression/DSE verification.
12. Publish post-fix evidence and update current documentation.

Critical path: 1/2/3 → 6/7/8/9 → 11 → 12 → F1-F4.

### Dependency matrix

| Todo | Depends on | Blocks | Can parallelize with |
|---|---|---|---|
| 1 | - | 6, 7, 8, 9, 11 | 2, 3, 4, 5 |
| 2 | - | 6 | 1, 3, 4, 5 |
| 3 | - | 8, 9 | 1, 2, 4, 5 |
| 4 | - | 10, 11 | 1, 2, 3, 5 |
| 5 | - | 6, 7, 8, 9, 11 | 1, 2, 3, 4 |
| 6 | 1, 2, 5 | 11 | 7, 8, 9, 10 |
| 7 | 1, 5 | 11 | 6, 8, 9, 10 |
| 8 | 1, 3, 5 | 11 | 6, 7, 9, 10 |
| 9 | 1, 3, 5 | 11 | 6, 7, 8, 10 |
| 10 | 4 | 11 | 6, 7, 8, 9 |
| 11 | 1-10 | 12 | - |
| 12 | 11 | F1-F4 | - |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [ ] 1. Track and normalize the engine regression contract
  What to do:
  - Preserve and add the user-provided `sim/tests/test_engines.py` and `sim/tests/test_engine_instantiate.py` to the standalone Arc test surface.
  - Add a shared fixture/config builder so every “same array” comparison uses exactly the same 64x64 dimensions, frequency, precision, DRAM bandwidth and efficiency.
  - Correct `test_os_systolic_decode` text that says 128x128 while `_engine_config()` constructs 64x64.
  - Parameterize Systolic/MXU cases by `(mode, M, op_name, K, N)` so one mismatch cannot hide later shapes.
  - Keep the intended contracts: Systolic exact parity, OS/Block same-band within 10%, TensorCore slower than Block, GMMA raw-DMA floor and bandwidth monotonicity.
  - Do not rebaseline `11.17` or `29.6` yet; those values remain red until Todo 11.
  Parallelization: Can parallel Y | Wave 1 | Blocks 6, 7, 8, 9, 11
  References:
  - `sim/tests/test_engines.py:15` shared base configuration.
  - `sim/tests/test_engines.py:131` TensorCore relative contract.
  - `sim/tests/test_engines.py:213` contradictory OS test description/config.
  - `sim/tests/test_engines.py:274` fail-fast Systolic/MXU loop.
  - `sim/tests/test_engines.py:303` GMMA contracts.
  - `sim/tests/test_engines.py:347` and `:366` stale CLI baselines.
  - `reports/dse-engine-model-bugs-2026-07-27.md:403` bug-to-file table.
  Acceptance criteria:
  - `git ls-files sim/tests/test_engines.py sim/tests/test_engine_instantiate.py` prints both paths.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sim python -m pytest -p no:cacheprovider sim/tests --collect-only -q` exits 0 and lists separate node IDs for every Qwen geometry at M=1, 2 and 128.
  - The red-phase run fails only on the explicitly documented engine/CLI contracts and its exit code is captured.
  QA scenarios:
  - Happy: collect all nodes and save output to `.omo/evidence/task-1-engine-contract-collect.txt`.
  - Failure: run the red suite and use a small parser to assert that every failure node belongs to the approved DSE-001..008 set; save `.omo/evidence/task-1-engine-contract-red.txt`.
  Commit: Y | `test(engine): establish standalone Arc timing contracts` | `sim/tests/test_engines.py`, `sim/tests/test_engine_instantiate.py`, optional `sim/tests/conftest.py`

- [ ] 2. Introduce and unit-test a pure shared Systolic timing helper
  What to do:
  - Add a dependency-free helper under `sim/models/` that returns the canonical per-tile schedule fields used by both `SystolicEngine` and `MXUModel`.
  - Encode current MXU authority exactly:
    - decode/interleaved: `H * (M + 1) + W`;
    - partial prefill `M < H`: `H + W + M`;
    - full prefill: `ceil(M/H) * (H + W + H)`.
  - Centralize tile counts, activation bytes and first-tile/double-buffer overlap rounding so the two callers cannot drift.
  - Add boundary tests for M=1, 2, 3, H-1, H, H+1 and 2H; reject non-positive M/K/N.
  - Do not change either production caller in this todo.
  Parallelization: Can parallel Y | Wave 1 | Blocks 6
  References:
  - `sim/engine/systolic_engine.py:20` duplicate decode schedule.
  - `sim/engine/systolic_engine.py:69` duplicate prefill schedule.
  - `sim/models/mxu.py:87` canonical interleaving formula.
  - `sim/models/mxu.py:140` canonical prefill branches.
  Acceptance criteria:
  - Helper unit tests pass independently.
  - Expected schedules are asserted as exact integers for all boundary values.
  - Invalid dimensions raise typed `ValueError` rather than producing zero/negative cycles.
  QA scenarios:
  - Happy: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=sim python -m pytest -p no:cacheprovider sim/tests/test_systolic_timing.py -q` → 0; evidence `.omo/evidence/task-2-systolic-helper.txt`.
  - Failure: parameterized zero and negative dimensions all raise `ValueError`; evidence `.omo/evidence/task-2-systolic-helper-invalid.txt`.
  Commit: Y | `refactor(timing): centralize systolic schedule formulas` | `sim/models/systolic_timing.py`, `sim/tests/test_systolic_timing.py`

- [ ] 3. Expose and validate existing TensorCore/GMMA calibration knobs
  What to do:
  - Parse the existing `dma.descriptor_overhead_cycles` in TensorCore instead of inventing a class-local setup constant.
  - Move the already-present GMMA value `0.05` to a named configuration field with the existing class constant only as backward-compatible default.
  - Move the already-present TMA overlap value `0.5` to a named configuration field with the existing class constant only as backward-compatible default.
  - Add bounds: descriptor overhead integer ≥0, GMMA pipeline scale `0 < scale <= 1`, TMA overlap `0 <= overlap < 1`.
  - Put the fields in both `sim/config/npu_config.yaml` and `sim/config/design_space.yaml`, with comments distinguishing calibrated assumption from physical bandwidth.
  - Do not introduce a nonzero barrier constant.
  Parallelization: Can parallel Y | Wave 1 | Blocks 8, 9
  References:
  - `sim/config/npu_config.yaml:71` existing descriptor overhead.
  - `sim/config/design_space.yaml:42` existing DMA block and `descriptor_overhead_cycles: 5`.
  - `sim/engine/tensor_core_engine.py:71` per-wave DMA currently ignores descriptors.
  - `sim/engine/gmma_engine.py:39` TMA overlap comments.
  - `sim/engine/gmma_engine.py:46` existing but unused `GMMA_PIPELINE_SCALE=0.05`.
  Acceptance criteria:
  - Default configs parse to descriptor=5 and pipeline scale=0.05.
  - Override configs change the parsed values.
  - Invalid values fail immediately with a field-specific `ValueError`.
  QA scenarios:
  - Happy: focused config tests pass; `.omo/evidence/task-3-calibration-config.txt`.
  - Failure: descriptor=-1, scale=0/1.01 and overlap=1 each fail closed; `.omo/evidence/task-3-calibration-invalid.txt`.
  Commit: Y | `feat(config): expose engine calibration parameters` | `sim/config/npu_config.yaml`, `sim/config/design_space.yaml`, `sim/engine/tensor_core_engine.py`, `sim/engine/gmma_engine.py`, focused tests

- [ ] 4. Make DSE error handling fail closed
  What to do:
  - Replace the bare catch-and-pass loop with structured accounting for generated, evaluated, filtered, valid and errored configurations.
  - Default CLI behavior returns nonzero on any evaluation exception and prints the engine type, dimensions, memory mode and exception to stderr.
  - Add explicit `--allow-partial` for exploratory scans; it may return results but must still report each error and a nonzero error count in output metadata.
  - Preserve area filtering as a distinct `filtered` count, not an error.
  - Ensure an empty result set always fails.
  Parallelization: Can parallel Y | Wave 1 | Blocks 10, 11
  References:
  - `sim/design_space_explorer.py:541` CLI parser.
  - `sim/design_space_explorer.py:594` generated-config count.
  - `sim/design_space_explorer.py:603` current silent exception loop.
  - `README.md:251` current quick-DSE command.
  Acceptance criteria:
  - Injecting one evaluator exception makes default mode exit nonzero and name the failing config.
  - The same injection under `--allow-partial` preserves valid results while reporting `errors=1`.
  - Normal quick DSE reports `errors=0` and exits 0.
  QA scenarios:
  - Happy: quick DSE output and exit code → `.omo/evidence/task-4-dse-strict-happy.txt`.
  - Failure: monkeypatched evaluator exception in pytest → `.omo/evidence/task-4-dse-strict-failure.txt`.
  Commit: Y | `fix(dse): fail closed on configuration errors` | `sim/design_space_explorer.py`, `sim/tests/test_dse_strict.py`

- [ ] 5. Add cross-engine result-contract tests
  What to do:
  - Add parameterized tests for all factory engines covering positive totals, finite utilization, correct ops/bytes, valid bottleneck labels and required diagnostic fields.
  - For OS, TensorCore and GMMA require explicit raw compute/DMA diagnostic fields so critical-path decisions are inspectable.
  - Test `estimate()` and `estimate_weight_cache_pair()` separately; do not assume `compute_cycles + dma_cycles == total_cycles` under overlap.
  - Require `total_cycles >= raw_dma_cycles` for GMMA and any other engine claiming a raw physical DMA floor.
  - Own only `sim/tests/test_engine_result_contract.py`; do not edit Todo 1's shared fixture files while the lanes run in parallel.
  Parallelization: Can parallel Y | Wave 1 | Blocks 6, 7, 8, 9, 11
  References:
  - `sim/engine/mac_engine.py:15` `EngineResult` schema.
  - `sim/engine/mac_engine.py:127` factory registration.
  - `sim/tests/test_engine_instantiate.py:15` current instantiate-only coverage.
  - `sim/engine/gmma_engine.py:108` missing single-estimate diagnostics versus pair diagnostics at `:168`.
  Acceptance criteria:
  - Tests collect every engine type returned by `create_engine`.
  - Each result exposes enough fields to recompute its bottleneck decision.
  - Current missing GMMA fields reproduce as red before Todo 9.
  QA scenarios:
  - Happy: post-fix contract suite passes; `.omo/evidence/task-5-engine-result-contract.txt`.
  - Failure: synthetic malformed result fixture is rejected; `.omo/evidence/task-5-engine-result-contract-invalid.txt`.
  Commit: Y | `test(engine): define inspectable result contracts` | `sim/tests/test_engine_result_contract.py`, minimal shared fixtures

- [ ] 6. Align `SystolicEngine` and `MXUModel`
  What to do:
  - Route both decode and prefill through the helper from Todo 2.
  - Preserve their public result types while deriving total cycles, tile counts, bytes and utilization from the same schedule.
  - Ensure mode dispatch agrees at M=1, 2, 3 and boundary values; remove contradictory comments such as “byte-identical” beside duplicate formulas.
  - Keep MXUModel as semantic authority; do not redesign its prefill dataflow in this bug-fix.
  Parallelization: Can parallel Y | Wave 2 | Blocks 11 | Blocked by 1, 2, 5
  References:
  - `sim/engine/systolic_engine.py:21` promised byte-identical decode.
  - `sim/engine/systolic_engine.py:70` promised byte-identical prefill.
  - `sim/models/mxu.py:91` authoritative decode formula.
  - `sim/models/mxu.py:163` authoritative prefill branch.
  - `sim/tests/test_engines.py:274` parity regression.
  Acceptance criteria:
  - Exact total-cycle parity for all seven Qwen geometries at M=1, 2, 128.
  - Exact parity for helper boundary matrix M=3, H-1, H, H+1, 2H.
  - No duplicated pipeline formula remains in the two callers.
  QA scenarios:
  - Happy: focused parity tests → `.omo/evidence/task-6-systolic-mxu-parity.txt`.
  - Failure: temporarily varied H/W configs still use helper and remain equal; `.omo/evidence/task-6-systolic-mxu-varied-array.txt`.
  Commit: Y | `fix(systolic): share MXU timing schedule` | `sim/engine/systolic_engine.py`, `sim/models/mxu.py`, focused tests

- [ ] 7. Correct OS-Systolic K-depth and Block-equivalent DMA accounting
  What to do:
  - Add `H` K-reduction depth to single and weight-cache-pair compute paths.
  - Replace raw per-tile external-DRAM accounting with the same aggregate weight/activation bytes, `_dram_eff_for_bytes` and roofline overlap used by Block for equivalent 64x64 work.
  - Keep OS-specific activation reuse diagnostics for the pair path.
  - Report `k_reduction_cycles`, raw DMA cycles, total compute and selected bottleneck.
  - Do not add an arbitrary area penalty in the timing engine.
  Parallelization: Can parallel Y | Wave 2 | Blocks 11 | Blocked by 1, 5
  References:
  - `sim/engine/os_systolic_engine.py:47` current tiling.
  - `sim/engine/os_systolic_engine.py:51` current raw DMA path.
  - `sim/engine/os_systolic_engine.py:56` missing H in compute.
  - `sim/engine/os_systolic_engine.py:103` pair path.
  - `sim/engine/block_engine.py:99` reference external-DRAM accounting.
  - `sim/engine/ppa_model.py:76` separate area-model authority.
  Acceptance criteria:
  - At 64x64 M=1 FFN_down, OS and Block total cycles differ by ≤10%.
  - OS details show `k_reduction_cycles == H`.
  - Single and pair paths report aggregate transferred bytes consistent with their documented activation-reuse behavior.
  - Pair path remains finite and exposes its activation-reuse saving without introducing an unmodeled area penalty.
  QA scenarios:
  - Happy: OS/Block same-band and pair tests → `.omo/evidence/task-7-os-systolic.txt`.
  - Failure: low/high bandwidth sweep shows bottleneck transitions without violating raw bytes → `.omo/evidence/task-7-os-systolic-bw-sweep.json`.
  Commit: Y | `fix(os-systolic): account for reduction depth and physical DMA` | `sim/engine/os_systolic_engine.py`, focused tests

- [ ] 8. Model TensorCore descriptor fragmentation
  What to do:
  - Add per-wave descriptor cost as `active_tcs * descriptor_overhead_cycles`, using the configured 5-cycle value.
  - Use `active_tcs` for a partial final wave instead of charging all TCs.
  - Keep compute and DMA overlap, but expose payload DMA, descriptor cycles, active TC count and waves separately.
  - Apply the same accounting to weight-cache-pair estimates.
  - Do not tune the descriptor value after seeing ranking results.
  Parallelization: Can parallel Y | Wave 2 | Blocks 11 | Blocked by 1, 3, 5
  References:
  - `sim/engine/tensor_core_engine.py:54` invocation count.
  - `sim/engine/tensor_core_engine.py:68` TC/wave count.
  - `sim/engine/tensor_core_engine.py:71` missing descriptor overhead.
  - `sim/config/design_space.yaml:45` existing 5-cycle calibration.
  - `reports/dse-engine-model-bugs-2026-07-27.md:208` fragmentation diagnosis; ignore its instruction to tune constants until a test passes.
  Acceptance criteria:
  - Default 64x64 FFN_down TensorCore total cycles exceed Block without changing descriptor=5.
  - Setting descriptor=0 reproduces the pre-fix payload-only behavior within rounding.
  - Final partial wave charges only active TCs.
  QA scenarios:
  - Happy: default and descriptor=0 comparison → `.omo/evidence/task-8-tensor-core.json`.
  - Failure: descriptor<0 rejected during construction → `.omo/evidence/task-8-tensor-core-invalid.txt`.
  Commit: Y | `fix(tensor-core): account for descriptor fragmentation` | `sim/engine/tensor_core_engine.py`, focused tests

- [ ] 9. Restore physical GMMA pipeline and raw-DMA semantics
  What to do:
  - Compute per-tile latency exactly as `max(1, ceil((H + M + W) * pipeline_scale))`.
  - Compute and expose raw DMA, theoretically hidden DMA and exposed DMA separately.
  - Select `total_cycles = max(total_compute, raw_dma_cycles)` so TMA cannot beat the physical byte-time ceiling; use overlap fields only for diagnostics.
  - Make single and weight-cache-pair results expose the same detail keys.
  - Validate bandwidth monotonicity across LPDDR5 and HBM2e.
  Parallelization: Can parallel Y | Wave 2 | Blocks 11 | Blocked by 1, 3, 5
  References:
  - `sim/engine/gmma_engine.py:39` intended TMA semantics.
  - `sim/engine/gmma_engine.py:46` unused pipeline scale.
  - `sim/engine/gmma_engine.py:59` current unscaled compute.
  - `sim/engine/gmma_engine.py:82` total DMA.
  - `sim/engine/gmma_engine.py:84` current overlap path.
  - `docs/NPU_Engines_Architecture_Guide.md:217` LPDDR/GMMA contract.
  Acceptance criteria:
  - LPDDR5 FFN_down is raw-DMA-bound and finite.
  - HBM2e throughput is >2x LPDDR5 for the approved representative shape.
  - `total_cycles >= ceil((transferred_weight_bytes + activation_bytes)/eff_bw)` for every tested bandwidth.
  - Single and pair details both include `raw_dma_cycles`, `tma_hidden_dma`, `tma_exposed_dma`, `per_tile_compute`, `pipeline_scale`.
  QA scenarios:
  - Happy: bandwidth sweep JSON → `.omo/evidence/task-9-gmma-bw-sweep.json`.
  - Failure: an extreme overlap value cannot lower total below raw DMA; `.omo/evidence/task-9-gmma-floor.txt`.
  Commit: Y | `fix(gmma): enforce pipeline scale and raw bandwidth floor` | `sim/engine/gmma_engine.py`, focused tests

- [ ] 10. Add strict all-engine DSE coverage and a repository-owned validation entrypoint
  What to do:
  - Add a root `pytest.ini` so `python -m pytest` discovers `sim/tests` and imports `sim` without ad-hoc user setup.
  - Add DSE tests that confirm the full engine set generated by `generate_configs(False)` matches the factory-supported set intended for DSE.
  - Keep quick mode small, but state exactly which engines it covers; add a deterministic all-engine smoke command for validation.
  - Add a validation-surface test that requires `pytest.ini`, tracked engine tests and the strict DSE test to exist in the standalone repository.
  - Do not create CI credentials or modify external repository settings.
  Parallelization: Can parallel Y | Wave 2 | Blocks 11 | Blocked by 4
  References:
  - `sim/design_space_explorer.py:163` config generation.
  - `sim/design_space_explorer.py:170` engine list.
  - `sim/engine/mac_engine.py:127` factory set.
  - `README.md:251` current quick command.
  - `docs/progress-summary-2026-06.html:834` stale source-repo pytest claim.
  Acceptance criteria:
  - From repository root, `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider --collect-only -q` exits 0.
  - The DSE/full-engine set comparison is exact or explicitly allowlists factory-only engines with a reason.
  - Removing any required validation asset makes the validation-surface test fail.
  QA scenarios:
  - Happy: root-level collection and all-engine smoke → `.omo/evidence/task-10-standalone-validation.txt`.
  - Failure: pytest tmp-path fixture simulates a missing required asset → `.omo/evidence/task-10-migration-gap.txt`.
  Commit: Y | `test(dse): enforce standalone all-engine coverage` | `pytest.ini`, DSE/validation-surface tests, minimal README command update

- [ ] 11. Rebaseline end-to-end CLI and run complete regression/DSE verification
  What to do:
  - After Todos 6-10, run `npu_sim.py --json` for Block and `--engine systolic --json`; record the new outputs before changing baseline assertions.
  - Replace stale expected values only with values captured from that exact repaired commit and config; keep ±1% tolerance.
  - Run the complete standalone test suite, strict quick DSE, deterministic all-engine DSE smoke and full DSE.
  - Run all seven engines through the same FFN_down shape and record total/compute/raw-DMA/bottleneck/ranking.
  - Fail if any DSE error/skip is nonzero or if any required command did not actually execute.
  Parallelization: Can parallel N | Wave 3 | Blocks 12 | Blocked by 1-10
  References:
  - `sim/tests/test_engines.py:347` Systolic CLI baseline.
  - `sim/tests/test_engines.py:366` Block CLI baseline.
  - `sim/npu_sim.py:475` CLI.
  - `sim/npu_sim.py:594` JSON schema.
  - `sim/design_space_explorer.py:541` DSE CLI.
  - `reports/dse-engine-model-bugs-2026-07-27.md:436` dependency on remeasurement.
  Acceptance criteria:
  - `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q` exits 0 with zero failures.
  - Strict quick/full DSE exit 0 with `errors=0`; full output contains every intended engine.
  - CLI baseline expected values equal captured JSON values from the same commit.
  - The original 8 failure nodes all pass.
  QA scenarios:
  - Happy: full command manifest and outputs → `.omo/evidence/task-11-verification-manifest.json`.
  - Failure: manifest verifier rejects missing command, nonzero exit, dirty commit mismatch or absent engine → `.omo/evidence/task-11-manifest-negative.txt`.
  Commit: Y | `test(cli): rebaseline repaired engine models` | `sim/tests/test_engines.py`, generated verification helper if required

- [ ] 12. Publish post-fix evidence and update current documentation
  What to do:
  - Create `reports/dse-engine-model-bugs-postfix-2026-07-27.md` containing before/after values for all eight bug IDs, exact commands, commit/config hashes, test totals, DSE counts and final three-scenario recommendation comparison.
  - Update `docs/NPU_Engines_Architecture_Guide.md` and current README validation instructions/numbers from Todo 11 evidence.
  - Update `docs/NPU硬件详细架构设计v0.1.md` only where it presents the old values as current signed-off facts; preserve its version history and label recalibrated values.
  - State that OS has no diagonal fill/drain but still has K reduction; TensorCore includes descriptor cost; GMMA TMA does not bypass raw bandwidth.
  - Do not modify `reports/dse-engine-model-bugs-2026-07-27.md` or other dated historical reports.
  Parallelization: Can parallel N | Wave 3 | Blocks F1-F4 | Blocked by 11
  References:
  - `docs/NPU_Engines_Architecture_Guide.md:14` stale ranking summary.
  - `docs/NPU_Engines_Architecture_Guide.md:73` stale OS zero-overhead wording.
  - `docs/NPU_Engines_Architecture_Guide.md:133` TensorCore section.
  - `docs/NPU_Engines_Architecture_Guide.md:191` GMMA section.
  - `docs/NPU硬件详细架构设计v0.1.md:97` and `:695` stale current performance tables.
  - `reports/dse-engine-model-bugs-2026-07-27.md:458` prior recommendation impact.
  Acceptance criteria:
  - Every BUG-DSE-001..008 appears exactly once in the post-fix result table with status and evidence link/path.
  - Every current numeric claim in modified docs matches Todo 11 machine output.
  - `sha256sum reports/dse-engine-model-bugs-2026-07-27.md` still equals `61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3`.
  - No “210/210” claim remains unless reproduced by the current standalone suite and labeled with commit.
  QA scenarios:
  - Happy: report consistency checker cross-validates JSON numbers against Markdown tables → `.omo/evidence/task-12-doc-consistency.txt`.
  - Failure: stale-number scan plus explicit allowlist for historical context → `.omo/evidence/task-12-stale-number-audit.txt`.
  Commit: Y | `docs(dse): publish repaired engine-model verification` | new post-fix report, current guide, current architecture doc, README

## Final verification wave (after ALL todos)
> Runs in parallel. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.

- [ ] F1. Plan compliance audit
  - Verify all 12 todos and BUG-DSE-001..008 have concrete evidence.
  - Reject missing commands, self-reported success, unexecuted full DSE or post-fix values not tied to the repaired commit.
  - Evidence: `.omo/evidence/final-f1-plan-compliance.md`.

- [ ] F2. Code quality review
  - Review formula ownership, rounding, result schema, configuration validation and error paths.
  - Run Python diagnostics/compile checks and focused tests on every changed module.
  - Reject duplicated Systolic formulas, unused calibration constants, silent exceptions or magic tuning.
  - Evidence: `.omo/evidence/final-f2-code-quality.md`.

- [ ] F3. Real manual QA
  - Actually invoke the seven-engine FFN_down matrix, Block/Systolic `npu_sim --json`, strict quick DSE and strict full DSE.
  - Parse outputs and verify monotonic bandwidth behavior, physical DMA floors, expected rankings and zero DSE errors.
  - Evidence: `.omo/evidence/final-f3-manual-qa.json`.

- [ ] F4. Scope fidelity
  - Compare final diff to the scope allowlist.
  - Verify unrelated `.omo/ultraresearch/20260723-vla-models/sources/` is untouched.
  - Verify the original user report still has SHA-256 `61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3` and no adjacent CaduceusCore file was modified.
  - Confirm standalone Arc contains all tests/configs used to claim success.
  - Evidence: `.omo/evidence/final-f4-scope-fidelity.md`.

## Commit strategy
- One atomic commit per todo as listed; do not amend and do not combine model fixes across engines.
- Before each commit, stage explicit file paths only; never use `git add .` or stage unrelated untracked files.
- The original report remains unmodified and unstaged unless the user separately asks to track it.
- Generated `.omo/evidence/` is retained for review but committed only if the repository's existing convention explicitly tracks evidence; otherwise it remains a local execution artifact.
- Suggested order follows dependency order: 1-5, 6-10, 11, 12.

## Success criteria
- Standalone root test command exits 0 and all original 8 failure nodes pass.
- Systolic/MXU parity holds across Qwen shapes and boundary M values.
- OS includes K reduction and stays within 10% of Block under the approved same-config case without timing-engine area penalties.
- TensorCore is slower than Block using the existing descriptor=5 calibration, with no tuned magic constant.
- GMMA is bandwidth-monotonic and never faster than raw transferred-byte time.
- Strict quick/full DSE report zero errors; full coverage includes every intended engine.
- CLI baselines and current documentation match the same repaired commit/config evidence.
- Original dated report and unrelated dirty-worktree paths are unchanged.
- F1, F2, F3 and F4 all return APPROVE, then the executor waits for explicit user acceptance before declaring the bug-fix complete.
