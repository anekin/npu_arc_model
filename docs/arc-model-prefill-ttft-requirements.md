# Arc Model DSE Prefill/TTFT 建模需求报告

**来源**: CaduceusCore — Func Model 性能验证规格 Gate 1b（DSE↔Func Model TTFT 一致性闭环）
**目标仓库**: `npu_arc_model`（/home/prj/zhengs/caduceuscore/npu_arc_model）
**日期**: 2026-08-13
**状态**: 待 npu_arc_model 侧 agent 实施；CaduceusCore 侧行动项见 §8

---

## 1. 需求背景

### 1.1 消费者与用途

CaduceusCore 的 Func Model 性能验证规格（`.omo/notes/func-model-perf-verification-spec.md`）在 Gate 1 建立 DSE 一致性门禁：

- **Gate 1a (TPS)**：已闭环。canonical 公式对齐 BlockEngine broadcast 模型（`M*(H+4)`），canonical TPS 30.75 vs DSE 25.1 = 1.22×，PASS。
- **Gate 1b (TTFT)**：**当前无法真正闭环**，因为 Arc Model 的 DSE 从未真实建模 prefill——trace 固定 batch_m=1，`--batch-m` 被限制在 `[1, 2]`，无 TTFT 输出。规格被迫标注 "DSE TTFT 未真正建模 prefill (trace 固定 M=1)，标记 DSE TTFT 不可用作验证目标"。

本报告即向 Arc Model 提出补齐该能力的需求，使 Gate 1b 获得真实可用的 DSE TTFT 验证目标。

### 1.2 当前状态（npu_arc_model，实测于 2026-08-13）

| 位置 | 现状 |
|------|------|
| `sim/design_space_explorer.py` L31, L59 | 模块级 `_LLM_TRACE = generate_trace_from_spec("qwen2.5-3b", batch_m=1)`，import 时固定 |
| `sim/design_space_explorer.py` L111-151 | `simulate_layer()` 复用模块级 trace，无 `simulate_prefill` |
| `sim/design_space_explorer.py` L933-935 | `--batch-m type=int, choices=[1, 2]` |
| `sim/design_space_explorer.py` L187-190 | quick 模式 dims `[(128,128),(128,256),(256,256)]`，无 64×64（full 模式含） |
| `sim/engine/block_engine.py` L53 | `estimate(M, K, N, weight_preloaded)` 引擎层对任意 M 正常（问题仅在 DSE trace 层） |
| `docs/arc_vs_func.md` L141-143 | stale 断言："prefill latency ❌ 未实现 / TTFT ❌ 未实现 / Arc 只算 decode M=1" |
| `docs/arc_vs_func.md` L156-157 | roadmap 已列待办："增加 `simulate_prefill()` 和 benchmark 入口"、"输出 TTFT、TPS、cycle breakdown" |

### 1.3 参考实现（CaduceusCore commit `e325776`，仅作对照，非权威）

CaduceusCore 旧 DSE 副本上已有一版完整参考实现（拆分前落错仓库），其**设计意图**可直接参考，但**实现必须适配 npu_arc_model 现有架构**（engine registry、contracts/、result v2 identity），不可照搬：

- `_LLM_TRACE` 全局 → `_DEFAULT_LLM_SPEC`/`_MODEL_ALIAS`/`_PREFILL_BATCH_M`；`simulate_layer(batch_m=None)` 按需 `generate_trace_from_spec` 重新生成 trace（None→1 保持 decode 语义）
- 循环体抽为 `_simulate_ops(config, ops, batch_m)`，`simulate_prefill(config, batch_m, model_alias)` 复用
- `ttft_ms_from_prefill(prefill_cycles, num_layers, freq_mhz)` = `round(cycles × layers / freq_mhz / 1000, 2)`
- CLI `--batch-m` 放宽为 `type=int` + `>=1` 校验
- `evaluate_config` 输出 `ttft_ms`；quick dims 增加 `(64, 64)`
- 已知缺点（新 repo 实现时应避免）：ttft_ms 塞进 config dict 而非结果结构体字段；`generate_trace_from_spec` 副作用写全局 `_KV_HEADS/_HEAD_DIM` 造成顺序依赖

---

## 2. 消费者需求（为什么 Func Model 需要）

Gate 1b 的判定需要从 DSE 拿到：

1. **Block 64×64 @ 1GHz, LPDDR5-64b, INT4, weight_cache 配置**下的 prefill TTFT 目标值（ms），覆盖两个规模：**M=128** 与 **M=2000**（Qwen2.5-3B，36 层）。
2. **TTFT 定义**：`prefill_layer_cycles × num_layers / freq_mhz` [ms]，**不含首 token decode**（与 Func Model uncertainty-kpis 的 `prefill_ms` 口径对齐）。
3. **可复现的证据**：JSON 输出含配置标识 + `ttft_ms`，可提交入库、可哈希校验。
4. **PASS 判定区间**：Func Model TTFT ∈ [0.5× DSE_TTFT, 2.0× DSE_TTFT]（当前 Func Model 实测：M=128 → 3,911.05 ms；M=2000 → 63,924.19 ms）。

---

## 3. 功能需求（FR）

- **FR-1 trace 按 batch_m 生成**：消除固定 M=1 的模块级 trace；`simulate_layer(batch_m=None)` 默认 1（decode 向后兼容），`batch_m>1` 时 attention/FFN 投影全部按 batch 生成（参考 `generate_trace_from_spec` 的 `m_attn=batch_m`、`m_ffn=batch_m if batch_m>1 else 1` 语义）。
- **FR-2 `simulate_prefill(config, batch_m, model_alias)`**：返回 per-layer prefill cycles（复用引擎 estimate 路径；KV prefill 写回 cost 暂按 0 处理为已知简化，需在代码注释与文档中显式声明）。
- **FR-3 TTFT 换算**：提供 `ttft_ms_from_prefill` 或等价函数；单位换算**必须走 `contracts.units`**，禁止裸公式。
- **FR-4 CLI `--batch-m`**：放宽为 `type=int` 且校验 `>=1`；`--batch-m` 仅驱动 prefill/TTFT 指标，decode `tok_s` 保持 batch_m=1 语义（help 文案明确说明）。
- **FR-5 结果输出 `ttft_ms`**：在结果 schema（遵循 result v2 identity 契约 + legacy 投影）中携带 `ttft_ms` 字段，非 CV 模式所有结果条目均含；CV 模式按现有约定处理（0.0 或省略，保持 schema 一致）。
- **FR-6 Block 64×64 可达性**：保证 Gate 1b 所需配置能通过 CLI 快速获取——quick dims 增加 `(64, 64)`，或提供显式获取该配置的等价入口。
- **FR-7 向后兼容**：默认（无 `--batch-m`）运行输出的 tok_s/area/power 与实施前逐配置一致；现有 test suite（test_dse_*、test_design_space_explorer.py 等）全部保持绿色。

---

## 4. 非功能需求（NFR）

- **NFR-1 可复现**：相同输入输出 sha256 稳定（已有 `--replay`/`--seed` 机制，prefill 输出纳入其中）。
- **NFR-2 证据入库**：TTFT 证据 JSON **提交到 git**（勿被 .gitignore 排除——旧实现踩过此坑）。
- **NFR-3 文档同步**：清除 `docs/arc_vs_func.md` L141-143 的 stale 断言（prefill/TTFT "未实现"）；勾除/更新 L156-157 roadmap 条目；在 README 的 DSE 能力矩阵中补充 prefill TTFT 支持说明。
- **NFR-4 测试纪律**：TDD（RED→GREEN→mutation）；新增测试覆盖 batch_m 单调性、TTFT 公式、CLI 参数校验、默认语义向后兼容。
- **NFR-5 架构遵循**：结果走 EngineResult/result v2 契约；参数经 typed config 校验；不引入裸全局状态（参考实现的 `_KV_HEADS/_HEAD_DIM` 副作用顺序依赖在新实现中应消除——KV 几何作为参数传递）。
- **NFR-6 与现有活跃工作共存**：当前有进行中的 boulder 工作（wmma-gmma-pe-recalibration）；本需求实施应避免与其文件冲突，必要时在其之后排期。

---

## 5. 参考数值（仅作对照，不要求一致）

CaduceusCore 参考实现（旧公式栈）产出：

| Prefill 规模 | DSE TTFT（参考实现） | Func Model 实测 | 比值 |
|------|------|------|------|
| M=128 | 2,649.49 ms | 3,911.05 ms | 1.48× |
| M=2000 | 41,398.27 ms | 63,924.19 ms | 1.54× |

> ⚠️ npu_arc_model 的引擎公式已修正（systolic prefill 公式、os_systolic K-reduction、物理 DMA floor 等），**重算后的数值预期会不同**。验收不要求与参考值一致，只要求：(a) 模型语义正确（compute-bound、随 M 线性）；(b) 与 Func Model 实测的比值落在 [0.5×, 2.0×]；(c) 数值可复现、可溯源。

已知结构差异（属正常，需在文档中说明）：DSE 用 7-op layer（Q/K/V/O + FFN×3），Func Model 用 17-op DAG（含 SFU/Vector/并行假设），两者比值 1.48–1.54× 即源于此。

---

## 6. 验收标准（Definition of Done）

- **AC-1**：新 repo pytest 全绿（含新增 prefill TTFT 测试）。
- **AC-2**：`--batch-m 128` / `--batch-m 2000` 输出含 Block 64×64 配置的 `ttft_ms`，且默认运行（无 `--batch-m`）与实施前结果逐配置一致。
- **AC-3**：`docs/arc_vs_func.md` stale 断言清除、roadmap 更新；README 能力矩阵更新。
- **AC-4**：TTFT 证据 JSON 提交入库（git 可见）。
- **AC-5**：产出 §7 的回流产物。

---

## 7. 交付给 CaduceusCore 的回流产物

1. **DSE TTFT 目标值**：Block 64×64 @ 1GHz LPDDR5-64b INT4（WC）的 M=128 / M=2000 TTFT（ms），以及 Func Model 比值判定（[0.5×, 2.0×]）。
2. **证据文件路径 + SHA-256**（npu_arc_model 仓库内 git 提交路径）。
3. **实施说明**：模型假设与简化（KV prefill write=0、7-op vs 17-op 差异）、与参考实现的数值差异原因。
4. （可选）arc_vs_func.md 更新后的片段引用。

CaduceusCore 侧收到后执行 §8-A2。

---

## 8. CaduceusCore 侧行动项（本仓库，另行安排执行）

- **A1 冻结标记旧 DSE 副本**（用户已确认：冻结、暂不删除）：在 `sim/design_space_explorer.py` 等 Arc 相关文件头部加 deprecation 注释，声明 Arc Model 已迁出至 `npu_arc_model`、以该仓库为权威、本副本仅供 Func Model signoff 参考、待移植完成后移除。
- **A2 收到 §7 回流产物后**：更新 `.omo/notes/func-model-perf-verification-spec.md` Gate 1b 的 DSE 目标值与证据引用（替换当前基于旧副本的 2,649.49/41,398.27 ms）；同步 `reports/func-model-perf-verification-report.md` §3.5。
- **A3（可选）**：修复上轮 review 的流程完整性问题——F1/F3 引用脚本缺失（`scripts/audit_plan_compliance.py`、`scripts/real_qa_check.py`）、被引用证据 JSON 未入库（`git add -f` 或改内联引用）、T1/T3/T4 mutation 回填。

---

## 9. 边界与不在此范围

- 不要求 npu_arc_model 修改引擎层 GEMM 公式（block_engine 已正确处理任意 M）。
- 不要求 KV prefill 写回建模（已知简化，未来随 arc-model-v3 规划处理）。
- 不要求 CV 模式 TTFT。
- 不要求两仓库代码同步机制（当前为报告/证据流，非代码流）。
- 本报告不涉及 CaduceusCore Func Model 侧的任何模型修改。
