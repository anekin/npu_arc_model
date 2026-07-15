# Arc Model Engine 整理、准入测试与 Signoff 报告

日期：2026-07-15
Arc Model：v3.7-weight-cache-variants
分支：feat/scenario-driven-dse
测试范围：10 个正式 DSE Engine、Framework 计算主路径、场景 A、场景 A Agent

## 1. 结论

本轮已经把 Engine 从按名称硬编码的“正式/研究”二元分类，改为机器可读的 maturity/evidence 体系：

- 所有 M1 以上 Engine 进入 Raw Exploration；
- M2 以上 Engine 进入 Comparison-ready Pareto；
- M3/M4 Engine 才能进入 Product-qualified Shortlist；
- 成熟度不修改原始 TPS、TTFT、面积或功耗，也不阻止研究架构出现在统一 Engine 对比表中。

测试结论：

- Engine admission audit：10/10 Engine 通过；
- 自动化回归：125 passed；
- canonical 计算主路径：语句覆盖率 97%，分支覆盖率 85%；
- 三次 quick DSE 输出 SHA-256 完全一致；
- 场景 A：1965/1965 配置有效，存在 M2 comparison-ready 候选，但没有 M3 product-qualified 候选；
- Agent：1980/1980 配置有效，没有任何架构同时满足 Decode TPS 和 TTFT 硬约束；
- 当前没有 Engine 达到 M3/M4，因此任何产品架构都尚未完成正式产品 Signoff。

Framework 计算主路径达到本轮测试门槛，但完整 Framework Signoff 仍为 **CONDITIONAL PASS**：CLI orchestration、CV/legacy 支撑代码没有达到同等覆盖率，本轮也只在 Windows/Python 3.12 环境执行。

## 2. Engine Inventory 与证据等级

| Engine | Maturity | Performance evidence | PPA evidence | Admission | DSE 资格 |
|---|---|---|---|---|---|
| systolic | M2 | RTL module calibrated；DSE 系统模型仍含 DMA/DRAM | analytical proxy | PASS | Raw + Comparison |
| os_systolic | M2 | dataflow analytical、公开模型交叉检查 | analytical proxy | PASS | Raw + Comparison |
| block | M2 | architecture analytical cross-check | analytical proxy | PASS | Raw + Comparison |
| block_fused_attention | M1 | FSA-inspired analytical | analytical proxy | PASS | Raw only |
| os_systolic_fused_attention | M1 | FSA-inspired analytical | analytical proxy | PASS | Raw only |
| tensor_core | M1 | architecture analytical | analytical proxy | PASS | Raw only |
| wmma | M1 | architecture analytical | analytical proxy | PASS | Raw only |
| gmma | M1 | architecture analytical | analytical proxy | PASS | Raw only |
| input_stationary | M1 | dataflow analytical | analytical proxy | PASS | Raw only |
| fsa | M1 whole engine；Attention component M2 | upstream schedule + system extrapolation | paper array proxy | PASS | Raw only |

每个 Engine 的完整 scope、fallback、不确定度、证据来源和 known gaps 保存在 `sim/config/engine_manifests.yaml`。当前 M2 只表示“可进行架构级公平比较”，不表示完成产品校准。

## 3. 本轮实现

### 3.1 Engine evidence framework

- 新增统一 Engine manifest 和严格 schema validator；
- manifest 必须声明 performance/PPA/functional/system 四个证据轴；
- maturity 升级受到最低证据等级约束，不能只修改等级字符串；
- factory、manifest 和完整 DSE 搜索集合必须完全一致；
- DSE point/provenance 记录 maturity、manifest hash、calibration tier 和三层资格。

### 3.2 三层 DSE 输出

JSON 新增：

- `raw_top_results` / `raw_recommended`；
- `top_results` / `recommended`：M2 comparison-ready；
- `product_top_results` / `product_recommended`：M3/M4；
- `raw_feasible`、`feasible`、`product_feasible`；
- 每个 Engine 的 maturity、comparison/product eligibility。

`engine_comparison` 始终按原始场景指标排序，不使用成熟度惩罚。

### 3.3 测试中发现并修正的问题

正式缺陷状态、根因、影响范围和关闭证据见 [`docs/bug-tracker.md`](../docs/bug-tracker.md)。

1. **ARC-BUG-001 — GMMA physical DMA ceiling**：旧模型把 TMA overlap 直接用于缩短物理 DMA 时间，可能令 kernel 总周期低于实际传输周期。现改为 compute 与完整 DMA 的 roofline 上限。
2. **ARC-BUG-002 — GMMA weight-cache pair**：合并 Gate/Up 慢于两个独立 GEMM 时，scheduler 现在显式 fallback，保证优化不产生性能倒退。
3. **ARC-BUG-003 — EngineResult.ops 口径**：Systolic、Tensor Core、WMMA、GMMA、IS 原先记录 MAC 数，Block/OS/FSA 记录 operation 数。现统一为 `M×K×N×ops_per_mac`。
4. **ARC-BUG-004 — Windows CLI 编码**：GBK stdout 无法输出状态符号导致搜索前崩溃。CLI 现在显式使用 UTF-8 输出并以 replacement 处理不支持字符。
5. **ARC-BUG-005 — WC 硬件成本与报告**：WC ON/OFF 现在作为独立硬件变体，分别计入性能、面积和功耗。
6. **ARC-BUG-006 — Systolic WC 单调性**：WC pair 较慢时回退到两个独立 GEMM，防止启用可选硬件后性能倒退。

## 4. 测试矩阵与结果

| 测试层 | 内容 | 结果 |
|---|---|---|
| T0 Schema | manifest root、字段、证据轴、maturity、uncertainty、scope、fallback 的正常与拒绝路径 | PASS |
| T1 Unit | Engine factory、周期/ops/利用率/PPA、约束和报告 | PASS |
| T2 Property | bandwidth/preload 非倒退、pair scheduler 非倒退、面积单调、物理上限 | PASS |
| T3 Microbench | 10 Engine × Decode/Prefill/Agent append 三组 shape | 30/30 PASS |
| T4 Workload | 每个 Engine 完成 Qwen2.5-3B Decode/Prefill smoke | 10/10 PASS |
| T5 DSE | 搜索集合、三层资格、raw ranking、provenance | PASS |
| T6 Scenario | 场景 A、Agent 完整 DSE | 完成；结论见第 6、7 节 |
| T7 Regression | 125 用例、覆盖率、三次 quick DSE hash | PASS/Conditional |

Engine admission 的每个 Engine 均通过以下十项检查：factory identity、三组 physical microbench、bandwidth non-regression、preload non-regression、pair scheduler non-regression、PPA positive、area monotonic、DSE candidate evaluation。

## 5. 覆盖率与可复现性

全量测试结果：

```text
125 passed
```

canonical 计算主路径排除 compiler、ISA、multicore、timeline 和 CLI orchestration 后：

```text
statement coverage: 97%  (1306 / 1347)
branch coverage:    85%  (203 / 238)
combined coverage:  95%
```

对整个被插桩集合计入未纳入当前 DSE 主路径的 compiler/ISA/multicore/timeline 以及 CLI 主函数后，总覆盖率为 60%。因此不能把 97% 解读为整个仓库已完成 Signoff。

v3.7 正式场景 A、Agent 及 75%/90% 带宽效率角已独立运行；最新 summary 为每个源结果保存 SHA-256。详细证据见 `sim/results/lpddr5_latest_dse_summary_2026-07-15.json`。

## 6. 场景 A 完整 DSE

搜索：1965 配置，1965 valid，invalid_configs=0。
约束：756 raw feasible，334 comparison-ready feasible，0 product-qualified。

| Engine | M | Status | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---|---:|---|---:|---:|---:|---:|---:|
| os_systolic | M2 | PASS/MET | 20.03 | 667.2 | 241.8ms | 42.8mm² | 8.8W |
| os_systolic_fused_attention | M1 | PASS/MET | 20.06 | 674.8 | 239.6ms | 42.9mm² | 8.9W |
| block | M2 | PASS/MET | 28.04 | 286.4 | 482.7ms | 44.3mm² | 8.84W |
| block_fused_attention | M1 | PASS/MET | 28.07 | 288.1 | 479.9ms | 44.4mm² | 8.91W |
| systolic | M2 | PASS/MET | 20.75 | 1433.4 | 137.5ms | 48.1mm² | 12.60W |
| fsa | M1 | PASS/MET | 20.46 | 1718.2 | 123.4ms | 47.9mm² | 11.4W |
| gmma | M1 | PASS/MET | 20.51 | 1673.7 | 125.2ms | 61.9mm² | 18.4W |
| tensor_core | M1 | PASS/MISS | 24.96 | 165.2 | 814.8ms | 44.3mm² | 10.3W |
| input_stationary | M1 | FAIL/MET | 10.60 | 472.6 | 365.2ms | 42.8mm² | 9.4W |
| wmma | M1 | FAIL/MISS | 0.06 | 1.1 | 137264.6ms | 51.2mm² | 14.5W |

Raw 和 Comparison-ready 的名义最优均为 OS 32×64 @1GHz：20.03 TPS、241.761ms、42.8mm²。但它相对 20 TPS 下限只有约 0.16% 余量，而 OS manifest 的性能不确定度为 25%，所以：

- `raw_recommended`：OS 32×64 @1GHz；
- `recommended`：OS 32×64 @1GHz，含义仅为 M2 架构比较结果；
- `product_recommended`：null；
- 场景 A 产品 Signoff：**NOT SIGNED**。

Block 64×64 @800MHz WC OFF 仍应与 OS 32×64 @1GHz 一起保留到下一轮 calibration shortlist，不能用当前小面积差直接冻结产品架构。WC ON/OFF 的完整比较以 `reports/lpddr5-latest-dse-2026-07-15.md` 为准。

## 7. 场景 A Agent 完整 DSE

搜索：1980 配置，1980 valid，invalid_configs=0。
约束：0 raw feasible，0 comparison-ready，0 product-qualified。

| Engine | M | Decode TPS | Prefill TPS | TTFT | 主要失败 |
|---|---:|---:|---:|---:|---|
| block_fused_attention | M1 | 16.22 | 179.1 | 4946.8ms | TPS |
| os_systolic_fused_attention | M1 | 14.45 | 2329.4 | 444.8ms | TPS |
| fsa | M1 | 10.39 | 984.3 | 985.2ms | TPS |
| block | M2 | 14.75 | 141.9 | 6232.2ms | TPS、TTFT |
| os_systolic | M2 | 12.69 | 153.5 | 5777.6ms | TPS、TTFT |
| gmma | M1 | 9.64 | 154.5 | 5766.2ms | TPS、TTFT |
| systolic | M2 | 7.61 | 146.1 | 6120.0ms | TPS、TTFT |
| tensor_core | M1 | 13.85 | 81.7 | 10781.9ms | TPS、TTFT |
| input_stationary | M1 | 0.76 | 139.8 | 7578.4ms | TPS、TTFT |
| wmma | M1 | 0.02 | 0.5 | 1958110.6ms | TPS、TTFT |

BFSA/OFSA/FSA 的 Attention 优化能够显著改善 TTFT，但没有解决完整模型权重和长 KV context 下的 Decode TPS。场景结论仍是 **INFEASIBLE**，不能把某个 M1 架构的单项优势当作产品推荐。

## 8. Signoff 状态

| Signoff 层级 | 状态 | 说明 |
|---|---|---|
| Framework computational core | PASS | 回归、canonical coverage、物理属性、invalid=0、三次确定性通过 |
| Full Framework | CONDITIONAL PASS | CLI/legacy/CV 全量覆盖和非 Windows 平台回归待补 |
| Systolic/OS/Block Engine | M2 | 可进入架构比较；没有 M3 产品资格 |
| 其余七个 Engine | M1 | 可进入 Raw Exploration；证据不足以进入 Comparison-ready |
| 场景 A architecture exploration | PASS | 有 raw/M2 候选，结果可复现 |
| 场景 A product | NOT SIGNED | 无 M3；名义 OS 点 guardband 不足 |
| Agent | INFEASIBLE | 所有候选违反至少一个硬约束 |

## 9. 后续优先级

1. 用独立模型、Func/RTL trace 校准 Block 和 OS，形成误差分布；
2. 将场景 A 的 TPS/TTFT guardband 直接计算进 product-qualified 判定；
3. 增加稳定的 golden summary diff，避免对十几 MB JSON 做脆弱快照；
4. 补 CLI orchestration、requirements/preflight、CV 分支测试和 Linux 回归；
5. 按统一流程验证 FSA paper-faithful 数据点；
6. BFSA/OFSA 完成独立周期模型和公平 PPA 后再申请 M2；
7. 对 Tensor Core、WMMA、GMMA、IS 的经验常数建立独立证据，不能仅凭模型可运行升级成熟度。

## 10. 可复现命令

```bash
.venv\Scripts\python -m pytest sim/tests -q
.venv\Scripts\python sim/engine_audit.py
.venv\Scripts\python sim/design_space_explorer.py \
  --scenario lpddr5_3b --top 50 \
  --output results/lpddr5_3b_latest_2026-07-15.json
.venv\Scripts\python sim/design_space_explorer.py \
  --scenario lpddr5_3b_agent --top 50 \
  --output results/lpddr5_3b_agent_latest_2026-07-15.json
```

## 11. 产物

- `sim/config/engine_manifests.yaml`
- `sim/results/engine_admission_audit_v36.json`
- `sim/results/lpddr5_3b_latest_2026-07-15.json`
- `sim/results/lpddr5_3b_agent_latest_2026-07-15.json`
- `sim/results/lpddr5_latest_dse_summary_2026-07-15.json`
- `reports/lpddr5-latest-dse-2026-07-15.md`
- `docs/new-engine-integration-methodology.md`
- `docs/arc-model-test-signoff-plan.md`
