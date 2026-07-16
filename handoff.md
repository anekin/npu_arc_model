# NPU Arc Model 工作交接

更新时间：2026-07-16
仓库：`git@github.com:anekin/npu_arc_model.git`
本地路径：`C:\Data\Codex\npu_arc_model`
工作分支：`feat/scenario-driven-dse`
当前功能基线：`38409d8 fix: cap on-chip DSE bandwidth`
Arc Model 版本：`v3.8-onchip-bandwidth-cap`

> 本文是接手工作的首要入口。开始前先执行 `git status -sb` 和
> `git log -3 --oneline`，确认实际 HEAD 与远端分支状态；本文随仓库继续更新。

## 1. 项目目标和边界

Arc Model 的目标是：输入不同应用场景的模型、上下文、TPS、TTFT、内存、面积、
功耗和工艺要求，通过设计空间搜索得到最合适的 NPU 架构候选，并展示选择依据、
失败原因和不确定度。

本仓库负责架构探索，不负责某个产品的具体实现：

| 层次 | 责任 | 仓库 |
|---|---|---|
| Arc Model | 应用需求 → DSE → NPU Engine/配置候选 | 本仓库 |
| Func Model | 冻结架构后的 bit-exact 功能模型 | `CaduceusCore` |
| RTL | 冻结架构后的可综合实现和实测 PPA | `CaduceusCore` |

Func Model、RTL、综合和硅后数据可作为 Arc Model 的校准证据，但不能成为 DSE
运行时依赖。不要把本仓库重新变成针对单一产品的实现仓库。

## 2. 当前架构和决策语义

正式入口是 `sim/design_space_explorer.py`。评估主路径位于 `sim/dse/`，Engine
位于 `sim/engine/`，机器可读的 Engine 证据位于
`sim/config/engine_manifests.yaml`。

DSE 输出分为三层：

- **Raw Exploration**：M1 以上，允许研究架构参与原始数值探索；
- **Comparison-ready**：M2 以上，可进行正式架构横向比较；
- **Product-qualified**：M3/M4，才允许形成产品 shortlist/recommendation。

`recommended` 是 M2 架构比较结果；只有 `product_recommended` 才表示产品资格
过滤后的结果。成熟度不修改 TPS、TTFT、面积或功耗，也不能因为某个 M1 候选数值
漂亮就把它写成正式推荐。

排序顺序是：硬约束过滤 → 软目标距离 → `objectives` 字典序 → tie-breaker。
没有硬约束可行点时只能输出 closest candidate，不能伪装成推荐结果。

## 3. Engine Inventory

| Engine | Maturity | WC 变体 | 当前定位 |
|---|---:|---|---|
| `systolic` | M2 | OFF / ON | Weight-stationary；WC ON 为双 weight 路径 |
| `os_systolic` | M2 | N/A | 独立 Output-stationary wavefront 模型 |
| `block` | M2 | OFF / ON | Broadcast Block Array；场景 A 风险调整主候选 |
| `block_fused_attention` | M1 | OFF / ON | Block + FSA-inspired fused attention |
| `os_systolic_fused_attention` | M1 | N/A | OS + FSA-inspired fused attention |
| `tensor_core` | M1 | N/A | 架构分析候选 |
| `wmma` | M1 | N/A | 架构分析候选 |
| `gmma` | M1 | OFF / ON | Group MMA/TMA 分析候选 |
| `input_stationary` | M1 | N/A | Input-stationary 分析候选 |
| `fsa` | M1 | N/A | FSA 论文参考；Attention component 有 M2 证据 |

Engine 准入和升级必须遵守
[`docs/new-engine-integration-methodology.md`](docs/new-engine-integration-methodology.md)。
新增 Engine 不能只实现一个同名类：必须补 architecture contract、独立证据、
manifest、microbench、PPA、适用边界、fallback、公平性审查和 DSE 报告。

## 4. Weight Cache 当前契约

WC ON/OFF 是不同硬件实现，不是免费的软件开关：

- `systolic`、`block`、`block_fused_attention`、`gmma` 同时搜索 WC OFF/ON；
- OS 等不支持该机制的 Engine 在报告中显示 `N/A`；
- JSON 保留兼容字段 `engine_comparison`，正式查看硬件变体应使用
  `engine_variant_comparison`；
- WC 调度保留两个独立 GEMM fallback，保证 pair 不比合法基线更慢；
- 当前 PE array 面积代理：Systolic +15%、Block/BFSA +10%、GMMA +5%；
- 功耗使用相同的 WC-adjusted PE logic area。

以上 PPA 只适用于架构阶段排序，尚未经过寄存器/SRAM 实现、综合、布局、活动率和
leakage 校准，不能作为 M3 或产品 Signoff 数据。参数来源和局限见
[`references/area_sources.md`](references/area_sources.md)。

注意：OS Engine 的 scratchpad activation reuse 不是 Weight Cache，不要把 OS
错误标成 WC ON/OFF。

## 5. 当前应用场景

场景定义位于 `sim/config/scenarios.yaml`。

### 5.1 场景 A：`lpddr5_3b`

- 低成本端侧算力扩展；
- Qwen2.5-3B，INT4 权重、INT8 激活、FP16 Attention/KV；
- 128-token prompt + 128-token output，batch=1，并发=1；
- 64-bit LPDDR5-6400，物理 51.2 GB/s，名义效率 85%；
- 带宽效率角：75% / 85% / 90%；
- Decode TPS 硬下限 20 tok/s；
- TTFT 目标 500 ms、硬上限 1000 ms；
- 面积硬上限 80 mm²；
- INT2 明确不进入场景 A 搜索；
- 低成本优先，目标达标后按面积、功耗、TPS 排序。

### 5.2 Agent 子场景：`lpddr5_3b_agent`

- 30,000-token cached prefix + 875-token append + 214-token output；
- 4 GB LPDDR，90% 可用容量；
- Decode TPS 硬下限 20 tok/s；
- TTFT 目标 2000 ms、硬上限 5000 ms；
- Prefill 目标 500 tok/s；
- 当前三个 LPDDR 效率角全部不可行。

Agent workload 假设及市场定位分析见
[`docs/agent-workload-requirements.md`](docs/agent-workload-requirements.md)。

### 5.3 场景 B：`onchip_7b`

- 面向具身智能/VLM/VLA；Qwen2.5-7B，1024-token prompt、128-token output；
- 5GB on-chip 3D DRAM，4.5GB 可用，额定带宽 500GB/s；
- Decode TPS≥100、TTFT≤200ms、area≤150mm²；当前未定义 power hard limit；
- 搜索 INT4/INT2 和 32×1536～128×1536 宽阵列；
- raw exploration 有 42 个 M1 OFSA 可行点，M2 comparison-ready 仍不可行。

场景 A 和 B 的目标函数不同，不能复用同一推荐排序结论。

## 6. 最新 DSE 结论

权威报告：

- [`reports/lpddr5-latest-dse-2026-07-15.md`](reports/lpddr5-latest-dse-2026-07-15.md)
- [`reports/scenario-b-onchip-7b-dse-2026-07-16.md`](reports/scenario-b-onchip-7b-dse-2026-07-16.md)

### 6.1 场景 A，85% 名义角

- 1965/1965 配置 valid；
- 756 raw feasible；
- 334 M2 comparison-ready feasible；
- 0 M3/M4 product-qualified；
- 名义成本最优：OS Systolic 32×64@1GHz、L2=1MB；
- 指标：20.03 TPS、TTFT 241.8 ms、42.8 mm²、8.84 W；
- TPS 只比 20 TPS 下限高约 0.15%，小于 OS 当前 25% 性能不确定度，不能冻结产品；
- 风险调整主 shortlist：Block 64×64@800MHz、WC OFF；
- Block WC OFF：28.04 TPS、44.3 mm²、8.84 W；
- Block WC ON：28.05 TPS、44.6 mm²、8.96 W，带宽瓶颈下不值得增加成本。

Systolic 128×128@1.2GHz、L2=1MB 的 WC 差异较明显：

| WC | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---|---:|---:|---:|---:|---:|
| OFF | 16.42 | 1190.8 | 168.4 ms | 47.2 mm² | 12.07 W |
| ON | 20.75 | 1433.4 | 137.5 ms | 48.1 mm² | 12.60 W |

结论不是“WC 一定好/坏”，而是必须针对 Engine、阵列和场景比较收益与实现成本。

### 6.2 Agent

- 1980/1980 配置 valid；
- 0 raw/M2/product feasible；
- 75%/85%/90% 三个效率角均为 `INFEASIBLE`；
- WC 没有改变结论；
- 下一步应研究带宽/通道、Prefix/KV 策略、工作负载约束或经校准的新 Engine，
  不能依赖 WC 单点优化解决。

### 6.3 场景 B

- 1710/1710 配置 valid，0 invalid；
- 42 raw feasible，全部属于 M1 `os_systolic_fused_attention`；
- 0 M2 comparison-ready feasible，0 M3/M4 product-qualified；
- DSE raw best：OFSA 48×1536 INT2@1.2GHz、L2=1MB，139.53 TPS、
  TTFT 175.15ms、100.0mm²、46.69W；INT2 精度未验证；
- 工程研究基线：同配置 INT4，116.60 TPS、TTFT 176.69ms、100.0mm²、46.69W；
- M2 最近点：OS Systolic 96×1536 INT2@1.2GHz，132.17 TPS、
  TTFT 251.80ms、145.7mm²、76.53W，因 TTFT 超限而 FAIL；
- ARC-BUG-007 已将面积耦合带宽封顶到额定 500GB/s；首次无封顶运行无效。

完整结果：

- `sim/results/lpddr5_3b_latest_2026-07-15.json`
- `sim/results/lpddr5_3b_agent_latest_2026-07-15.json`
- `sim/results/lpddr5_latest_dse_summary_2026-07-15.json`
- `sim/results/onchip_7b_latest_2026-07-16.json`
- `sim/results/onchip_7b_dse_summary_2026-07-16.json`

## 7. 测试与 Signoff 基线

测试方案：[`docs/arc-model-test-signoff-plan.md`](docs/arc-model-test-signoff-plan.md)
执行报告：
[`reports/engine-admission-and-signoff-test-2026-07-15.md`](reports/engine-admission-and-signoff-test-2026-07-15.md)

当前基线：

- 126 tests passed；
- Engine admission audit：10/10 Engine 通过；
- canonical 计算主路径 statement coverage 97%、branch coverage 85%；
- 场景 A、Agent 和场景 B 全量搜索 invalid config 均为 0；
- Framework computational core：PASS；
- Full Framework：CONDITIONAL PASS；
- 场景 A architecture exploration：PASS；
- 场景 A product：NOT SIGNED；
- Agent：INFEASIBLE；
- 场景 B raw exploration：PASS；
- 场景 B comparison-ready：INFEASIBLE；
- 场景 B product：NOT SIGNED；
- 当前没有 M3/M4 Engine。

正式变更至少执行：

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp C:\tmp\npu_arc_model_handoff
.venv\Scripts\python.exe sim\engine_audit.py
git diff --check
```

改动 Engine、工作负载、约束、PPA 或 ranking 时，还必须重跑受影响的完整场景，
不能只跑 `--quick` 后更新正式报告。

## 8. Bug Track

正式台账：[`docs/bug-tracker.md`](docs/bug-tracker.md)。

| ID | 状态 | 内容 | 修复提交 |
|---|---|---|---|
| ARC-BUG-001 | CLOSED | GMMA TMA overlap 突破物理 DMA ceiling | `2897dff` |
| ARC-BUG-002 | CLOSED | GMMA WC pair 可能性能倒退 | `2897dff` |
| ARC-BUG-003 | CLOSED | EngineResult.ops 单位不一致 | `2897dff` |
| ARC-BUG-004 | CLOSED | Windows GBK stdout 导致 CLI 崩溃 | `2897dff` |
| ARC-BUG-005 | CLOSED | WC PPA 成本缺失、报告合并 ON/OFF | `2ddcccf` |
| ARC-BUG-006 | CLOSED | Systolic WC pair 缺少单调 fallback | `2ddcccf` |
| ARC-BUG-007 | CLOSED | On-chip memory 面积耦合突破额定 500GB/s | `38409d8` |

新增缺陷必须先登记编号、复现、影响、修复和 regression。P0/P1 在目标分支发布前必须
`CLOSED`；模型精度缺口和软件 Bug 不要混用同一状态。

## 9. 关键文件地图

| 文件 | 用途 |
|---|---|
| `README.md` | 项目边界和快速入口 |
| `sim/design_space_explorer.py` | 正式 DSE CLI、搜索空间和输出编排 |
| `sim/dse/evaluator.py` | canonical workload/performance/PPA 评估 |
| `sim/dse/reporting.py` | Engine 与硬件变体比较表 |
| `sim/dse/constraints.py` | 硬约束 |
| `sim/dse/workload.py` | LLM workload/trace |
| `sim/engine/mac_engine.py` | Engine factory 和公共契约 |
| `sim/engine/ppa_model.py` | 架构阶段面积/功耗模型 |
| `sim/config/scenarios.yaml` | 内置场景需求 |
| `sim/config/design_space.yaml` | 基线参数和 PPA proxy |
| `sim/config/engine_manifests.yaml` | Engine maturity/evidence/scope/gap |
| `sim/tests/` | 回归、物理属性和报告测试 |
| `docs/new-engine-integration-methodology.md` | 新 Engine 引入方法 |
| `docs/arc-model-test-signoff-plan.md` | 测试与 Signoff 标准 |
| `docs/bug-tracker.md` | 正式 Bug 台账 |
| `reports/lpddr5-latest-dse-2026-07-15.md` | 当前 LPDDR 权威报告 |

`sim/arc_model.py`、Golden Executor、量化前向和 ISA/调度相关代码属于迁移保留的
旧版校准/研究工具，不是当前 DSE 主入口。

## 10. 复现命令

安装环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

场景 A 名义全量搜索：

```powershell
.venv\Scripts\python.exe sim\design_space_explorer.py `
  --scenario lpddr5_3b --top 50 `
  --output results\lpddr5_3b_latest_2026-07-15.json
```

Agent 名义全量搜索：

```powershell
.venv\Scripts\python.exe sim\design_space_explorer.py `
  --scenario lpddr5_3b_agent --top 50 `
  --output results\lpddr5_3b_agent_latest_2026-07-15.json
```

注意：`--output results\...` 由 CLI 解析到 `sim/results/`。不要写成
`--output sim\results\...`，否则会误生成 `sim/sim/results/`。

75%/90% 效率角应从对应场景复制精确 requirements YAML，只修改
`memory.dram_efficiency` 和 `memory.effective_bw_gbps`，独立运行并在 summary 中记录
场景定义和源文件 SHA256，不能把不同角的点混入同一次排序。

## 11. 已知缺口和下一步优先级

建议按以下顺序继续：

1. **Block/OS 独立校准**：用 Func/RTL trace 建立误差分布，推动 M2→M3；
2. **Guardband 产品判定**：把 Engine uncertainty 和场景 guardband 显式纳入
   `product_eligible`，避免名义 0.15% 裕量成为产品推荐；
3. **WC PPA 校准**：明确 dual register/SRAM、读写端口、时钟和 leakage，替换
   +15%/+10%/+5% proxy；
4. **Agent 可行性研究**：优先分析 memory channels/bandwidth、Prefix/KV policy、
   工作负载约束，不要先扩大 MAC array；
5. **FSA 系列证据升级**：保留 paper-faithful reference 与 FSA-inspired Engine 的
   区别；完成独立周期/PPA/RTL 证据后再申请 M2；
6. **场景 B 收敛**：校准 OFSA fused-attention、验证 INT2 精度、补功耗上限和 guardband；
7. **报告流水线**：将六角运行和 summary/Markdown 生成固化为稳定脚本，减少手工
   报告漂移和十几 MB JSON 的脆弱快照；
8. **跨平台与覆盖率**：补 Linux、CLI orchestration、requirements/preflight、CV
   分支测试。

## 12. 工作规则和常见陷阱

- DSE 结果必须展示所有 Engine；失败 Engine 要显示 closest point 和失败原因；
- WC-capable Engine 的报告必须同时显示 WC OFF/ON、TPS、TTFT、Area、Power；
- Decode TPS 不得突破有效带宽 ceiling；模型结果超过 ceiling 时先查单位和 floor；
- 场景 A 禁止 INT2；
- 不要把 FSA/BFSA/OFSA 的 M1 结果直接写成正式推荐；
- 面积/功耗可保持架构阶段粗估，但所有收益必须有非零实现成本和证据标签；
- 任何可选优化都应保留合法 fallback，不能启用后性能倒退；
- 修改场景需求必须给出来源或假设状态，不能拍脑袋改 TTFT/TPS；
- 历史报告只作对比，当前结论以最新版本可复现结果为准；
- Windows stdout 必须保持 UTF-8；
- 仓库可能位于默认 workspace 之外，编辑时需使用受控补丁/权限流程；
- 用户要求每轮完成后自动提交并推送。测试失败、报告检查失败或远端冲突时不得强推。

## 13. 每轮完成清单

1. `git status -sb`，确认没有覆盖他人未提交改动；
2. 更新实现、测试、manifest/来源/bug track；
3. 跑定向测试和全量测试；
4. 重跑受影响的正式 DSE，不以 quick 结果替代；
5. 更新 JSON、summary、Markdown、README/handoff；
6. `git diff --check`，校验 JSON 可解析、版本和变体表完整；
7. 审阅 `git diff`，确认报告数值与 JSON 一致；
8. 创建语义明确的 commit；
9. `git push origin feat/scenario-driven-dse`；
10. 再次确认工作区干净且本地分支与远端同步。
