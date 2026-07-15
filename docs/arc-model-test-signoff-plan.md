# Arc Model 测试与 Signoff 方案

状态：Draft for execution  
版本：v0.1  
日期：2026-07-15  
适用基线：Arc Model v3.5-os-dataflow 及后续版本

## 1. 目的与范围

本方案用于验证 Arc Model 是否能够：

- 正确计算架构性能、带宽、容量、面积和功耗；
- 在统一应用需求下公平比较不同 NPU Engine；
- 正确执行硬约束、目标、Pareto 和推荐排序；
- 给出可复现、可解释并带置信度的 DSE 结论；
- 支持新架构从研究模型逐步升级到产品候选。

测试对象包括正式 DSE 主路径：场景输入、model/workload、memory、所有 Engine、PPA、constraints、ranking、reporting 和结果 provenance。旧版研究脚本仅作为校准证据，不作为主路径 signoff 对象。

## 2. Signoff 分级

Arc Model 不采用单一“全部通过”结论，而采用三级 signoff：

| 层级 | 回答的问题 | 结果 |
|---|---|---|
| Framework Signoff | DSE 框架、物理约束、排序和报告是否可靠 | 工具可用于架构研究 |
| Engine Signoff | 某个 Engine 的模型达到什么证据等级 | M1/M2/M3/M4 |
| Scenario Signoff | 特定场景下的推荐是否在误差范围内仍成立 | 探索结论、候选 shortlist 或产品推荐 |

框架 signoff 不要求所有研究 Engine 都有 RTL，但要求每个 Engine 的成熟度、证据、适用范围和不确定度标注准确。

## 3. 测试目标

### 3.1 计算正确性

- GB/s、bytes/cycle、有效带宽和 memory ceiling；
- Decode TPS、Aggregate TPS、Prefill TPS、TTFT、ITL 和 E2E latency；
- 权重、KV Cache、activation、runtime reserve 和可用容量；
- Engine 分项周期、访存字节和利用率；
- PPA 组成、工艺缩放和 Engine 特有开销；
- constraints、targets、objectives、Pareto 和 tie-breaker。

### 3.2 物理合理性

- Decode TPS 不超过完整权重读取 ceiling；
- context/prompt 增大时 TTFT、Attention 工作量和 KV 容量不能下降；
- 带宽或计算资源增加不能无原因降低性能；
- 阵列和 SRAM 增大不能降低面积；
- 频率提高时 compute time 不增加，且物理 GB/s 不随频率错误增加；
- Engine 特有数据流在 Decode/Prefill 上表现出预期方向性差异；
- 融合收益必须同时核算新增硬件和未融合算子。

### 3.3 决策正确性

- 不漏掉设计空间中的合法配置；
- 不把非法或违反硬约束的候选推荐为可行方案；
- 研究 Engine 可以进入原始探索和比较，不因成熟度被修改原始指标；
- 排名与独立参考的架构趋势一致；
- 差异小于模型误差时输出候选档，而不是伪造唯一最优。

### 3.4 可复现性和可审计性

- 相同输入、代码和校准数据产生相同结果；
- 输出包含场景、模型、Engine、校准数据和代码版本；
- 结果变化可由输入、公式或证据版本解释；
- 历史报告不能被误认为当前版本的可复现结论。

## 4. 测试分层

| 层级 | 内容 | Oracle | 自动化要求 |
|---|---|---|---|
| T0 Static/Schema | YAML、manifest、必填字段、枚举、单位 | schema/contract | 每次提交 |
| T1 Unit | 带宽、容量、周期、PPA、约束、排序 | 手工 closed-form | 每次提交 |
| T2 Property | 单调性、上限、守恒、边界 shape | 物理不变量 | 每次提交 |
| T3 Engine Microbench | GEMM/Attention kernel 周期、流量、利用率 | 独立模型/RTL | Engine 变更时 |
| T4 Workload | Qwen 层级和端到端 Decode/Prefill | Func/trace | 模型或场景变更时 |
| T5 DSE | 搜索完整性、Pareto、ranking、gating | 穷举/独立排序 | 每次发布 |
| T6 Scenario | A、Agent、B 完整 DSE | 产品需求/trace | 场景 signoff |
| T7 Regression | Golden 摘要、跨平台、性能预算 | 已签核基线 | CI/发布 |

## 5. 测试数据集

### 5.1 Kernel shape

- M=1：Decode；
- M=8/32/64/128：小批量和常规 Prefill；
- M=875：Agent append；
- M=H-1/H/H+1、N=W-1/W/W+1：阵列 tile 边界；
- K 小于、等于、大于阵列维度以及不能整除的 K；
- 需要 K chunk、preload、flush 和 writeback 的情况；
- 模型真实 Q/K/V、Projection、FFN Gate/Up/Down shape。

### 5.2 Attention/context

- 128-token 场景 A；
- 875-token Agent append；
- 30,000-token cached prefix；
- 32,768-token capacity 边界；
- causal/non-causal、GQA、不同 attention/KV precision；
- cache hit/miss 和无 cached prefix 情况。

### 5.3 系统参数

- LPDDR 有效效率：75%、85%、90%；
- 800、1000、1200 MHz；
- 全部正式搜索阵列和 L2 SRAM 容量；
- Weight Cache 开启/关闭；
- INT4 场景 A，INT2 仅在显式允许场景；
- 容量刚好低于、等于、高于需求；
- 硬约束阈值的 -1%、等于、+1% 边界点。

## 6. Oracle 与证据

测试期望值按可信度使用：

1. 维度分析和 closed-form golden；
2. 与生产实现独立的事件模型或公开模拟器；
3. 论文公开数据和 RTL 调度；
4. CaduceusCore Func Model 性能 trace；
5. RTL 仿真 cycle trace；
6. 综合、布局布线和硅后数据。

禁止直接调用被测生产函数生成 golden。论文数据必须保存来源、配置、单位换算和提取过程。

## 7. Framework Signoff 标准

必须全部满足：

- 所有 P0/P1 自动化测试通过，无 skip、xfail；
- canonical DSE 代码行覆盖率不低于 90%，分支覆盖率不低于 85%；
- 完整搜索 invalid_configs=0；
- 搜索空间无重复和遗漏，实际数量符合组合规则；
- 每个 M1 以上 Engine 均出现在 raw comparison；
- 不支持字段、非法单位、负数和缺失必填项必须 fail closed；
- 同一输入连续运行三次，关键指标、Pareto 和排序一致；
- constraints、targets 和 recommendation maturity 互不混淆；
- raw、comparison-ready 和 product-qualified 三套结果均可生成；
- provenance 包含代码、场景、manifest 和校准数据版本。

浮点 closed-form 单元测试使用明确容差；不得用过宽的百分比掩盖单位错误。

## 8. Engine Signoff 标准

Engine M1/M2 门槛按《新 NPU Engine 引入与 DSE 准入方法》执行。M3 性能精度要求：

| 指标 | 门槛 |
|---|---:|
| Kernel cycle 中位绝对误差 | ≤10% |
| Kernel cycle P95 误差 | ≤20% |
| Kernel 单点最大误差 | ≤30% |
| 模型级 TPS/TTFT 中位误差 | ≤15% |
| 模型级 TPS/TTFT P95 误差 | ≤25% |
| 架构排序 Spearman 相关系数 | ≥0.90 |

校准集必须同时覆盖 Decode、Prefill、长 context 和边界 tile，不能只选择模型表现最好的样本。

## 9. Scenario Signoff 标准

### 9.1 输入需求

- 模型规格来自版本化配置；
- prompt、output、并发和 cache 命中率来自真实 trace 或明确标注的假设；
- 硬约束、设计目标和优化目标由场景 owner 确认；
- provisional 需求不能伪装成已签核需求；
- LPDDR 场景至少覆盖 75%/85%/90% 带宽效率角。

### 9.2 约束和 guardband

设性能模型 P95 相对误差为 e：

```text
TPS_conservative  = TPS_predicted × (1 - e)
TTFT_conservative = TTFT_predicted × (1 + e)
```

只有保守值仍满足硬约束，候选才可进入 product-qualified shortlist。最低性能 guardband 为 max(10%, 当前 Engine P95 误差)。

硬约束分类必须满足：

- false PASS = 0；
- 边界样本全部检查；
- 无可行架构时明确输出 infeasible 和 closest candidates；
- 不得用设计目标代替硬约束，也不得把 closest candidate 标为推荐。

### 9.3 推荐质量

- 独立参考的最优架构应进入 DSE Top 3；
- DSE Top 1 相对参考 Pareto 最优点的场景目标损失不超过 10%；
- M2 以上架构排序相关性不低于 0.90；
- 参数和带宽角扰动后，推荐变化必须能够由瓶颈迁移解释；
- 候选差异小于误差区间时输出 shortlist/tie band。

## 10. PPA Signoff 处理

架构阶段允许 PPA 比性能模型粗糙，但不能忽略：

- 面积参考误差目标：约 ±30%；
- 功耗参考误差目标：约 ±35%；
- 面积、功耗必须为正并随资源规模合理单调；
- Engine 特有单元必须计入；
- 未校准时输出区间而不是精确小数差异；
- 两个候选的 PPA 差异小于不确定度时，视为同一成本档。

场景 A 以低成本为主要目标，因此在 PPA 误差尚大时，不能凭 1～2 mm² 差异给出唯一最优架构。

## 11. 必测属性

### 11.1 Memory

- physical GB/s 不随 Engine 频率变化；
- bytes/cycle 按频率反向换算；
- effective bandwidth 不超过 physical bandwidth；
- Decode TPS 不超过完整模型 memory ceiling；
- capacity 不足时所有 Engine 均 fail。

### 11.2 Workload

- prompt/context 增大使 prefill cycles、TTFT 和 Attention 工作量非减；
- cached prefix 只减少可复用部分，不能删除长 context Attention；
- output token 增大使 E2E latency 非减；
- concurrency/batch 的单请求与 aggregate 指标不混淆。

### 11.3 Engine

- Block 与 OS 不复用周期结果；
- OS 大 M 和 M=1 的阵列利用率方向正确；
- fused Engine 只降低被融合算子，不改变无关 workload；
- fused Engine 的面积/功耗不低于对应 baseline；
- fallback 的输出与被复用基线一致。

### 11.4 Ranking/reporting

- hard constraint 优先于 target/objective；
- target 达标优先于成本排序；
- tie-breaker 只处理数值相同或同一误差档的候选；
- maturity 不修改 raw metric；
- research/低成熟度 Engine 不被隐藏；
- product shortlist 不包含低于 M3 的 Engine。

## 12. 自动化与交付物

每轮 Framework Signoff 产生：

- JUnit/pytest 结果；
- coverage 报告；
- schema/manifest validation；
- 完整 DSE JSON；
- 关键指标 golden summary 和与上版 diff；
- invalid/duplicate/missing config 审计；
- Engine maturity/evidence 表；
- calibration error 分布；
- 场景报告和 signoff checklist；
- known limitations、风险 owner 和 waiver。

完整 JSON 用于追溯；回归门禁应优先比较稳定的摘要和 provenance，避免把输出顺序或无意义格式变化当成模型回归。

## 13. 缺陷等级与 Waiver

- P0：物理上限错误、false PASS、推荐不可复现、单位错误；阻止所有 signoff。
- P1：Engine 主要路径周期错误、排序错误、搜索遗漏；阻止相关 Framework/Engine/Scenario signoff。
- P2：误差超标但已正确降级成熟度、非关键报告问题；可带 owner 和期限 waiver。
- P3：文档、展示和低风险可用性问题；不阻止 signoff。

任何 waiver 必须包含影响范围、临时保护措施、owner、到期日和关闭证据。

所有 Signoff 缺陷必须登记到 [`docs/bug-tracker.md`](bug-tracker.md)，使用
`ARC-BUG-NNN` 编号并关联复现、根因、回归测试、修复提交和 GitHub Issue。
Engine manifest 的 `known_gaps` 只记录已声明且已正确隔离的模型不确定性；一旦
违反物理/单位/排序/报告契约或造成错误推荐，必须转为正式 Bug。

## 14. 执行阶段

### Phase 0：冻结规范

- 合入本方案和新 Engine 引入方法；
- 定义 manifest/schema、成熟度和误差统计格式；
- 冻结场景 A/Agent/B 的需求版本。

退出标准：测试项、门槛、owner 和校准数据格式完成评审。

### Phase 1：Framework 基础门禁

- 补齐 schema、属性、边界、搜索完整性和三套排名测试；
- 引入 coverage、golden summary 和 provenance 检查。

退出标准：Framework Signoff 全部通过。

### Phase 2：Block/OS 校准

- 建立真实 GEMM/Attention shape 微基准；
- 导入独立模型、Func/RTL trace；
- 形成误差分布并确定成熟度。

退出标准：Block/OS 明确为 M2 或 M3，不再依赖默认名称判断。

### Phase 3：场景 A Signoff

- 运行 75%/85%/90% 带宽角；
- 使用性能 guardband 和 PPA tie band；
- 输出 raw、comparison-ready 和 product-qualified 报告。

退出标准：推荐 shortlist 在保守角仍满足硬约束，或明确判定当前无可签核架构。

### Phase 4：Agent 与场景 B

- 用本地真实 agent trace 校准 30K Prefix + 875 Append；
- 校准场景 B 实时约束和并发行为。

退出标准：分别完成场景 signoff，不能复用场景 A 结论。

### Phase 5：FSA 系列流程验证

- FSA 完成 paper-faithful 与 normalized 两套报告；
- BFSA/OFSA 完成独立周期交叉验证和公平 PPA；
- 按统一门槛从 M1 升级到 M2/M3。

## 15. 当前基线与已知缺口

截至 2026-07-15、提交 2757fc7：

- 现有 36 项测试通过；
- 场景 A 1965/1965 配置有效，invalid_configs=0；
- Agent 1980/1980 配置有效，invalid_configs=0；
- FSA/BFSA/OFSA 当前以硬编码 recommendation gate 隔离；
- OS 为独立 dataflow analytical 模型，但缺少本项目 RTL 校准；
- 场景 A 名义推荐只有约 0.16% TPS 余量，不能通过 guardband signoff；
- PPA 不确定度尚未传播到成本排序；
- Agent workload 仍需本地真实 trace。

因此当前版本可以作为架构研究基线，但尚未完成 Framework、Engine 和 Scenario 的正式三级 signoff。

## 16. 首轮可执行 Backlog

| 优先级 | 工作项 | 主要交付物 | 完成标准 |
|---|---|---|---|
| P0 | Engine manifest/schema | schema、所有 Engine manifest、校验测试 | 非法/缺失字段 fail closed |
| P0 | 三套 DSE 结果 | raw/comparison/product JSON 与报告 | maturity 不修改 raw metric |
| P0 | 搜索完整性测试 | 数量、重复、invalid 审计 | 全场景 invalid=0 |
| P0 | 物理属性测试 | bandwidth/ceiling/capacity/monotonic tests | 全部通过 |
| P1 | Block/OS 微基准 | shape corpus、golden、误差报告 | 明确 M2/M3 |
| P1 | guardband/tie band | 保守指标和不确定排序 | false PASS=0 |
| P1 | 场景 A signoff | 三带宽角完整报告 | shortlist 或明确 infeasible |
| P2 | Agent trace | trace schema、统计分布、场景更新 | 需求假设可追溯 |
| P2 | FSA 流程验证 | paper-faithful、normalized、升级评审 | 达到 M2 或保留 M1及原因 |

执行过程中，每个阶段必须先满足退出标准再进入下一阶段；新增 Engine 可以并行走自身 M0→M3 流程，但不得绕过统一 schema、fairness review 和测试门禁。
