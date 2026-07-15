# 新 NPU Engine 引入与 DSE 准入方法

状态：Draft for execution  
版本：v0.1  
日期：2026-07-15

## 1. 目的

本方法用于把一个原创 NPU 架构、已有内部架构或论文架构引入 Arc Model，并在统一应用场景下与现有 Engine 公平比较。

Arc Model 是架构研究工具，因此“证据尚未达到产品签核标准”不能成为拒绝新架构进入 DSE 的理由。流程必须分别回答：

1. 该 Engine 是否已经完整到可以执行和比较；
2. 该 Engine 的结果有多高置信度；
3. 该 Engine 是否已经具备产品推荐资格。

## 2. 核心原则

- **探索准入与产品推荐分离**：达到最低建模完整度即可进入探索；只有经过校准和 guardband 检查的候选才能进入产品 shortlist。
- **原始结果不施加成熟度惩罚**：成熟度不能被偷偷折算成 TPS、TTFT 或面积惩罚。DSE 应同时给出原始排名、置信度感知排名和产品候选。
- **不允许只建模优势算子**：不支持的算子必须显式 fallback，并计入周期、访存、面积和功耗。
- **统一比较口径**：同一 workload、精度、工艺、带宽、容量和系统开销下比较。
- **证据可追溯**：公式、论文数据点、RTL trace、综合结果和校准脚本必须版本化。
- **不制造伪精度**：模型误差大于候选差异时，应输出同一候选档或不确定排序。

## 3. Engine 成熟度等级

| 等级 | 名称 | 定义 | DSE 行为 |
|---|---|---|---|
| M0 | Concept | 只有想法或局部公式，无法完成 workload | 不进入正式搜索，可保留实验脚本 |
| M1 | Executable | 周期、流量、fallback 和粗略 PPA 完整，可运行全 workload | 进入 Raw Exploration，显示低置信度 |
| M2 | Comparison-ready | 已通过独立交叉检查，能够在统一条件下公平比较 | 进入正式 Pareto 和架构排名 |
| M3 | Calibrated | 有独立模拟器、Func Model 或 RTL trace 校准 | 进入置信度感知推荐 |
| M4 | Product-signed | 在目标场景、工艺和 guardband 下完成签核 | 进入产品推荐结论 |

成熟度不能只由一个字符串主观指定。每个 Engine 还必须分别声明以下证据轴：

| 证据轴 | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Performance | 无模型 | 分析公式 | 论文复现或独立模型 | Func/RTL 校准 | 硅后数据 |
| PPA | 未计入 | 参数化估算 | 文献/缩放交叉检查 | 本项目综合 | 布局布线/硅后 |
| Functional scope | 局部概念 | 支持范围明确 | 全 workload + 显式 fallback | Func trace 验证 | RTL/系统验证 |
| System integration | Kernel only | 计入模型级算子 | 计入内存、调度和容量 | 软件/RTL trace | 实际系统 |

Engine 的综合成熟度取关键证据轴的最低可接受等级，而不是取平均分。

## 4. Engine Architecture Contract

每个新 Engine 在编码前必须提交架构契约，至少包括：

### 4.1 功能范围

- 支持的算子、矩阵 shape、数据类型、精度和 batch；
- 不支持的算子及 fallback Engine；
- 是否支持 causal attention、GQA、cached prefix、长 context；
- 数值语义是否与现有模型一致。

### 4.2 数据流和调度

- M、N、K 分别映射到空间维度还是时间维度；
- PE/MAC 数量、每周期操作定义及利用率；
- tile、pipeline、preload、flush 和 writeback；
- activation、weight、KV 和 partial sum 的驻留位置；
- 跨 tile、跨 head、跨 layer 的复用条件；
- weight cache 或融合路径的生效范围。

### 4.3 存储和带宽

- 每个算子的逻辑读取/写入字节数；
- SRAM/register/accumulator 容量和端口假设；
- bank conflict、backpressure 和并发传输假设；
- 片外 memory ceiling 与有效带宽效率；
- 容量不足时的失败或 spill 行为。

### 4.4 PPA

- 基线 PE/MAC、控制、buffer、transposer、SFU 等组成；
- 新增硬件不能隐藏在现有 Engine 的固定开销内；
- 工艺、频率和电压假设；
- 估算来源、缩放方法和误差范围。

### 4.5 适用边界

- 已验证和未验证的 shape/precision/workload；
- 当前忽略的开销；
- 预期误差方向；
- 失效条件和禁止外推范围。

## 5. 引入流程

### 阶段 0：Proposal

交付物：架构假设、预期收益、目标 workload、与现有 Engine 的本质差异、待验证风险。

退出标准：能够说明收益来自数据流、存储复用、计算单元还是调度，而不是一个经验倍率。

### 阶段 1：Standalone Analytical Model

实现独立 kernel 模型，输出：

- compute cycles；
- memory cycles；
- logical/physical bytes；
- utilization；
- 分项 latency；
- 模型假设和 warning。

退出标准：closed-form 单元测试、边界 shape 和物理单调性测试全部通过。

### 阶段 2：论文复刻或独立交叉验证

论文架构必须先做 Paper-faithful 模式：

- 使用论文相同阵列、频率、工艺、精度、buffer 和 workload；
- 复现至少三个公开数据点；
- 解释无法复现的数据和输入缺失；
- 保存原始来源、提取过程和误差表。

原创架构没有论文锚点时，至少采用两种相互独立的方法交叉验证，例如 closed-form 与事件模拟器。

退出标准：论文数据点建议误差不超过 15%；超出时必须保留偏差说明，不能直接升级 M2。

### 阶段 3：Normalized DSE Integration

把 Engine 接入统一 evaluator，并使用本项目的：

- 模型规格和 workload；
- 权重、activation、attention 和 KV 精度；
- 工艺、频率、带宽和容量；
- 系统级 DMA、SRAM 和软件开销；
- 面积和功耗模型。

论文原始结果和归一化 DSE 结果必须分开保存，不能混为同一份性能证据。

退出标准：可完成全 workload；所有 fallback 显式；全搜索无异常和缺失指标。

### 阶段 4：Fairness Review

独立复核以下问题：

- 相同的 H×W 是否代表相同 MAC 数和相同操作能力；
- Engine 特有 buffer/SFU/transposer 是否计入 PPA；
- 是否使用了比基线更有利的精度、频率、带宽或 cache；
- 是否遗漏 Projection、FFN、Norm、RoPE、Residual 或 writeback；
- 是否把逻辑复用错误地当作物理零成本；
- fallback 是否使用相同基线和系统开销。

退出标准：iso-area、iso-power、iso-bandwidth 和场景自由搜索均可生成报告。

### 阶段 5：DSE Admission Review

按第 6 节门槛确定 M1/M2 等级，并将 Engine manifest、测试和证据合入主分支。

### 阶段 6：Calibration Promotion

获得独立模拟器、Func Model、RTL、综合或硅后数据后，更新版本化校准数据，而不是修改公式去拟合单个结果。

退出标准：满足 M3/M4 的量化误差和目标场景 guardband。

## 6. 准入与升级标准

### 6.1 M1：进入 Raw Exploration

必须全部满足：

- 能计算完整 workload；
- unsupported op 有显式 fallback；
- 周期、流量、容量和粗略 PPA 均非空；
- 不超过物理带宽和计算上限；
- 边界 tile、M=1 Decode 和大 M Prefill 可执行；
- 参数变化符合物理单调性；
- 相同输入输出可复现；
- manifest 声明证据、范围、不确定度和已知缺口。

### 6.2 M2：进入正式架构比较

在 M1 基础上必须满足：

- 论文锚点误差建议不超过 15%，或通过两种独立模型交叉验证；
- 所有模型级算子和 memory traffic 已归因；
- PPA 至少有参数化估算和误差范围；
- 通过独立 Fairness Review；
- 可生成统一 per-engine、Pareto 和四种归一化比较报告；
- 不使用隐藏的特殊场景参数。

### 6.3 M3：进入置信度感知推荐

- kernel cycle 中位绝对误差不超过 10%，P95 不超过 20%；
- 模型级 TPS/TTFT 中位误差不超过 15%，P95 不超过 25%；
- 架构排序 Spearman 相关系数不低于 0.90；
- Decode、Prefill、长 context 和边界 tile 均有校准样本；
- calibration dataset、工具版本和适用范围可追溯。

### 6.4 M4：进入产品推荐

- 目标场景需求已签核；
- 使用误差下界/上界后仍满足全部硬约束；
- 性能、带宽、容量、面积和功耗风险均有 owner；
- 没有未关闭的 P0/P1 缺陷；
- waiver 经过场景 owner 和架构 owner 批准。

## 7. Engine Manifest 建议格式

每个 Engine 应有机器可读 manifest，例如：

```yaml
engine: os_systolic_fused_attention
model_version: v1
maturity: M1
evidence:
  performance: analytical
  ppa: analytical
  functional_scope: full_model_with_explicit_fallback
  system_integration: workload_level
scope:
  weight_bits: [4]
  attention_bits: [16]
  causal_attention: true
fallbacks:
  projection: os_systolic
  ffn: os_systolic
uncertainty:
  performance_pct: 25
  area_pct: 30
  power_pct: 35
known_gaps:
  - native RTL calibration
  - SRAM bank conflict
calibration_dataset: null
```

字段应由 schema 校验；成熟度升级必须同时更新证据和测试，不能只修改等级字符串。

## 8. DSE 输出要求

每次搜索同时产生三套结论：

1. **Raw Exploration Ranking**：所有 M1 以上 Engine，按原始预测指标排名，不施加成熟度惩罚；
2. **Comparison-ready Pareto**：所有 M2 以上 Engine，按统一条件生成 Pareto；
3. **Product-qualified Shortlist**：M3/M4 Engine，使用误差和 guardband 判断硬约束。

每个 Engine 均应显示：

- maturity 和各证据轴；
- TPS、Prefill TPS、TTFT、ITL、面积和功耗；
- 原始值、保守值和不确定度；
- supported/fallback 组成；
- hard constraint、target 和推荐资格；
- warning、provenance 和校准数据版本。

不得因为成熟度低而隐藏性能最好的研究架构；也不得把低证据结果包装成产品结论。

## 9. 当前 Engine 的迁移建议

以下是待证据审计的初始定位，不是最终签核：

| Engine | 初始等级 | 说明 |
|---|---|---|
| block | M2 | 全模型可执行；需整理性能/PPA 校准数据后判断 M3 |
| os_systolic | M2 | 独立数据流模型；尚缺本项目 RTL cycle 校准 |
| fsa | Attention kernel M2，整机 M1/M2 | 有论文/公开 RTL依据；系统级性能和 PPA存在外推 |
| block_fused_attention | M1 | FSA 思想导入 Block；待独立交叉验证和公平 PPA |
| os_systolic_fused_attention | M1 | FSA 思想导入 OS；待独立交叉验证和公平 PPA |

其他现有 Engine 也必须走相同证据审计，不能因为历史上默认可推荐就自动视为 M3。

## 10. 首轮执行清单

1. 定义 manifest schema 和 maturity/evidence 枚举；
2. 为所有现有 Engine 建立 manifest；
3. 删除按 Engine 名称硬编码的 research-only 判断；
4. 保留所有 M1 Engine 的原始排名；
5. 增加 comparison-ready 与 product-qualified 两套结果；
6. 增加 fallback、uncertainty 和 calibration provenance；
7. 建立新 Engine contract 模板和 admission checklist；
8. 先用 FSA、BFSA、OFSA 验证完整流程。

## 11. 评审与责任

- Engine author：提交 contract、模型、测试和证据；
- Independent reviewer：复核公式、论文复刻和公平性；
- Arc Model owner：维护统一 evaluator、schema 和排序逻辑；
- Scenario owner：确认需求、workload 和硬约束；
- Signoff owner：批准 M3/M4 升级及 waiver。

同一人可以承担多个角色，但 M2 以上至少需要一次独立复核。
