# FSA 启发的融合 Attention 架构候选与 DSE 对比

日期：2026-07-14  
Arc Model：v3.4-fused-attention-candidates  
场景：lpddr5_3b、lpddr5_3b_agent

## 结论

Arc Model 新增两个独立候选：

- block_fused_attention：以 Block engine 作为 Projection/FFN 基线，在同一阵列内增加 QK、在线 Softmax、PV 的融合数据通路与调度。
- os_systolic_fused_attention：以 Output-Stationary Systolic engine 作为 Projection/FFN 基线，增加相同的融合 Attention 能力。

这两个候选吸收 FSA 的核心思想，但不是照搬 FSA 论文的专用阵列。原始 fsa 仍作为论文参考点保留。两个新候选当前均为 RESEARCH，不进入默认自动推荐；它们需要原生 RTL 和综合数据完成校准后，才能升级为可推荐架构。

基础场景 A 的正式推荐没有变化，仍为 Block 64×64 @ 800 MHz。短 Prompt 下融合 Attention 收益很小。30K cached-prefix Agent 子场景中融合收益明显，但 Decode TPS 仍未达到 20 TPS，当前搜索中仍没有满足全部硬约束的架构。

## 架构对照关系

| 基线 | 融合候选 | Projection / FFN | Attention |
|---|---|---|---|
| block | block_fused_attention | 完全复用 Block 模型 | QK + online Softmax + PV 融合 |
| os_systolic | os_systolic_fused_attention | 完全复用 OS 模型 | QK + online Softmax + PV 融合 |
| 无直接产品基线 | fsa | 论文外推代理 | 上游 FSA 调度参考 |

当前 Block 与 OS 的分析级 GEMM 周期和 PPA 基线接近，因此两个融合候选在本轮 DSE 中经常得到相同数值。这不是把它们合并成同一架构；独立候选使后续 Block/OS 的数据流、buffer、布线和频率校准能够分别进入 DSE。

## 当前建模假设

- Projection 和 FFN 调用对应的 Block/OS 基线模型，融合能力不会虚构这些算子的加速。
- Attention 使用基线阵列估算 QK/PV，再建模在线 Softmax、外部 Softmax 消除和 QK/PV 重叠。
- 分析级 overlap factor 暂为 0.90，每个 tile 增加 2 个控制周期。
- 因果 Prefill 的计算量按实际可见 query-key pair 折算；K/V 物理流量仍按完整上下文计入。
- cached-prefix 原生 RTL 必须提供 query position offset，当前模型会输出此实现要求。
- 7 nm 的阵列面积基线从 4.00 mm² 增加到 4.24 mm²；增量采用 FSA 论文的 systolic-array 增量作为一阶代理，不代表已综合的整芯片 PPA。
- calibration_tier 为 fsa_inspired_analytical，场景未显式允许前 recommendation_eligible 为 false。

## 场景 A：128-token Prompt

相同 64×64、800 MHz、1 MB L2 的比较：

| Engine | Decode TPS | Prefill TPS | TTFT | Area | Power | 资格 |
|---|---:|---:|---:|---:|---:|---|
| block | 28.05 | 286.35 | 482.668 ms | 44.3 mm² | 8.8 W | REC |
| os_systolic | 28.04 | 286.35 | 482.670 ms | 44.3 mm² | 8.8 W | REC |
| block_fused_attention | 28.08 | 288.13 | 479.860 ms | 44.4 mm² | 8.9 W | R&D |
| os_systolic_fused_attention | 28.07 | 288.13 | 479.862 ms | 44.4 mm² | 8.9 W | R&D |

相对 Block，block_fused_attention 的 Prefill 约提升 0.62%，TTFT 约降低 0.58%，面积增加约 0.1 mm²。该场景主要受 LPDDR5 权重带宽限制，融合 Attention 不是决定性因素。正式推荐仍为 Block 64×64 @ 800 MHz、weight cache、1 MB L2。

## 场景 A Agent：30K Prefix + 875 Append

同为 64×64、1.2 GHz、1 MB L2、无 weight cache 的机制对比：

| Engine | Decode TPS | Prefill TPS | TTFT | Area | Power | 硬约束 |
|---|---:|---:|---:|---:|---:|---|
| block | 14.75 | 83.40 | 10559.817 ms | 44.3 mm² | 10.3 W | TPS、TTFT 失败 |
| os_systolic | 14.75 | 83.40 | 10559.817 ms | 44.3 mm² | 10.3 W | TPS、TTFT 失败 |
| block_fused_attention | 16.22 | 179.11 | 4946.845 ms | 44.4 mm² | 10.4 W | TPS 失败 |
| os_systolic_fused_attention | 16.22 | 179.11 | 4946.845 ms | 44.4 mm² | 10.4 W | TPS 失败 |

相对基线，融合候选 Decode 提升约 10.0%，Prefill 提升约 114.8%，TTFT 降低约 53.2%，面积增加约 0.23%。这说明融合 Attention 对长上下文有明确探索价值。

按 Agent 产品目标排序的首个研究点为 block_fused_attention 128×128 @ 1 GHz：Decode 16.22 TPS、Prefill 588.62 TPS、TTFT 1548.192 ms、面积 53.8 mm²、功耗 14.3 W。它满足 Prefill 和 TTFT 目标，但仍因 Decode TPS < 20 失败，不能作为场景推荐。

## 升级为正式 DSE 候选的条件

两个融合架构已是可搜索、可比较的候选，但目前仅具研究资格。升级为可自动推荐至少需要：

1. 定义 Block/OS 各自的融合数据流、PE 改动、buffer 容量、互连和调度状态机。
2. 完成 FP16 Attention、INT4/INT8 Projection、GQA 和 cached-prefix causal 的功能验证。
3. 用 RTL trace 校准 QK、online Softmax、PV 的 overlap factor 和 tile 控制开销。
4. 用综合结果分别校准面积、频率和功耗；不能继续让两个候选永久共享同一代理值。
5. 用 LPDDR5 端到端 trace 校准 KV 流量、DMA 并发和带宽效率。
6. 校准完成后，在场景中显式加入 allow_experimental_engines，经过回归后再取消研究门控。

## 可复现结果

- sim/results/scenario_a_lpddr5_v34_fused.json
- sim/results/scenario_a_agent_lpddr5_v34_fused.json
- 回归测试：33 passed