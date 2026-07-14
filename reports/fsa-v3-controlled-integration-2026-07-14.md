# FSA 在 Arc Model v3 的受控引入与 DSE 复评

日期：2026-07-14  
Arc Model：`v3.3-fsa-controlled`  
场景：`lpddr5_3b`、`lpddr5_3b_agent`

## 结论

FSA 应保留在 v3，但当前只能作为 `research candidate`，不能参与默认自动推荐。原因不是 FSA 没有价值，而是公开证据只标定了 FP16/FP32、单头、非因果 attention，尚不足以支撑场景 A 的 INT4/INT8 投影、Qwen2.5-3B GQA、cached-prefix causal、64-bit LPDDR5 和整芯片 PPA 结论。

本次重构后，FSA 的优势与局限被分开呈现：它在长上下文 prefill/TTFT 上很有潜力，但在当前 Agent 产品约束下仍未达到 20 TPS decode；Block/OS 同样未同时满足 20 TPS 与 5 s TTFT。因此结果是“当前内存与上下文契约下无可行架构”，不是强行推荐 FSA 或 Block。

## 公开证据边界

- [FSA 论文](https://arxiv.org/html/2507.11331)提出在单个 systolic array 内融合 QK、online softmax 和 PV，并在 PE 中加入向上数据通路、比较器以及 Split/PWL 逻辑。
- 论文的 12% 面积开销是 array-only，不包含 SRAM、DMA、控制核和外部内存 PHY；v3 因此只把 systolic array 基线从 2.00 调为 2.24，整芯片仍独立累加其他模块。
- 论文性能点是 FP16 输入、FP32 累加、128×128 array、单 attention head、head_dim=128、序列 2K–16K、约 820 GB/s，并且不含 causal masking。它不能直接换算成 Qwen2.5-3B 的端到端 TPS。
- [官方 FSA 仓库](https://github.com/VCA-EPFL/FSA)当前内核支持 `seq_q != seq_kv` 和 causal block 跳过；但 cached-prefix 增量请求缺少 query-position offset，不能直接正确表达“30K 已缓存前缀 + 875 新增 token”的 causal mask。

## v3.3 模型改动

1. 工作负载新增 `cached_prefix_tokens`、`attention_bits` 和 `causal_attention`。
2. Agent 子场景明确为 30,000 cached prefix + 875 append，总 prefill K/V context 为 30,875；32K 仍作为容量上限，214-token 输出后总 active tokens 为 31,089。
3. Attention 使用 FP16 数据宽度；INT8 只保留给投影激活，避免把论文 FP16 attention 当作 INT8 流量。
4. 普通 engine 的 causal prefill 计算量按可见 pair 数折算，但 K/V 物理读取仍按完整 30,875 context 计入。
5. FSA attention cycle 改为从上游 `ExecutionPlan.scala` 的 score/value、8-piece exp2 PWL 和 reciprocal/finalize 调度推导，不再使用固定“每 tile 5 cycles softmax”。
6. 修复 FSA `FFN_gate + FFN_up` 被错误当作一次 GEMM 的问题；现在计为两组独立权重和两次计算。
7. FSA 搜索点必须满足 `array_height == head_dim`。无 cached prefix 的原生 causal 路径还要求 square array；对 Agent cached-prefix 路径采用保守矩形性能估算并给出警告。
8. JSON 新增 `recommendation_eligible`、`calibration_tier` 和 `research_candidates`。场景未显式设置 `allow_experimental_engines: [fsa]` 时，FSA 只展示、不推荐。

## DSE 结果

### 场景 A 基础负载：128-token prompt

| 类型 | Engine | Decode TPS | Prefill TPS | TTFT | Area | Power | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| 正式推荐 | Block 64×64 @ 800 MHz, WC, 1 MB L2 | 28.05 | 287.32 | 481.16 ms | 44.3 mm² | 8.8 W | 满足全部约束与目标 |
| FSA 研究点 | FSA 128×128 @ 1 GHz, 1 MB L2 | 20.46 | 1718.23 | 123.38 ms | 47.9 mm² | 11.4 W | 指标通过，但校准不足，不进入自动推荐 |

Decode 的 INT4 权重带宽上限为 `43.52 GB/s ÷ 1.545 GB = 28.17 TPS`；正式推荐的 28.05 TPS 没有越过 memory ceiling。

### 场景 A Agent：30K prefix + 875 append

| 类型 | Engine | Decode TPS | Prefill TPS | TTFT | Area | Power | 失败项 |
|---|---|---:|---:|---:|---:|---:|---|
| 最近的正式候选 | Block 128×256 @ 1.2 GHz, WC, 1 MB L2 | 14.75 | 141.94 | 6232.17 ms | 64.8 mm² | 22.7 W | TPS < 20；TTFT > 5 s |
| FSA 研究点 | FSA 128×256 @ 1.2 GHz, 1 MB L2 | 10.39 | 984.30 | 985.16 ms | 54.5 mm² | 16.4 W | TPS < 20；且仍有校准/功能缺口 |

容量估算为 3.009 GB，低于 4 GB LPDDR 的 3.6 GB 可用容量，因此瓶颈是性能而不是容量。FSA 对 prefill/TTFT 很有吸引力，但不能据此宣称它是最合适的完整 NPU engine：当前投影路径仍是 v3 代理模型，且 decode 不达标。

## 架构决策

- v3 默认推荐集合继续使用经过现有统一口径比较的 Block、OS Systolic 等 engine。
- FSA 保留为专用 attention engine / hybrid NPU 的研究方向，优先探索“通用 INT4 projection engine + FP16 FSA attention engine”，而不是假定一个公开 FSA array 已原生支持全部 LLM 算子和精度。
- 在把 FSA 升级为可推荐 engine 之前，至少需要：cached-prefix causal 正确性、Qwen GQA 映射、INT4/INT8 projection 数据通路或 hybrid 调度、LPDDR5 end-to-end trace、综合后的整芯片 PPA 五项校准。
- Agent 场景还需要产品决策：维持 30K 活跃上下文时增加带宽/并行内存；或定义较小的“性能代表 context”，把 32K 仅作为容量/压力边界。不能继续用 875 token 同时冒充 append 长度和完整 attention context。

## 可复现结果

- `sim/results/scenario_a_lpddr5_v33.json`
- `sim/results/scenario_a_agent_lpddr5_v33.json`
- 回归测试：28 passed