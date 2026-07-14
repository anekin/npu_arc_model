# Output-Stationary 独立建模与 v3.5 DSE 报告

日期：2026-07-14  
Arc Model：v3.5-os-dataflow  
场景：lpddr5_3b、lpddr5_3b_agent

## 结论

v3.4 中的 os_systolic 不是独立性能模型。其 estimate 和 estimate_weight_cache_pair 直接调用 BlockEngine，因此同一配置下 Block、OS 及其融合版本会得到相同周期。v3.5 已移除这项临时等价关系，OS 现在按照 M×N 空间展开、K 时间累加的数据流独立建模。

修正后的结果符合两种架构的基本特性：

- Block 将阵列高度用于 K 维并行归约，对 M=1 Decode 更有效。
- OS 将阵列行列用于 M×N 输出空间并行，K 在 PE 内连续累加；Prefill 的 M 较大时利用率更高，但 M=1 Decode 会浪费大部分阵列行。
- 在场景 A Agent 的同一 128×256 配置中，OS Prefill 比 Block 高 8.2%，TTFT 低 7.3%，但 Decode TPS 低 14.0%。两者不再是重复候选。
- Agent 场景仍没有满足全部硬约束的架构。

基础场景 A 的名义最小面积点变为 OS 32×64 @ 1 GHz，但其 Decode 只有 20.03 TPS，距离 20 TPS 下限仅 0.16%，不能视为稳健产品结论。具有合理余量的低成本选择是 OS 32×64 @ 1.2 GHz，或继续采用 Block 64×64 @ 800 MHz。

## v3.5 OS 模型

对 GEMM (M×K) × (K×N)，H×W OS 阵列采用：

- 空间维度：H 映射 M，W 映射 N。
- 时间维度：每个 PE 对 K 连续累加并保持输出 partial sum。
- 每个有效输出 tile 的 wavefront 周期：
  K + active_M + active_N - 2
- 每个 tile 额外增加 2 个可配置 preload/flush 控制周期。
- 总计算周期为所有 M×N tile 的 wavefront 与控制周期之和。
- 外存 Roofline 读取逻辑 A 和 B 各一次，并与计算周期重叠。
- Gate/Up 不再使用 Block 的双 Weight Register；OS 将其建模为两个独立输出 wavefront，共享 scratchpad 中的 activation。

模型依据：

- [Gemmini 论文](https://people.eecs.berkeley.edu/~alonamid/papers/gemmini-arxiv-1911.09925.pdf)：OS 将 D 预装入 PE accumulator，A/B 流经阵列，C 留在 PE；OS 还需要输入 transposer。
- [Gemmini 官方仓库](https://github.com/ucb-bar/gemmini)：说明 OS/WS 数据流、transposer、scratchpad/accumulator 和 decoupled access-execute。
- [SCALE-Sim 论文](https://arxiv.org/abs/1811.02883)：OS 将输出元素固定到 PE，并通过有限 PE 资源进行空间/时间复用。
- [SCALE-Sim 官方仓库](https://github.com/scalesim-project/SCALE-Sim)：其分析计算周期由 RTL 验证，并分别建模数据流与内存带宽。

## 校准边界

当前模型是 architecture-stage analytical model，不是 Gemmini RTL 的逐周期复制。以下项目仍未校准：

- Transposer 的 backpressure 和额外启动延迟。
- Scratchpad bank conflict、端口数量和本地带宽。
- 多个 GEMM tile 之间的 preload/flush 重叠。
- 当前假设 K 可以连续流过阵列、partial sum 在完整 K 期间留在 PE；若未来 ISA 或 buffer 强制 K 分块，需要增加每个 K chunk 的 wavefront/preload 成本。
- 输出 writeback 与下一层输入复用。
- 物理综合后的最大频率、动态功耗和 accumulator/transposer 面积。

因此 OS 候选的 provenance 标记为 dataflow_analytical。任何通过硬约束但 Decode TPS 裕量小于 5% 的候选都会在 JSON 中记录结构化警告。

## PPA 修正

PPA 代码现在分别读取：

- block_pe_area_mm2
- os_pe_area_mm2
- block_fused_attention_pe_area_mm2
- os_fused_attention_pe_area_mm2

Block 和 OS 的默认 7 nm、128×128 阵列基线目前都为 4.0 mm²，所以相同 H×W 下的面积和功耗仍会相同。这是可独立调整的粗粒度假设，不再是代码错误地引用同一个变量。

Gemmini 的公开综合结果表明 OS 的 PE 内 32-bit accumulator 会带来额外功耗，但没有可直接用于本项目 Block 广播阵列与 OS 阵列的定量换算。因此本轮没有人为制造 PPA 差异，留待 RTL/综合数据校准。

## 场景 A：低成本端侧算力扩展

需求：Qwen2.5-3B、INT4、128-token Prompt、64-bit LPDDR5-6400、85% 有效效率、Decode TPS ≥20、TTFT ≤1 s，设计目标 TTFT ≤500 ms。

完整搜索：1965 个配置，771 个通过应用硬约束，其中 465 个具备正式推荐资格。

| 类型 | Engine / 配置 | Decode TPS | Prefill TPS | TTFT | Area | Power | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| 名义最小面积 | OS 32×64 @ 1 GHz | 20.03 | 667.22 | 241.761 ms | 42.8 mm² | 8.8 W | PASS，但 TPS 余量仅 0.16% |
| 面积优先稳健点 | OS 32×64 @ 1.2 GHz | 24.04 | 800.57 | 201.485 ms | 42.8 mm² | 9.4 W | PASS，20.2% TPS 余量 |
| 功耗/成熟度优先 | Block 64×64 @ 800 MHz, WC | 28.05 | 286.35 | 482.668 ms | 44.3 mm² | 8.8 W | PASS，接近 LPDDR5 带宽上限 |
| 研究候选 | OFSA 32×64 @ 1 GHz | 20.06 | 674.8 | 239.6 ms | 42.9 mm² | 8.9 W | 指标通过，融合模型仍为 R&D |
| 研究候选 | BFSA 64×64 @ 800 MHz | 28.08 | 288.1 | 479.9 ms | 44.4 mm² | 8.9 W | 指标通过，融合模型仍为 R&D |

DSE 按 area、power、-TPS 的字典序目标选择名义点，因此 42.8 mm² 的 OS 1 GHz 排在 Block 前面。但产品选型不应忽略模型误差：0.16% 裕量远低于架构模型精度。建议将 OS 1.2 GHz 和 Block 800 MHz 同时保留到下一轮校准。

## 场景 A Agent：30K Prefix + 875 Append

完整搜索：1980 个配置，没有架构同时满足 Decode TPS ≥20 和 TTFT ≤5 s。

各正式 Engine 距离硬约束最近的点：

| Engine / 配置 | Decode TPS | Prefill TPS | TTFT | Area | Power | 失败项 |
|---|---:|---:|---:|---:|---:|---|
| Block 128×256 @ 1.2 GHz, WC | 14.75 | 141.94 | 6232.17 ms | 64.8 mm² | 22.7 W | TPS、TTFT |
| OS 128×256 @ 1.2 GHz | 12.69 | 153.54 | 5777.57 ms | 64.8 mm² | 22.7 W | TPS、TTFT |
| GMMA 192×128 @ 1.2 GHz | 9.74 | 154.6 | 5763.4 ms | 72.2 mm² | 27.1 W | TPS、TTFT |
| Systolic 256×256 @ 1.2 GHz | 7.60 | 146.1 | 6120.1 ms | 64.8 mm² | 22.7 W | TPS、TTFT |

Block 和 OS 的同配置机制对比：

- OS Prefill 比 Block 提升约 8.2%。
- OS TTFT 比 Block降低约 7.3%。
- OS Decode 比 Block降低约 14.0%。
- 这是 OS 的 M×N 空间映射在 M=875 Prefill 和 M=1 Decode 下产生的方向性差异。

同一 128×256 @ 1.2 GHz、1 MB L2、无 weight cache 的四架构比较：

| Engine | Decode TPS | Prefill TPS | TTFT | Area | Power | 资格 |
|---|---:|---:|---:|---:|---:|---|
| Block | 14.75 | 141.94 | 6232.175 ms | 64.8 mm² | 22.7 W | 正式，但不可行 |
| OS | 12.69 | 153.54 | 5777.565 ms | 64.8 mm² | 22.7 W | 正式，但不可行 |
| BFSA | 16.22 | 1077.64 | 873.616 ms | 66.2 mm² | 23.5 W | R&D，TPS 失败 |
| OFSA | 14.45 | 2329.36 | 444.840 ms | 66.2 mm² | 23.5 W | R&D，TPS 失败 |

修正后的 OFSA 与 BFSA 也不再相同。OFSA 的长 Prefill 利用 M×N OS wavefront 获得更高吞吐，但 M=1 Decode 仍弱于 Block 融合架构。两者都没有达到 20 TPS，不能成为 Agent 场景正式推荐。

## 建议

1. 场景 A 下一轮同时校准 OS 32×64 @ 1.2 GHz 和 Block 64×64 @ 800 MHz。
2. OS 校准重点应放在 M=1 输出头、长上下文 Attention、transposer 和 SRAM bank conflict。
3. Agent 场景的主要矛盾仍是 Decode：融合 Attention 能显著改善 TTFT，但无法解决完整模型权重与长 KV context 的 Decode 性能约束。
4. 在 OFSA/BFSA 升级为可推荐架构前，需要原生 RTL trace 和综合 PPA。
5. 如果产品必须保证 20 TPS，应为 DSE 增加显式性能 guardband，而不是采用刚好越过阈值的名义 PASS 点。

## 可复现结果

- sim/results/scenario_a_lpddr5_v35_os.json
- sim/results/scenario_a_agent_lpddr5_v35_os.json
- 回归测试：36 passed
