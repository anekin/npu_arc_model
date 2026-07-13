# 场景 A：LPDDR5 低成本端侧算力扩展 DSE 报告

日期：2026-07-13；Arc Model：v3.2-performance-contract

## 结论

普通交互最低成本推荐：**Block 64×64@800MHz**，标称 28.09 decode TPS、
TTFT 484.96ms、44.3mm²、8.8W。

将审查后合理的 Agent 要求纳入场景 A 时，统一产品推荐：**Block
64×128@1GHz**，标称 27.72 decode TPS、602.47 prefill TPS、TTFT
1488.43ms、47.2mm²、11.0W。面积比 64×64 增加 2.9mm²（约 6.5%），
作用是改善增量 prefill，而不是突破 LPDDR5 decode 带宽上限。Agent 只是
可选能力时，可保留 64×64 低成本 SKU。

## agent-workload-requirements.md 审查

合理并纳入的方向：

- Agent 多次交替 prefill/decode，工具结果构成增量 append。
- 历史前缀高复用；DSE 只模拟新 append，避免重算缓存前缀。
- 单 Agent baseline 仍是 concurrency=1、decode batch=1。
- 完整 KV 放 LPDDR，SRAM 作为热 tile/buffer，需要分层预取和驱逐。
- 同时评估 prefill/decode TPS、TTFT、ITL、E2E 和所有 engine。

需要修正的假设：

| 原假设 | 审查结论 | 场景 A 修正 |
|---|---|---|
| 典型 50–200 轮 | TraceLab 每用户请求平均约 8.8 次 LLM 调用，200 更接近上界 | 16 步仅为保守元数据 |
| 典型 50K–200K，最高 1M | 云端 coding-agent 样本不匹配 4GB 低成本设备 | 32K baseline；128K 仅压力测试 |
| KV 必须放 SRAM | 长上下文物理上放不下 | 完整 KV 放 LPDDR，SRAM 放工作集 |
| INT4 KV=0.5KB/token | 对 Qwen2.5-3B 错误 | FP16=36,864B/token；INT4=9,216B/token |
| 200K INT4 KV≈100MB | 错误 | 约 1.843GB |
| batch=3–5、reasoning×1.5 | 可选策略或缺少依据 | 不进入低成本 baseline；用输出长度和 E2E 建模 |
| 固定 10:1–100:1 比例 | 依赖工具、缓存和模型 | 使用 append/output 分布 |

数据来源：

- [TraceLab](https://arxiv.org/abs/2606.30560) 是主要定量依据：约 4,300
  sessions、350K LLM steps。本报告采用约 875-token median append、
  214-token median output 和高 prefix-cache-hit 特征，但标注 coding-agent
  域偏差。
- [AA-AgentPerf](https://artificialanalysis.ai/articles/aa-agentperf) 支持
  长会话、prefix caching 和 step latency 的重要性。云服务 P95 TTFT 为秒级
  到十秒级，只能说明 2s/5s 暂定门限并非离谱，不能证明端侧市场 SLO。
- [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/v1.0.1/blog/agentic-inference)
  支持 system/tool prefix 高复用和多层 KV 管理，但不是“KV 必须在 SRAM”
  的证据。
- [Qwen2.5-3B 官方配置](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)
  是 36 层、2 KV heads、head_dim=128 的权威输入。
- 未给可核验链接的二次材料不作为硬件参数依据。原文日期 2026-07-14 晚于
  本次审查日期，不能作为来源时效证据。

## 纳入场景 A 的需求

基础子场景 lpddr5_3b：Qwen2.5-3B INT4-only（INT2 不搜索），prompt/output
128/128，LPDDR5-6400 64-bit，效率 75%/85%/90%；硬约束 TPS≥20、
TTFT≤1000ms、area≤80mm²；软目标 TTFT≤500ms；成本优先排序。

Agent 子场景 lpddr5_3b_agent：单步 append=875、output=214、32K 最大上下文；
16 步、90% prefix hit、10K stress append 为元数据；4GB LPDDR、90% 可用、
FP16 KV；硬约束 TPS≥20、TTFT≤5s、area≤80mm²；暂定软目标 prefill TPS≥500、
TTFT≤2s。2s/5s 必须用本地产品 trace 校准。

容量：1.545GB INT4 weights + 1.208GB 32K FP16 KV + 约 0.256GB reserve
=3.009GB，小于 3.6GB 可用容量。

## 物理自洽性与方法

每个子场景、每个角点搜索 1,650 个候选，六次共 9,900 个候选。先过滤硬约束，
再按软目标、area、power、-TPS 排序。本轮补齐 LM head，并以完整模型权重设置
decode memory floor。标称上限 43.52/1.545=28.17 TPS，推荐 28.09/27.72；
75% 上限 24.85 TPS，推荐 24.79/24.48，均未越界。

Block 与 OS Systolic 使用同一架构阶段 roofline；相同点按工程偏好选择 Block，
不表示已有实测证据证明 Block 更快。

## 标称 85%：所有 Engine

基础交互：

| Engine | 状态/目标 | 阵列@MHz | Dec TPS | Pre TPS | TTFT ms | Area | Power |
|---|---|---:|---:|---:|---:|---:|---:|
| block | PASS/MET | 64×64@800 | 28.09 | 284.9 | 485.0 | 44.3 | 8.8 |
| os_systolic | PASS/MET | 64×64@800 | 28.09 | 284.9 | 485.0 | 44.3 | 8.8 |
| systolic | PASS/MET | 128×128@1200 | 20.75 | 1416.5 | 138.6 | 47.2 | 12.1 |
| fsa | PASS/MET | 128×128@1000 | 20.52 | 1731.8 | 122.6 | 47.8 | 11.3 |
| gmma | PASS/MET | 128×128@1000 | 20.51 | 1644.9 | 126.6 | 61.9 | 18.4 |
| tensor_core | PASS/MISS | 64×256@1200 | 24.98 | 165.0 | 815.6 | 53.1 | 15.6 |
| input_stationary | FAIL/MET | 32×64@1200 | 10.60 | 470.3 | 366.6 | 42.8 | 9.4 |
| wmma | FAIL/MISS | 96×96@1200 | 0.06 | 1.1 | 137569.9 | 51.2 | 14.5 |

Agent 增量负载：

| Engine | 状态/目标 | 阵列@MHz | Dec TPS | Pre TPS | TTFT ms | Area | Power |
|---|---|---:|---:|---:|---:|---:|---:|
| block | PASS/MET | 64×128@1000 | 27.72 | 602.5 | 1488.4 | 47.2 | 11.0 |
| os_systolic | PASS/MET | 64×128@1000 | 27.72 | 602.5 | 1488.4 | 47.2 | 11.0 |
| fsa | PASS/MET | 128×128@1200 | 23.75 | 732.2 | 1237.1 | 47.8 | 12.4 |
| systolic | PASS/MET | 192×128@1200 | 20.49 | 1545.9 | 614.8 | 50.1 | 13.8 |
| gmma | PASS/MET | 128×128@1200 | 23.66 | 2545.3 | 386.0 | 61.9 | 20.9 |
| tensor_core | FAIL/MISS | 128×256@1200 | 24.71 | 160.8 | 5480.5 | 64.8 | 22.7 |
| input_stationary | FAIL/MISS | 32×64@1200 | 8.80 | 439.7 | 2103.6 | 42.8 | 9.4 |
| wmma | FAIL/MISS | 32×32@1200 | 0.06 | 1.1 | 782983.2 | 42.4 | 9.2 |

## 带宽角点

| 子场景 | 效率/有效带宽 | 推荐 | Dec TPS | Pre TPS | TTFT ms | Area | Power |
|---|---:|---|---:|---:|---:|---:|---:|
| 基础 | 75% / 38.40GB/s | Block 64×64@800 | 24.79 | 284.54 | 490.18 | 44.3 | 8.8 |
| 基础 | 85% / 43.52GB/s | Block 64×64@800 | 28.09 | 284.85 | 484.96 | 44.3 | 8.8 |
| 基础 | 90% / 46.08GB/s | Block 64×64@800 | 29.74 | 284.98 | 482.79 | 44.3 | 8.8 |
| Agent | 75% / 38.40GB/s | Block 64×128@1000 | 24.48 | 602.26 | 1493.73 | 47.2 | 11.0 |
| Agent | 85% / 43.52GB/s | Block 64×128@1000 | 27.72 | 602.47 | 1488.43 | 47.2 | 11.0 |
| Agent | 90% / 46.08GB/s | Block 64×128@1000 | 29.34 | 602.56 | 1486.21 | 47.2 | 11.0 |

## 后续校准

1. 收集本地 Agent trace，以 step、append/output、cache hit、P50/P95 latency
   替换暂定值。
2. 增加 10K append、128K capacity 和可选 batch=3/5 压力测试，但不升级为
   低成本 baseline。
3. 用 Func Model、RTL 或综合数据校准 Block/OS 搬运、利用率和 PPA。
4. 面积/功耗仍为架构级粗估，不能直接冻结物理规格或成本。

可复现 JSON 位于 sim/results/scenario_a_*lpddr5_{75,85,90}.json。