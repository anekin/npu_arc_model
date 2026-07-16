# RK3588 + RK1828 双芯片方案 Agent/场景 B 可行性调研

日期：2026-07-16
状态：公开资料研究与模型推算，尚未完成实板 Signoff

## 1. 结论

RK3588 + RK1828 的系统分工与场景 B 很接近：RK3588 负责 Linux、Agent 编排、传感器/多媒体、工具调用和 I/O，RK1828 负责 LLM/VLM 推理，并用 5GB 内置高带宽 DRAM 保存模型与 KV Cache。RKNN3 还提供 Function Call、异步 Session、KV Cache 导入/导出和多 Session 并行等能力，软件形态适合端侧 Agent。

但是，必须把两个验收合同分开：

| 验收对象 | 当前合同 | 调研结论 |
|---|---|---|
| `lpddr5_3b_agent` | Qwen2.5-3B、30K cached prefix + 875 append、32K context、Decode TPS ≥20、TTFT 硬上限 5s/目标 2s | **有条件可行，值得实板验证**。容量明确可行；按公开短上下文实测校准，Decode 约 42–60 TPS，TTFT 中心估计约 1.39s，但公开资料没有 30K+875 实测，不能直接 Signoff。 |
| `onchip_7b` 场景 B | Qwen2.5-7B、1024 prompt、Decode TPS ≥100、TTFT ≤200ms | **不满足当前硬约束**。公开数据只有 128-token 输入：70.26 TPS 已低于 100；158.06ms TTFT 不能外推为 1024-token 达标，按同一实测点校准约为 1.28s。 |

因此，这套双芯片方案可以作为 Arc Model 的重要外部产品校准点，但不能写成“场景 B 已经被商用方案证明可行”。更准确的结论是：**它很可能满足 3B 长上下文 Agent 的硬约束，但不能满足当前 7B/1024/100TPS/200ms 的场景 B 合同。**

## 2. 调研边界与证据等级

- 芯片能力优先采用 Rockchip RK182X Datasheet Rev 1.5 和 Rockchip 官方 RKNN3 发布说明。
- 性能采用 Forlinx `OK3588-C + RK182X`、PCIe、performance mode、Input/New Tokens 均为 128 的实测表。
- 30K/32K 结果为基于 Arc Model 模型规格和公开实测点的推算，不是厂商实测。
- 未找到公开的 RK1828 内置 DRAM 精确带宽、芯片功耗、32K 长上下文曲线、90% prefix-cache 命中率或端到端 Agent trace 数据。

## 3. 双芯片架构是否适合 Agent

### 3.1 推荐的系统分工

| 部件 | 适合承担的工作 |
|---|---|
| RK3588 | Linux/Android、Agent 状态机、Tokenizer/Embedding 回调、JSON/Function Call、工具调用、网络、传感器、视频编解码和非 LLM 控制面 |
| RK1828 | W4A16 LLM/VLM 主推理、模型权重、KV Cache、Prefill/Decode 数据面 |
| PCIe | 命令、输入 embedding、输出 token/logit 和控制同步；不应在每个 token 上搬运权重或完整 KV |

RK1828 数据手册给出 20 TOPS、INT4/INT8/INT16/FP8/FP16/BF16 混合计算、5GB 内置动态内存，以及两个可复用为 PCIe 2.1/USB3 的 PHY。每个 PCIe 2.1 接口为 x1、5.0GT/s；按 8b/10b 编码计算，单链路理论上限为 500MB/s/方向，实际有效载荷会更低。公开开发板资料没有证明两条链路会同时聚合给同一个主机，因此本报告按单链路看待。

这个互联不会成为正常 token 控制流的主要瓶颈，前提是模型和 KV 常驻 RK1828。它也意味着：

- RK3588 的 6 TOPS 与 RK1828 的 20 TOPS 不能简单相加为单模型 26 TOPS；
- RK3588 主内存与 RK1828 5GB 内存不是统一内存；
- 把冷 KV 溢出到 RK3588 LPDDR 需要显式的软件支持，会引入 PCIe 流量和延迟；当前公开 RKNN3 资料没有给出这种分层 KV 的性能保证。

### 3.2 Agent 软件能力

RKNN3 V1.0.0 官方说明已经支持 mRoPE、Function Call、数据传输与推理并行；公开 Session API 包含同步/异步运行、Chat Template 和清理 KV Cache。最新版公开 RKNN3 Toolkit V1.0.4 又增加了 Session pause/resume、KV Cache import/export、多 Session 并行和 streaming weight loading。

这些能力说明 Agent 所需的“多轮会话 + 工具调用 + 缓存状态”有软件接口基础。但“存在 KV API”不等于已经证明 30K prefix 在 16 个 Agent step 中可稳定达到 90% 命中；后者仍需实板 trace 验证。

## 4. 公开性能基线

Forlinx 公布的条件是 `OK3588-C + RK182X`、PCIe、performance mode，Input/New Tokens 均为 128：

| 模型 | 芯片 | TTFT | TPOT | Decode TPS |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | RK182X | 83.44ms | 9.80ms | 102.01 |
| Qwen2.5-7B | RK1828 | 158.06ms | 14.23ms | 70.26 |
| Qwen2.5-VL-3B | RK1828 | LLM 84.69ms；Vision 274.80ms@392×392 | — | 102.58 |
| Qwen2.5-VL-7B | RK1828 | LLM 159.42ms；Vision 279.34ms@392×392 | — | 70.02 |

量化为 W4A16 G32。公开 GSM8K 表中，Qwen2.5-3B 从 FP32 的 79.91 变为 RKNN3 W4A16 的 80.52；这只说明该数据集上没有可见回退，不能替代 Function Call、长上下文召回、VLM/VLA 或具体 Agent 任务的精度验收。

## 5. 3B Agent 定量评估

### 5.1 容量

Arc Model 中 Qwen2.5-3B 使用 3.09B 参数、INT4 权重、36 层、2 个 KV head、head dimension 128、FP16 KV：

| 项目 | 计算 | 容量 |
|---|---|---:|
| INT4 权重 | 当前场景输入 | 1.545GB |
| 32K FP16 KV | `2(K/V) × 36 × 2 × 128 × 2B × 32768` | 1.208GB |
| Runtime reserve | 当前 Arc Model 约定 | 0.256GB |
| 总需求 | 权重 + KV + reserve | **3.009GB** |
| RK1828 可用容量代理 | 5GB × 90% | **4.5GB** |
| 余量 | 4.5 − 3.009 | **1.491GB** |

容量结论为 PASS。该方案比 4GB LPDDR5 场景更关键的优势不是容量本身，而是模型和 KV 都可以放在 RK1828 的内置高带宽内存中。

### 5.2 Decode TPS

公开 128-token 实测为 102.01 TPS。用 1.545GB 权重和短 KV 流量反推，等效数据供给约 158GB/s；30K context 时，每个 Decode token 约需要读取：

- 权重：1.545GB；
- 30K FP16 KV：约 1.106GB；
- 合计：约 2.651GB/token。

保持同等效率时中心估计为 `158 / 2.651 ≈ 59.6 TPS`；再施加 30% 长上下文效率折损约为 41.7 TPS，仍高于 20 TPS 硬门槛。该推导假设权重/KV 主要流量与短上下文实测具有可比性；由于厂商没有公布实际 DRAM 带宽、缓存命中和 30K kernel 效率，这只能判为“高概率满足”，不能判为实测 PASS。

### 5.3 Prefill/TTFT

以 128-token TTFT=83.44ms 校准，短 prompt 对应的近似有效计算吞吐约 9.51TOP/s。30K cached prefix + 875 append 的粗略计算量为：

- 投影/FFN：`2 × 3.09B × 875 ≈ 5.41TOP`；
- 对 30K prefix 的 QK/PV attention：约 7.85TOP；
- 合计约 13.26TOP，对应中心估计 **1.39s**。

这相当于约 629 prefill token/s，名义上同时满足 TTFT≤2s 和 Prefill≥500 token/s 设计目标；但 20%–30% 的长序列 kernel、PCIe、调度和系统 guardband 就会把结果推到 1.7–2.0s 附近。因此：

- TTFT≤5s 硬上限：高概率满足；
- TTFT≤2s 与 Prefill≥500：边界可行，必须测 P95；
- 如果 30K prefix 不能复用而需要每步重新 prefill，则方案不满足当前 Agent 合同。

## 6. 当前场景 B（7B/1024）评估

公开 Qwen2.5-7B 的 70.26 TPS 已经低于场景 B 的 100 TPS 下限。TTFT 158.06ms 是在 128-token 输入下测得，而场景 B 是 1024-token prompt。用同一实测点的有效吞吐外推，1024-token Prefill 的中心估计约 **1.28s**，显著高于 200ms。

| 场景 B 指标 | 要求 | RK1828 公开/推算 | 判定 |
|---|---:|---:|---|
| Decode TPS | ≥100 | 70.26@128 input | FAIL |
| TTFT | ≤200ms@1024 prompt | 158.06ms@128；约 1.28s@1024 推算 | 未实测且高概率 FAIL |
| 容量 | 5GB 内存可容纳 7B+1024 KV | 约 4.13GB 含 reserve | PASS |
| Area | ≤150mm² | 无可信公开芯片面积 | UNKNOWN |
| Power | 当前报告尚无硬上限 | 无可信公开芯片/系统实测功耗 | UNKNOWN |

如果把 7B 也扩展到 32K FP16 KV，需求约为 `3.81 + 1.879 + 0.256 = 5.945GB`，超过 RK1828 的 5GB 原始容量，更超过 4.5GB 可用容量代理。因此“7B + 32K Agent”不能在不压缩 KV、不缩短上下文或不做分层存储的情况下完全常驻 RK1828。

## 7. 成本、功耗和工程风险

双芯片方案的优点是把高带宽内存和 LLM NPU 做成可复用协处理器，主芯片继续使用成熟的 RK3588 生态；缺点是新增芯片、5GB memory package、PCIe/USB PHY、独立供电、PCB、散热和软件服务。公开资料没有可核验的 RK1828 单芯片价格和功耗，开发板的 12V/5A 供电规格也不能当作芯片 TDP，因此目前不能对“低成本”或功耗做量化 Signoff。

主要风险如下：

1. 公开性能只有 128-token 输入，缺少 30K/32K 曲线；
2. Prefix/KV import/export 的功能存在，但 90% 命中率、导入时延和跨 Session 复用未测；
3. 长 attention 的实际 kernel 效率和内置 DRAM 有效带宽未公开；
4. VLM 公开数据没有覆盖持续视频、多摄像头、控制回路和 LLM 并发；
5. 功耗、温升、降频、BOM 与芯片面积缺少公开可审计数据；
6. W4A16 的 GSM8K 精度不能代表 Agent Function Call 和长上下文精度。

## 8. 实板验证与 Signoff 建议

### 8.1 测试配置

- 硬件：RK3588 + RK1828 5GB，PCIe；
- 软件：RKNN3 Toolkit/Runtime V1.0.4 或更高固定版本；
- 模型：Qwen2.5-3B W4A16 G32；
- 先建立 30,000-token prefix KV，再执行 16-step Agent trace；
- 代表 step：875 append + 214 output，max context 32768，batch=1；
- 冷启动、暖 Session、KV import/export、cache miss 和温度稳态分别测量。

### 8.2 必测数据

- TTFT、Prefill TPS、Decode TPS 的 P50/P95/P99；
- 生成前段/中段/末段的 TPS，验证随 context 增长的退化；
- Prefix cache 实际命中率、KV import/export 时间、峰值 device/host memory；
- PCIe payload、Host CPU、RK3588 NPU 与 RK1828 利用率；
- 30 分钟稳态功耗、温度、频率和 throttling；
- Agent Function Call 成功率、长上下文检索和任务完成率。

### 8.3 3B Agent Signoff 门槛

| 项目 | Signoff 标准 |
|---|---|
| 容量 | 32K context 无 OOM，峰值保留明确安全余量 |
| Prefix | 16-step trace 的有效缓存命中率 ≥90%，且无错误复用 |
| Decode | P95 ≥20 TPS |
| TTFT 硬约束 | P95 ≤5s |
| TTFT 目标 | P95 ≤2s；否则只判硬约束 PASS、目标 MISS |
| Prefill 目标 | P95 ≥500 token/s |
| 稳态 | 30 分钟内无崩溃，热降频后的 TPS/TTFT 仍通过硬约束 |
| 精度 | Function Call、长上下文和代表 Agent 任务通过独立精度门槛 |
| 成本/功耗 | 在场景 B 明确系统 BOM 和功耗上限后补签，当前不可省略为“默认通过” |

## 9. 对 Arc Model 的建议

1. 把 RK3588 + RK1828 作为“外部产品校准点”，不是 Arc Engine 候选本身；Arc DSE 应对标其 5GB 容量、短上下文实测 TPS/TTFT 和双芯片互联代价。
2. 外部 benchmark 必须携带模型、量化、input/new token、SDK、连接方式和性能模式；不能继续用 `tps_7b: [59, 180]` 这类缺少 workload 元数据的宽范围判断达标。
3. 为 `onchip_7b` 增加 128/1024/长 context 校准曲线；在拿到 RK1828 实板数据前，70.26 TPS@128 只能用于短上下文交叉检查。
4. 给 Agent 模型增加 prefix-cache policy、KV import/export/迁移开销和 inter-chip traffic 字段；当前 DSE 默认模型/KV 在同一 memory domain，不能直接代表双芯片系统。
5. 场景 B 的 100TPS/200ms 合同保持不变，除非有应用需求证据，而不是为了让 RK1828 基线“看起来达标”而放宽。

## 10. 主要来源

- [Rockchip RK1820/RK1828 Datasheet Rev 1.5](https://rockchip.fr/RK182X%20datasheet%20V1.5.pdf)
- [Rockchip：RKNN3 SDK V1.0.0 发布说明](https://www.rock-chips.com/a/cn/news/rockchip/2026/0309/2163.html)
- [airockchip/rknn3-toolkit（V1.0.4 能力与变更）](https://github.com/airockchip/rknn3-toolkit)
- [Forlinx RK1820/RK1828 AI Accelerator Development Guide](https://docs.forlinx.net/ai-accelerator/rk1820_rk1828/RK1820_RK1828_AI_Accelerator_Development_Guide.html)
- [Firefly RK182X Development Kit Specifications](https://download.t-firefly.com/Spec/Suite/RK182X-Development-Kit_Specifications_CN.pdf)
- 本仓库：`sim/config/scenarios.yaml`、`sim/model_specs.py`、`sim/results/lpddr5_3b_agent_latest_2026-07-15.json`
