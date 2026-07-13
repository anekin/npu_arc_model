---
type: 研究报告
topic: NPU 架构设计
date: 2026-07-14
source: conversation_research
tags: [agent, NPU, prefill, decode, KV-cache, batch-size, edge-inference]
---

# Agent 应用对端侧 NPU 的新需求分析

> 传统 LLM 推理 = 一次问答。Agent 推理 = 多轮迭代 + 工具调用 + 上下文膨胀。
> 这两者的硬件需求完全不同。本报告梳理差异，明确对端侧 NPU 的影响。

---

## 一、Agent 推理 vs 传统推理：五个根本区别

| 维度 | 传统 LLM 推理 | Agent 推理 |
|------|-------------|-----------|
| **轮次** | 1 轮：Prompt → Response | **50-200 轮**：Think→Act→Observe 循环 |
| **上下文增长** | 一次输入，逐 token 输出 | **每轮注入新上下文**（工具返回 → Prefill），序列长度轻松超 100K tokens |
| **Prefill/Decode 比例** | ~1:1（输入≈输出） | **10:1~100:1**（大量 Prefill，少量 Decode） |
| **KV Cache** | 一次性计算，逐 token 追加 | **高频复用**：同一前缀反复出现（system prompt 不变，历史对话递增） |
| **Batch 模式** | 可大量并发（多用户独立对话） | 单 Agent 串行，多 Agent 才可 batch |

数据来源：AA-AgentPerf (Artificial Analysis, 2026.03)、TraceLab (arxiv 2606.30560)、Token Economics for Agent (arxiv 2605.09104)

---

## 二、对 NPU 硬件的六项新需求

### 需求 1：Prefill 能力不再是"配角"

传统端侧 NPU 设计偏重 Decode（M=1 GEMV），因为用户聊天场景 Prefill 只发生一次（输入 prompt）。但 Agent 场景完全不同：

```
用户: "帮我查一下量子计算的药物研发文献"
  → Agent Prefill: prompt + system + 记忆 (500 tokens)
  → Agent Decode: <tool>search(...) (50 tokens)
  → 工具返回: 网页摘要 (3,000 tokens)
  → Agent Prefill: 3,000 tokens ← 第二次 Prefill
  → Agent Decode: 分析结果 (200 tokens)
  → 工具返回: 第二篇网页 (2,500 tokens)
  → Agent Prefill: 2,500 tokens ← 第三次 Prefill
  ...
```

**每次工具调用都触发一次新的 Prefill。** 一个 50 轮的 Agent 对话可能有 20-30 次 Prefill。Prefill 从"一次性开销"变成"高频操作"。

**对 NPU 的要求**：不能只优化 Decode。Prefill 需要足够的 GEMM 算力来处理每次工具返回后的大段文本注入。

---

### 需求 2：KV Cache 复用是性能核心

Agent 推理的特点是**大量前缀重复**：

```
每轮对话的前缀 = System Prompt + 前 N-1 轮历史
                  ↑ 这部分完全不变
新增 = 最新工具返回
```

KV Cache 复用率在 Agent 场景可达 **80-95%**。传统推理引擎的 Prefix Caching 在 Agent 场景下收益巨大。

**对 NPU 的要求**：
- **KV Cache 必须放在片上 SRAM** 或近存（LPDDR5 也可以，但要保证带宽）
- decode token 生成时，只有新增 token 的 KV 需要计算
- **但 Prefill 时**，新注入的工具返回文本（3,000 tokens）需要计算新的 KV——这需要快速的 GEMM

---

### 需求 3：上下文窗口爆炸，SRAM 压力剧增

| 场景 | 典型上下文长度 |
|------|:---:|
| 单次聊天 | 2K-4K tokens |
| RAG 问答 | 4K-8K tokens |
| **Agent 多轮推理** | **50K-200K tokens** |
| Agent + 长文档分析 | 200K-1M tokens |

你的 CaduceusCore 目前 SRAM 只有 4MB。即使用了 INT4 KV Cache（每 token 约 0.5KB），4MB 也只能存 ~8K tokens 的 KV Cache。

**200K tokens 的 KV Cache ≈ 100MB（INT4）。** 这远超片上 SRAM。必须在 LPDDR5 和 SRAM 之间做 KV Cache 分层管理。

**对 NPU 的要求**：
- 片上 SRAM 存**热 KV**（最近 N 轮，高频访问的 prefix）
- LPDDR5 存**冷 KV**（早期对话）
- 需要硬件支持的 KV Cache 预取/驱逐策略

---

### 需求 4：Decode 的 batch size 没变，但利用率问题更严重

Agent 的 Decode 阶段仍然是 M=1 GEMV（单 token 自回归生成）。但 Agent 场景下 Decode 占比更低：

```
传统聊天: Prefill 1 次 + Decode 500 tokens → 利用率低但解码占比大
Agent 推理: Prefill 30 次 + Decode 200 tokens → 计算窗口更碎片化
```

| 场景 | Prefill 比例 | Decode 比例 | MAC 利用率 |
|------|:-----------:|:----------:|:---------:|
| 传统聊天 | 5-10% | 90%+ | ~1%（Decode 占主导） |
| Agent 推理 | 60-80% | 20-40% | 略高（更多 Prefill 可批量处理） |

好消息：Agent 场景下 Prefill 占比高，可以部分缓解 MAC 利用率问题（GEMM 比 GEMV 利用率高）。

**对 NPU 的要求**：不是改 batch size，是**优先保证 Prefill 时 GEMM 能喂饱 MAC 阵列**。Block 64×64 在 M=1 时利用率 1%，但在 Prefill 时 M=3,000 tokens 可以接近满利用率。

---

### 需求 5：Self-Consistency 多路推理引入小 Batch Decode

这是一个新兴需求。Agent 在关键决策点可以同时生成多条推理路径并投票：

```
用户: "这段 RTL 的 bug 在哪？" 
Agent: 同时生成 3 条分析路径（batch=3 decode）
  路径 A: "检查时序，fifo_almost_full 在 clk2 域采样..."
  路径 B: "检查状态机，IDLE→BURST 缺少超时保护..."
  路径 C: "检查跨时钟域，gray_code 计数器跳变..."
→ 投票 → 最一致路径输出
```

**对 NPU 的要求**：
- Batch decode 时**权重只读一次，N 路共享** → DRAM 带宽压力不变
- MAC 利用率从 ~1% (batch=1) 提升到 ~3-5% (batch=3-5)
- 芯片在 batch>1 时的表现会成为差异化竞争力

这不是必须的，但可以作为"增值特性"展示。

---

### 需求 6：Reasoning tokens 改变 Decode 的负载分布

DeepSeek-R1 式的 reasoning 模型在输出答案之前会生成大量 `<think>` 内部推理 token。

```
用户输入 (50 tokens)
  → Reasoning: 可能 2,000-5,000 tokens 的 <think> 链
  → 最终答案: 500 tokens
```

Reasoning 发生在 Decode 阶段（自回归生成），不是 Prefill。**这意味着 Decode 的实际负载可能是可见答案的 5-10 倍。**

**对 NPU 的要求**：
- Decode 带宽需求不能只按"输出 token 数"估算
- Reasoning-heavy agent 的 Decode 占比会比表面看起来高很多
- 如果你是 reasoning 优先的设计（你的 NPU 场景），Decode 带宽不能省

---

## 三、对 CaduceusCore 架构的具体影响

| 设计要素 | 传统假设 | Agent 新需求 | 影响等级 |
|---------|--------|------------|:------:|
| **Prefill 算力** | 一次就够了 | 每轮工具返回都触发 Prefill | ⚠️ 中 |
| **SRAM 容量** | 4MB → 存短上下文 KV | 需要存热 KV + 冷 KV 分层 | 🔴 高 |
| **Decode batch** | M=1 为主 | 保持不变，但加 batch=3-5 参考 | ✅ 低 |
| **DRAM 带宽** | 42.5 GB/s decode 够用 | 增加 Prefill GEMM 压力 | ⚠️ 中 |
| **KV Cache 复用** | 未专门设计 | 核心优化点，决定 Agent 性能 | 🔴 高 |
| **Reasoning 开销** | 未建模 | Decode 实际负载 5-10x | ⚠️ 中 |

---

## 四、建议的评估更新

### 当前 Arc Model 评估
```
decode: M=1, batch=1 → 22 tok/s
prefill: 仅在初始 prompt 时计算一次
```

### 建议新增评估维度

```python
# 1. Agent loop 模拟：多次 Prefill + Decode 交替
seq_q=1, seq_kv=4096   # 初始 prompt (单次 Prefill)
seq_q=3000, seq_kv=4096  # 工具返回注入 (高频 Prefill)
seq_q=3000, seq_kv=8000  # 工具返回注入 + 历史增长

# 2. KV Cache 复用收益
# 对比：每次重算 vs 复用 KV Cache 的 tok/s 差异

# 3. Batch decode 参考值
batch=1 → 22 tok/s (baseline)
batch=3 → ~? tok/s (self-consistency 参考)
batch=5 → ~? tok/s (上限参考)

# 4. Reasoning overhead
# 如果要支持 R1 类 reasoning agent：
decode 带宽预算 × 1.5（reasoning 的额外开销）
```

---

## 五、一句话总结

**Agent 对端侧 NPU 的最大变化不是 batch size，是 Prefill 从一次变成了几十次，KV Cache 从"能放下就行"变成了"分层管理才够用"。**

Batch=1 decode 还是对的——Agent 控制流天然串行。但评估体系要加上多次 Prefill 和 KV Cache 复用的场景，才能反映 Agent 时代的真实负载。

---

*参考来源*

- AA-AgentPerf: 首个 Agent 推理硬件基准 (Artificial Analysis, 2026.03)
- TraceLab: 编码 Agent 工作负载特征分析 (arxiv 2606.30560)
- Token Economics for LLM Agents (arxiv 2605.09104)
- Agent Loop KV Cache 复用分析 (Utilix, 2026)
- PD 分离架构详解 (大模型学习-ing, 2026.06)
- CaduceusCore Arc Model 当前评估代码 (`sim/arc_model.py`)
