---
date: 2026-07-10
tags: [NPU, Softmax, architecture, tiny-NPU, CaduceusCore, SFU]
related: [[tiny-NPU vs CaduceusCore 深度对比]], [[tiny-NPU五大计算引擎RTL解剖]]
---

# 流式 3-pass Softmax vs LUT 单 Pass Softmax — 深度对比

## 一、两种架构

### tiny-NPU：流式 3-pass

```
Pass 1: 读 x[0..N] → 找 max → 保存全局最大值
Pass 2: 读 x[0..N] → -max → exp LUT → 累加 sum → exp 结果写入 scratch SRAM
Pass 3: 读 scratch SRAM → × recip(sum) → clamp → 写回 dst SRAM
```

- FSM: **17 个状态**，覆盖 INT8/FP16 双精度
- SRAM: 主 SRAM **8-bit 宽**，FP16 需 2 周期/元素
- scratch SRAM: 16-bit 宽，暂存 exp 中间结果
- LUT: exp 查表（文章未透露条目数，推断 ≤256）
- INT8: Pass 1 后需 S_P1_WAIT 等 reduce_max 模块
- FP16: Pass 1 在线跟踪 max，直接进 Pass 2

### CaduceusCore：LUT 单 pass 分解

```
Step 1: max_reduce(x)        →  Vector (3 cycles/batch)
Step 2: x - max              →  Vector (1 cycle/batch)
Step 3: exp(x_sub) via LUT   →  SFU (12 cycles/batch)
Step 4: sum(exp)             →  Vector (3 cycles/batch)
Step 5: exp / sum            →  SFU (16 cycles/batch)
```

- 5 步流水线，SFU + Vector 协同
- LUT: **256 条目**（RTL），线性插值，覆盖 [-20, 0]
- 精度: abs_tol=2e-3（RTL）/ 1e-5（Func Model 4096 条目）
- SFU width: 每周期处理 width 个元素（批量流水）
- Vector width: 同上

---

## 二、面积 / 存储对比

| | tiny-NPU 流式 | CaduceusCore LUT |
|:---|:---|:---|
| **中间存储** | **scratch SRAM** — 存一个向量 exp 结果（16-bit × N） | **无** — 数据已在 SRAM，不额外占空间 |
| **LUT 存储** | exp ROM（小，≤256 条目） | exp ROM（256 条目 RTL / 4096 Func） |
| **控制逻辑** | 17 状态 FSM | 5 步序列 + pipeline 控制 |
| **SRAM 访问次数** | **3 读 + 1 写** 主 SRAM + 1 读 1 写 scratch | **1 读 + 1 写**（exp→SRAM，div 从 SRAM 读） |
| **总 SRAM 带宽需求** | 高 — 3 次遍历输入 | 低 — 单次遍历 |

**关键差异**：流式方案需要一块独立的 scratch SRAM 做 Pass 2→3 的数据传递。LUT 方案不需要——数据一直在主 SRAM，SFU/Vector 通过 MMIO 桥读写同一地址空间。

---

## 三、延迟对比

假设 N=1024 元素，SFU width=64，频率=1GHz：

| 步骤 | tiny-NPU 流式 | CaduceusCore LUT |
|:---|:---|:---|
| 找 max | Pass 1: N 周期（FP16 在线）/ N+reduce 周期（INT8） | max_reduce: 3 × 16 batch = 48 周期 |
| 减 max | Pass 2 内合并 | sub: 16 周期 |
| exp | Pass 2: N × LUT_latency 周期 | exp: 12 × 16 batch = 192 周期 |
| 累加 sum | Pass 2 内合并 | sum_reduce: 48 周期 |
| 归一化 | Pass 3: N 周期 | div: 16 × 16 batch = 256 周期 |

**粗略估算**（N=1024）：

| | 周期数 | 备注 |
|:---|:---|:---|
| tiny-NPU 流式 | ~3×1024 = **3072 周期** | 3-pass，每元素每 pass 约 1 周期（不含 LUT 流水） |
| CaduceusCore LUT | 48+16+192+48+256 = **560 周期** | 批量流水，N 越大优势越明显 |

**结论**：LUT 方案在大向量上显著更快（5.5× gap），因为批量流水利用了 SFU width 并行度。流式方案受限于"每个元素逐个过 FSM"。

---

## 四、精度对比

| | tiny-NPU 流式 | CaduceusCore LUT |
|:---|:---|:---|
| exp 精度 | 依赖 LUT 条目数和插值方式 | 256 条目 + 线性插值，abs_tol=2e-3 |
| max 精度 | 精确（在线比较或 reduce_max） | 精确 |
| 累加精度 | 依赖 scratch SRAM 位宽（16-bit FP） | BF16/FP32（Vector 累加器） |
| 归一化精度 | 直接除法 | 迭代除法（SFU），有轻微累积误差 |

**tiny-NPU 的精度隐患**：scratch SRAM 只有 16-bit，exp 结果可能溢出（exp(0)=1.0 没问题，但减 max 前的大正值在 Pass 1 已处理）。实际精度取决于 LUT 查表的条目数。文章未详细说明 pass 2 的 exp LUT 实现。

---

## 五、带宽利用率

这是流式方案最大的**劣势**：

```
tiny-NPU: 3 次读 SRAM + 1 次写 SRAM + 1 次读 scratch + 1 次写 scratch
          = 主 SRAM 带宽消耗 = 4 × N elements × 8-bit (INT8) 或 × 16-bit (FP16)

CaduceusCore: 1 次读 SRAM + 1 次写 SRAM
              = 主 SRAM 带宽消耗 = 2 × N × 16-bit (FP16)
```

**在 DRAM 瓶颈场景下**（LPDDR5-64b），流式方案更吃带宽。CaduceusCore 的一次读一次写对 DRAM 压力更小。

但 tiny-NPU 的 scratch SRAM 是**片上**的——如果 scratch SRAM 和主 SRAM 是独立 bank，带宽不冲突。

---

## 六、核心权衡

| 维度 | tiny-NPU 流式 | CaduceusCore LUT | 谁赢 |
|:---|:---|:---|:--|
| **面积** | 需 scratch SRAM（16-bit × max_seq_len），控制简单 | 无需额外 SRAM，但 SFU+Vector 各占 256KB | **场景依赖** |
| **延迟** | O(N) 逐元素 | O(N/width) 批量流水 | **LUT 赢**（大 N 时） |
| **带宽** | 主 SRAM 4× 访问 | 主 SRAM 2× 访问 | **LUT 赢** |
| **控制复杂度** | 17 状态 FSM 独立引擎 | 5 步序列，SFU/Vec 协调 | **流式赢**（更自包含） |
| **精度** | 依赖 LUT 条目数 | 256 条目 2e-3 公差 | **持平** |
| **可扩展性** | scratch 大小随 N 线性增长 | 不随 N 增长 | **LUT 赢** |

---

## 七、移植建议

如果把 tiny-NPU 的流式 Softmax 移植到 CaduceusCore：

### 适合的场景
- **小 seq_len**（< 256）：scratch SRAM 小，3-pass 开销可控
- **面积敏感**（LPDDR5 低成本路径）：去掉 SFU 的 exp/div pipeline，省 SFU 面积
- **独立并发**：Softmax 引擎独立运行时不占 Vector/SFU，可以与 LayerNorm 重叠

### 不适合的场景
- **长 seq_len**（VLM/VLA > 1024）：scratch 需 16KB+，3-pass 延迟恶化
- **高吞吐**（3D DRAM 高性能路径）：LUT 批量流水更快
- **已投入 SFU 设计的项目**：SFU 已有 exp/div，再加 scratch SRAM 是冗余

### 最小移植方案
1. 加一块 4KB scratch SRAM（支持 max 2048 seq_len 的 FP16 exp）
2. 加一个 3-pass Softmax 引擎（~17 状态 FSM）
3. 记分板分配一个 bit
4. 软件可选路径：seq_len < 256 → 流式引擎，否则 → SFU+Vector

---

## 八、一句话总结

**流式 3-pass = 用更多 SRAM 读写换更简单更小的控制逻辑。LUT 单 pass = 用更复杂的流水线换更少的内存带宽和更低延迟。**

tiny-NPU 的哲学是"我不在乎多读几遍 SRAM，我在乎硬件简单"。CaduceusCore 的哲学是"我只读一次写一次，把复杂度交给 SFU 流水线"。两种哲学在面对不同瓶颈（面积 vs 延迟 vs 带宽）时各有胜负。
