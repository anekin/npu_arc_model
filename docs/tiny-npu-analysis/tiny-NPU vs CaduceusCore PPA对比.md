---
date: 2026-07-10
tags: [NPU, PPA, LLM, CV, tiny-NPU, CaduceusCore, architecture]
related: [[tiny-NPU vs CaduceusCore 深度对比]], [[Streaming vs LUT Softmax 深度对比]], [[Model Zoo]]
---

# tiny-NPU vs CaduceusCore — PPA × LLM+CV 需求对比

## 一、先看 CV 需要什么、tiny-NPU 缺什么

### 1.1 Conv2D → im2col + GEMM

CV 模型的核心运算是卷积，不是原生矩阵乘。NPU 处理卷积的标准做法是 **im2col**（把滑动窗口展开成矩阵行，卷积变 GEMM）。

| | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| im2col | ❌ 无 — 没有 im2col 硬件，GEMM 只接受标准矩阵 | ✅ SFU/Vector 支持 im2col + tiled GEMM |
| 3×3 Conv | 需软件/编译器做 im2col 转置后喂 GEMM | MXU 可直吃 im2col 展开的矩阵 |
| Depthwise Conv | ❌ 无 — depthwise 是逐通道卷积，GEMM 用不上 | Vector 可处理（逐元素乘） |

**影响**：tiny-NPU 跑 MobileNet/YOLO 需要 CPU 做 im2col 预处理，相当于 GEMM 引擎对 CV 是"半残"状态。

### 1.2 Pooling

| | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| MaxPool | ❌ — Vec ALU 只有 ADD/MUL/MAX/MIN 逐元素 | ✅ SFU maxpool 3c/batch |
| AvgPool | ❌ | ✅ SFU avgpool 3c/batch |
| Global Avg Pool | ❌ | ✅ SFU 8c（归约树） |

### 1.3 CV 专用激活

| | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| h-swish | ❌ | ✅ 4c（clip+add+mul+div pipeline） |
| hard_sigmoid | ❌ | ✅ 3c |

MobileNet/EfficientNet 重度依赖这两个激活函数。tiny-NPU 只能用通用 ALU 模拟，效率差一个数量级。

---

## 二、LLM 需求对照

| 算子 | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| GEMM | ✅ 脉动阵列 | ✅ 8 种 MAC 引擎可选 |
| Softmax | ✅ 专用引擎 3-pass | ✅ SFU LUT 256 条目 |
| LayerNorm/RMSNorm | ✅ 专用引擎 2-pass | ✅ SFU + Vector |
| GELU/SiLU | ✅ 查表法双模式 | ✅ SFU LUT 64 条目 |
| RoPE | ❌ — 未提及 | ✅ CORDIC 12 级流水线 |
| Residual Add | ❌ — Vec 无此操作 | ✅ Vector VRESID |
| INT4 量化 | ❌ — 只 INT8/FP16 | ✅ per-block(g=128) 经 cos_sim 验证 |
| KV Cache | ❌ — 未建模 | ✅ KV tile buffer（SRAM 40%） |

**tiny-NPU 两个硬伤**：
1. **没有 RoPE** — 不能跑 LLaMA/Qwen 等主流 LLM（RoPE 是标准位置编码）
2. **没有 INT4** — 3B+ 模型必须 INT4 量化才能在 LPDDR5 带宽下跑，INT8 太重

---

## 三、PPA 对比

### 3.1 Area

| | tiny-NPU | CaduceusCore S1 (LPDDR5) | CaduceusCore S2 (3D DRAM) |
|:---|:---|:---|:---|
| MAC 阵列 | 1 个 GEMM（尺寸未公布） | block 64×128 (8.2 TOPS) | block 32×1536 (49 TOPS) |
| PE 面积 | 未公布 | 4mm² × 2.94 = **11.8mm²** @12nm | 4mm² × 2.94 × 4.5 = **53mm²** @12nm |
| SFU/Vector | 分散到 4 个引擎 | **集中 SFU+Vector** | 同上 |
| SRAM | 未公布 + scratch | **4MB（含 KV buffer）** | **512KB-1MB**（on-chip 不需要大 SRAM） |
| DRAM PHY | ❌ 未建模 | 14.7mm² | 0（TSV 替代，3.1mm²） |
| PCIe | ❌ 未建模 | 5.9mm² | 5.9mm² |
| **总面积** | **未公布** | **61mm² @12nm** | **66mm² @12nm** |

**tiny-NPU 的 area 数字根本没给**— 教程级项目不做 PPA 分析。但可以从设计决策推断：8-bit SRAM、流式 Softmax、查表 GELU 都是面积最小的选择。如果也用 block MAC 阵列，同类配置下 tiny-NPU 应该比 CaduceusCore 小（没有 SFU/Vector 的冗余面积，没有 DRAM PHY 建模）。

### 3.2 Performance

| 指标 | tiny-NPU | CaduceusCore S1 | CaduceusCore S2 |
|:---|:---|:---|:---|
| LLM 3B decode | ❌ 无 RoPE/INT4 | **23 tok/s** | — |
| LLM 7B decode | ❌ | — | **148 tok/s** |
| LLM 3B TTFT (seq=128) | ❌ 无数据 | **45ms** | — |
| LLM 7B TTFT (seq=1024) | ❌ | — | **160ms** |
| ViT-B/16 @224 | ❌ 无 im2col pool | **2880 FPS** | **2880+ FPS** |
| Qwen-VL ViT 4crop | ❌ | **15 FPS** | **15+ FPS** |
| YOLOv8n | ❌ | ✅ via MXU | ✅ |
| MobileNetV3 | ❌ 无 h-swish | ✅ | ✅ |
| ResNet-50 | 需软件 im2col | ✅ | ✅ |
| SD 1.5 (50 steps) | ❌ 无 im2col | — | **5 img/s** |

### 3.3 Power

两者的功耗都没有公布。但可以从架构反推：

| 因素 | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| 多引擎并发 | 5 引擎可同时跑（记分板） | SFU+Vector+MXU 可以重叠 |
| SRAM 访问 | 每 Softmax 3-5 次 SRAM 读（功耗高） | SFU 批量流水（功耗更集中但总量低） |
| 精度 | INT8/FP16 | INT4 权重（DRAM 流量减半 → 功耗大幅降低） |

**CaduceusCore 的 INT4 是功耗杀手锏**——DRAM 读写是 NPU 最大功耗源，INT4 相比 INT8 减半 DRAM 流量。

---

## 四、需求满足度总评

| 需求类别 | tiny-NPU | CaduceusCore | 差距 |
|:---|:---|:---|:--|
| **LLM 3B decode** | ❌ 缺 RoPE + INT4 | ✅ 23 tok/s | **不可用 vs 生产级** |
| **LLM 7B decode** | ❌ | ✅ 148 tok/s (3D DRAM) | 不在同一量级 |
| **CNN (ResNet/YOLO)** | ⚠️ 缺 im2col/pool/h-swish，需 CPU 补偿 | ✅ 原生支持 | **架构缺失 vs 完整** |
| **ViT** | ⚠️ 可跑但无优化 | ✅ + 12 种 CV 算子 | 明显差距 |
| **VLM (Qwen-VL)** | ❌ | ✅ 15 FPS 4crop | **不可用** |
| **面积可控** | ✅ 设计取向就是最小面积 | ✅ 61-66mm² @12nm | tiny-NPU 更小但不可比 |
| **多场景覆盖** | ❌ 只有一种配置 | ✅ 双技术路径（低成本/高性能） | 不是同一维度的产品 |
| **软件栈** | ❌ 无 | ✅ llama.cpp + ExecuTorch | — |
| **验证完备性** | ❌ 无量化验证/DRAM 分析 | ✅ Arc→Func→E2E→RTL 四阶段 | — |

---

## 五、根因分析

tiny-NPU 和 CaduceusCore 的差距不是"谁设计得更好"的问题——是**定位根本不同**：

| | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| **本质** | 教学级 RTL 参考设计 | 产品级架构探索 |
| **目标** | 展示"NPU 怎么做" | 找"NPU 怎么做最优" |
| **关心的指标** | 能跑通 Transformer | tok/s、tok/mm²、TTFT、带宽利用率 |
| **缺失的维度** | PPA 模型、DRAM 墙壁、SW 栈、量化验证、CV 算子、多场景 DSE | — |

tiny-NPU 的价值在于**设计思想的清晰性**（流式 Softmax、记分板、统一接口），不在于 LLM+CV 需求的完整覆盖。拿产品需求去对标教学项目，本身就问错了问题。

---

## 六、从 tiny-NPU 能抄什么到 CaduceusCore

反向提问更有价值——tiny-NPU 有什么是 CaduceusCore 没有但值得加进来的：

| 特性 | 移植收益 | 优先级 |
|:---|:---|:--|
| 记分板并发 | 替代固件串行调度，MXU+DMA+SFU 可重叠 | 🔴 高 |
| 统一 cmd/done 接口 | 新引擎即插即用，MMIO 地址不再碎片化 | 🔴 高 |
| 流式 3-pass Softmax | 小 seq_len 省 SFU 面积，大 seq_len 走 SFU | 🟡 中 |
| 专用 LayerNorm 引擎 | 和 Softmax/GELU 可并行 | 🟢 低 |
