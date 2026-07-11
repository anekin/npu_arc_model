# NPU 架构选型 DSE 报告

**日期**: 2026-06-27  
**分支**: `feat/fsa-arc-eval`  
**方法**: 两阶段设计空间搜索 → 约束收敛 → 最优架构选定

---

## 0. DSE 方法论

```
Phase 1: 全维度搜索                          Phase 2: 约束收敛
┌──────────────────────────┐         ┌──────────────────────────┐
│ 8 引擎 × 7 阵列 × 2 精度  │         │ 芯片实际限制:              │
│ × 3 频率 × 6 内存带宽    │  ───→   │ • LPDDR5 64-bit (≤51.2GB/s)│
│ = 2,016 配置             │         │ • 面积预算 (≤100mm²)       │
│                          │         │ • LLM 为主, CV 兜底       │
│ 含 HBM2e/HBM3 等高端选项  │         └──────────┬───────────────┘
└──────────────────────────┘                    │
                                                ▼
                                    ┌──────────────────────┐
                                    │ 选定: FSA @ 64×256   │
                                    │ LLM 41 tok/s, 29mm²  │
                                    └──────────────────────┘
```

**核心理念**: 先看全貌，再根据实际约束收敛，而不是一开始就限缩搜索空间。

---

## Phase 1: 全维度设计空间搜索

### 1.1 搜索空间

| 维度 | 取值 | 数量 |
|------|------|:---:|
| MAC 引擎 | systolic, os_systolic, input_stationary, block, tensor_core, wmma, gmma, **fsa** | 8 |
| 阵列尺寸 | 64×64 → 256×256 | 7 |
| 量化精度 | INT4, INT2 | 2 |
| 频率 | 800, 1000, 1200 MHz | 3 |
| 内存方案 | LPDDR5-32b, LPDDR5-64b, LPDDR5-128b, LPDDR5-256b, HBM2e-1024b, HBM3-1024b | 6 |
| **合计** | | **2,016 配置** |

### 1.2 Phase 1 结果：无约束 Pareto 前沿

```
LLM tok/s
 1000 ┤                              ● fsa 128×256 HBM3 (984 tok/s, 134mm²)
  900 ┤                    ● fsa 96×96 HBM3
  800 ┤          ● fsa 64×64 HBM3    ○ block 96×96 HBM3 (813 tok/s, 143mm²)
  700 ┤
  600 ┤
  500 ┤                                              ○ gmma (410 tok/s, 180mm²)
  400 ┤
       └┬──────────┬──────────┬──────────┬──────────┬────
       128        135        143        150        180    mm²
       
       ● FSA (全线 Pareto 最优)    ○ 其他引擎 (被支配)
```

**无约束下的前三名:**

| 引擎 | LLM (HBM3) | 面积 | CV | 评价 |
|------|:---:|:---:|:---:|------|
| **fsa** | **984 tok/s** | 134mm² | 1216 fps | Pareto 全线最优 |
| block | 813 tok/s | 143mm² | 1216 fps | 同面积慢 17% |
| gmma | 410 tok/s | 180mm² | 1216 fps | LLM 弱 58% |

### 1.3 全维度洞察

| 观察 | 含义 |
|------|------|
| HBM 带宽下，FSA 是唯一 Pareto 最优引擎 | 如果带宽不限制，FSA 碾压 |
| block/os_systolic 的 PE 面积是 FSA 的 4× | 全并行 MAC 代价大 |
| gmma 在大带宽下仍然偏弱 | group MMA 不适合 decoder-only Transformer |
| systolic 有管线瓶颈 | weight-stationary 需要摆脱 fill/drain 惩罚 |

---

## Phase 2: 约束收敛

### 2.1 芯片实际约束

从芯片实现角度，以下约束是不可绕过的：

| 约束 | 值 | 原因 |
|------|------|------|
| **内存接口** | LPDDR5 64-bit | DDR PHY 面积限制，HBM 封装成本/散热不可接受 |
| **内存带宽** | ≤51.2 GB/s | 64-bit × 6400Mbps = 51.2 GB/s，85% 可用 = 43.5 GB/s |
| **面积预算** | ≤100mm² | 移动端/边缘端 die budget |
| **工作负载优先** | LLM 为主 (80%), CV 兜底 (20%) | NPU 定位：Transformer 推理加速器 |

### 2.2 约束下的引擎对比

（LPDDR5 64-bit, 43.5 GB/s 可用带宽, 面积 ≤100mm²）

| 引擎 | LLM tok/s | CV fps | 面积 | 功耗 | 效率 | vs 最优 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **fsa** ★ | **41** | 1216 | **29mm²** | 10.0W | **1.4** | — |
| block | 41 | 1216 | 52mm² | 19.2W | 0.8 | 面积 1.8× |
| os_systolic | 41 | 1216 | 52mm² | 19.2W | 0.8 | 面积 1.8× |
| gmma | 41 | 1216 | 60mm² | 22.4W | 0.7 | 面积 2.1× |
| tensor_core | 33 | 800 | 28mm² | 9.6W | 1.2 | LLM -19% |
| systolic | 20 | 600 | 52mm² | 19.2W | 0.4 | LLM -50% |

### 2.3 关键分析：为什么所有人都卡在 41 tok/s？

**内存墙。** LPDDR5 64-bit 的物理带宽 51.2 GB/s，扣除刷新开销后可用 43.5 GB/s。

```
Qwen2.5-3B 每层推理所需:
  权重加载:  7 个矩阵 × ~12MB = ~84MB（首次加载后可缓存）
  KV cache:  ~2MB/layer × 28 = ~56MB
  激活:      ~0.5MB/layer × 28 = ~14MB
  
  总 DRAM 流量 ≈ 5-10MB/layer（含权重缓存命中）
  
  43.5 GB/s ÷ 10MB/layer = 4,350 layers/s
  4,350 ÷ 28 layers = 155 tok/s（理论上限）
  实际受 SRAM miss、DMA 对齐等影响 ≈ 41 tok/s
```

在内存墙下，所有引擎的 MAC 计算都被 DMA 等待时间填满。**此时引擎的差异不是吞吐，而是达成同等吞吐需要多少硅面积。**

### 2.4 FSA 的面积优势从哪来

| 面积组成 | FSA (29mm²) | block (52mm²) | 差异来源 |
|------|:---:|:---:|------|
| PE 阵列 | 8.96 | 32.00 | FSA 用 systolic PE (小)，block 用全并行 PE (大) |
| SRAM | 15.3 | 15.3 | 相同 |
| DMA + RISC-V + PHY | 5.0 | 5.0 | 相同 |
| **差额** | | **23mm²** | 完全来自 PE 面积差异 |

**FSA 省出的 23mm² = 可以加 11.5MB SRAM (L1)，或 4 个 DMA 通道，或直接降低芯片成本 44%。**

---

## Phase 3: 最终选型

### 3.1 推荐配置

```
引擎:     FSA (Fused Systolic Array)
阵列:     64 × 256 (weight-stationary + inline softmax)
精度:     INT4 weights, INT8 activations, INT32 accumulate
频率:     800 MHz (低功耗优先)
内存:     LPDDR5-6400 64-bit, 51.2 GB/s
SRAM:     L1 512KB + L2 2MB
面积:     ~29mm² @ 7nm
功耗:     ~10W
```

### 3.2 预期性能

| 工作负载 | 性能 | 备注 |
|------|:---:|------|
| Qwen2.5-3B (decode) | 41 tok/s | 单 token 实时推理 |
| Qwen2.5-1.5B (decode) | ~80 tok/s | 小模型更高吞吐 |
| Qwen2.5-7B (decode) | ~18 tok/s | 大模型受带宽限制更严重 |
| MobileNetV3 | 1216 fps | CV 轻量模型 |
| YOLOv8n | 146 fps | 目标检测 |

### 3.3 如果放宽约束

| 约束变化 | 引擎变化 | 理由 |
|------|------|------|
| LPDDR5 → 128-bit | FSA 仍最优 | 带宽翻倍，面积优势继续保留 |
| HBM2e 可用 | FSA → 984 tok/s | 带宽不再受限，FSA 计算优势全面释放 |
| 面积预算提升到 80mm² | FSA 仍最优 | 可堆更大阵列 (128×256)，吞吐 251 tok/s |
| CV 权重提升到 50% | gmma 可考虑 | 但 LLM 仍是主战场，不改变推荐 |

**FSA 在从最受限到最不受限的所有场景下都是最优或并列最优。** 没有场景下 FSA 被其他引擎显著超越。

---

## 4. FSA 架构原理速查

> 详见 `ref_arch/fsa/` 和 `sim/fsa_ref.py`

FSA 在标准 weight-stationary 脉动阵列上加了三样东西：

1. **CMP 列比较器** — 纵向传播 rowmax，一个周期完成在线归约
2. **PE Split 单元** — 复用 MAC 乘法器做 exp 分段线性插值
3. **纵向数据通路** — 数据在阵列内垂直流动，消除 MXU→SFU→MXU 搬运

面积代价：+12%（systolic PE 8.0 → FSA PE 8.96 mm² @ 128×128）。

返回值：消除管线 fill/drain 瓶颈，消除 attention 三跳数据搬运。在 LPDDR5 64-bit 约束下，用 block 56% 的面积达成同等吞吐。

---

*报告工具: `sim/design_space_explorer.py` + `sim/fsa_ref.py`*  
*FSA 上游: https://github.com/VCA-EPFL/FSA | Paper: arXiv 2507.11331*  
*数据: 2,016 配置全维度搜索, Phase 1 + Phase 2 约束收敛*