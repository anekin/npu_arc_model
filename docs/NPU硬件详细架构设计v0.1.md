# NPU 硬件详细架构设计 v0.6

> 基于 Phase 1 MRD/PRD 的技术指标 + DSE 验证，进入微架构设计
> 2026-06-22
> **v0.6 更新**：MXU 从 Systolic 改为 64×64 Block Engine（DSE 验证）
> **v0.2 更新**：NPU 改为 IP 核架构，支持单核/多核实例化

---

## 一、设计原则

1. **不重复造轮子**：TPUv1/OpenTPU 验证过的模块直接复用或参考，专注重写 Transformer 特有部分
2. **IP 核化**：NPU 核心作为可参数化 IP，SoC 客户可实例化 1-N 个核，共享外设
3. **片内多核扩展**：核间走片上 Crossbar/FIFO（~500 GB/s），不走 PCIe P2P（7.88 GB/s）
4. **模块化**：每个模块独立验证，通过标准 AXI4 总线互联
5. **混合精度**：权重 INT4，激活 BF16，累加 INT32，KV Cache INT8
6. **约简**：不做多余的灵活性和可编程性——这是一个推理专用加速器

---

## 二、顶层架构

### 2.1 芯片顶层框图（多核配置示例：4 核）

```
┌──────────────────────────────────────────────────────────────┐
│              NPU SoC @ TSMC 12nm                              │
│                                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                       │
│  │ 核₀  │ │ 核₁  │ │ 核₂  │ │ 核₃  │  ← 每个核：MXU+SFU+L1 │
│  │64×  │ │64×  │ │64×  │ │64×  │                       │
│  │64   │ │64   │ │64   │ │64   │                       │
│  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘                       │
│     │        │        │        │                             │
│     ├────────┼──┬─────┼────────┤  ← 核间 FIFO（流水线并行）  │
│     │        │  │     │        │                             │
│     └────┬───┴──┴─────┴───┬────┘                             │
│          │                │                                   │
│     ┌────┴────────────────┴────┐                              │
│     │   Crossbar / NoC 互联     │  ← ~500 GB/s 核间+L2+外设  │
│     │   共享 L2 SRAM 2-8 MB    │                              │
│     └────┬────────────────┬────┘                              │
│          │                │                                   │
│  ┌───────┴───────┐ ┌──────┴──────────┐                       │
│  │ LPDDR5 控制器  │ │ RISC-V + DMA    │                       │
│  │ 64/128-bit    │ │ + PCIe EP        │                       │
│  │ (共享)        │ │ (共享)           │                       │
│  └───────────────┘ └─────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 核内结构（单 NPU 核心）

```
┌─────────────────────────────────┐
│         NPU Core                │
│  ┌──────┐  ┌──────┐  ┌───────┐ │
│  │ MXU  │  │ SFU  │  │ KV    │ │
│  │64×   │  │7算子 │  │ Cache │ │
│  │64    │  │      │  │ 256KB │ │
│  └──┬───┘  └──┬───┘  └───┬───┘ │
│     │         │          │     │
│  ┌──┴─────────┴──────────┴───┐ │
│  │    L1 SRAM 512KB (双bank) │ │
│  └────────────┬──────────────┘ │
│               │ ← 核间 FIFO    │
│  ┌────────────┴──────────────┐ │
│  │   AXI4 从接口（挂 Crossbar）│ │
│  └───────────────────────────┘ │
└─────────────────────────────────┘
```

### 2.3 模块清单（按配置缩放）

### 2.3 模块清单（按配置缩放）

| 模块 | 每核 | 共享 | 复用程度 | 开发量 |
|------|:---:|:---:|------|:---:|
| MXU（64×64 Block Array，broadcast-based） | ~10mm² | — | 参考 TPUv4 VMU / OpenTPU PE | 中等 |
| L1 SRAM 512KB（双 bank） | 1mm² | — | 标准 SRAM 宏 | 低 |
| SFU（7 算子） | 2mm² | — | 中科院+自研 | 高 |
| KV Cache SRAM 256KB | 1mm² | — | 自研 | 高 |
| 核间 FIFO 4KB×2 | 0.5mm² | — | 标准 FIFO IP | 低 |
| **每核心小计** | **~14.5mm²** | | | |
| | | | | |
| Crossbar / NoC | — | 1.5mm² | 标准互联 | 低 |
| L2 SRAM 2-8 MB | — | 5-20mm² | 标准 SRAM | 低 |
| RISC-V 核 + 指令发射 | — | 1mm² | Coral NPU 范式 | 低 |
| DMA 引擎 | — | 1mm² | 业界参考 | 中等 |
| PCIe Gen4 x4 EP | — | 3.5mm² | 商用 IP（采购） | 无 |
| LPDDR5 控制器+PHY | — | 7mm² | 商用 IP（采购） | 无 |

### 2.4 按核心数缩放的总面积

| 配置 | 核心×面积 | +共享 | 总面积 | INT8 TOPS | Decode (M=1) | 备注 |
|------|:---:|:---:|:---:|------|------|------|
| 1 核 | 1×14.5 | 13.7 | **~28.2 mm²** | 6.6 | **29.6 tok/s** ✅ | Block 64×64，DRAM-bound |
| 2 核 | 2×14.5 | 16 | **~45 mm²** | 13.2 | 56 tok/s | DP, -10% contention |
| 4 核 | 4×14.5 | 19 | **~72 mm²** | 26.4 | 100 tok/s | |
| 8 核 | 8×14.5 | 23 | **~139 mm²** | 52.8 | 157 tok/s | |

> **v0.5 更新（DSE 验证）**：基于 DSE 确认的 Block 64×64 架构。单核 29.6 tok/s 已达 LPDDR5-6400 有效带宽上限（43.5 GB/s 下的物理极限 ~35 tok/s，扣除 KV Cache/SFU 开销后 ~29.6 tok/s）。Block engine 零 pipeline fill/drain 开销，M=1 decode 即 DRAM-bound，无需 continuous batching 恢复效率。多核数据并行受共享 DRAM 带宽限制，缩放因子约 1.9×/3.4×/5.3×。

> 共享增量含：L2 扩容（2/4/8/12MB）+ Crossbar 端口 ×N + 更多 DMA 通道

---

## 三、模块详细设计

### 3.1 MXU：64×64 Block Array（broadcast-based）

#### 设计参考
- **TPUv4 VMU**（Google, ISCA 2021）：Block-based 2D tiling，全并行 MAC，零流水线填充
- **OpenTPU**（UCSB ArchLab）：PE 模块和 MAC 单元可复用
- **TensorCore 小块思想**（NVIDIA A100）：16×16 子块并行，本设计放大为 64×64 单块
- **Broadcom/Google Axion**：Block engine 用于 LLM 推理，端侧适配

#### 结构

```
         ← weights 预加载到 PE 寄存器
    ┌─────────────────────────────┐
    │  PE₀,₀  PE₀,₁  ... PE₀,₆₃ │ ← activation 广播到整行
    │    ↓      ↓         ↓      │
    │  PE₁,₀  PE₁,₁  ... PE₁,₆₃ │
    │    ↓      ↓         ↓      │
    │   ...    ...       ...     │
    │    ↓      ↓         ↓      │
    │  PE₆₃,₀ PE₆₃,₁ ... PE₆₃,₆₃│
    └─────────────────────────────┘
         ↓ 部分和向下累积（树规约）
```

每个 PE：**INT4×INT8 → INT32 乘累加**，一个周期完成。

**与 Systolic Array 的关键区别**：Block engine 没有对角线 pipeline fill/drain。所有 PE 在同一周期开始计算——activation 广播到每一列，weight 驻留在 PE 本地寄存器。对 M=1 decode 这意味着零 tiling 开销，单 token 也能达到 100% PE 利用率。

#### 关键设计参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 阵列尺寸 | 64×64 | DSE 验证端侧最优，DRAM 受限下更大阵列无收益 |
| MAC 数 | 4,096 | |
| 频率 | 800 MHz | TSMC 12nm 保守 |
| INT8 算力 | 6.6 TOPS | 64²×2×0.8 GHz |
| 权重精度 | INT4 | 每个 PE 存 4-bit 权重 |
| 激活精度 | INT8 | 广播到阵列 |
| 累加精度 | INT32 | 避免溢出 |
| 数据流 | Block Broadcast | 激活广播，权重驻留 PE |
| 流水线开销 | **零** | 全并行，无 fill/drain 周期 |
| 双缓冲 | 是 | 一组运算时另一组加载权重 |

#### 为什么 64×64 而不是更大阵列

设计空间探索（详见第九章）表明，在 LPDDR5-6400（51.2 GB/s）带宽约束下，更大阵列相对 64×64 **几乎无 decode 性能收益**：

| 引擎 | 阵列 | 3B tok/s | 面积 |
|------|:---:|:--------:|:----:|
| Block | 64×64 | 29.6 | 28.2 mm² |
| Block | 96×96 | 29.0 | ~38 mm² |
| Block | 更大（翻倍） | 30.0 | ~52 mm² |

性能仅提升 1%，面积增加 84%。DRAM 是天花板，阵列再大也没用。64×64 在面积和功耗上是最优选择。

#### 与 Systolic Array 的差异

| 维度 | Systolic（TPUv1） | Block（本设计） | 原因 |
|------|------|------|------|
| 数据流 | Weight-Stationary，对角线流水 | Broadcast Block，全并行 | M=1 decode 零流水线开销 |
| 阵列 | 256×256 | 64×64 | 面积/功耗/带宽匹配 |
| 精度 | INT8×INT8 | INT4×INT8 | 3B INT4 量化 |
| 流水线开销 | 385 cycles/tile | 0 | 全并行，无 fill/drain |
| 控制 | 主机 CPU 逐指令发 | RISC-V 核本地调度 | 降低 PCIe 交互延迟 |

#### 开发策略
1. 从 OpenTPU Verilog 代码提取 PE 模块和 MAC 单元
2. 修改 PE 内部 MAC 为 INT4×INT8→INT32
3. 配置为 64×64 全并行 Block 拓扑（非 systolic 对角线连接）
4. 增加 activation 广播总线：单周期将激活值分发到所有 PE
5. 增加双缓冲控制逻辑
6. 加 INT4 解包逻辑（每个字节存 2 个 INT4 权重）
7. 移除 systolic pipeline fill/drain 控制状态机

---

### 3.2 Unified Buffer + Accumulator

#### 设计参考
- **TPUv1**：Unified Buffer 24MB，双缓冲，直接连 MXU
- **TPUv2+**：合并 Accumulator 和 Activation Storage 为 Vector Memory

#### 结构

```
   LPDDR5                    MXU
     │                        │
     ▼                        ▼
┌─────────┐  ┌─────────────────────┐
│  DMA    │  │  Unified Buffer      │
│  Engine │──┤  (Scratchpad SRAM)   │
└─────────┘  │  4 MB, 16 Banks      │
             │  双缓冲 Ping/Pong    │
             └──────────┬──────────┘
                        │
                   ┌────┴────┐
                   │ Accum   │ ← INT32 → BF16 转换
                   │ Regs    │
                   └────┬────┘
                        │
                    ┌───┴───┐
                    │  SFU  │  ← Activation pipeline
                    └───────┘
```

#### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 容量 | 4 MB | 约 TPUv1 的 1/6，匹配 64×64 阵列 |
| Bank | 16 | 并行读写避免冲突 |
| 双缓冲 | Ping-Pong | 一区运算，一区 DMA 加载 |
| 接口 | 读写 256-bit/cycle | 喂饱 64×64 阵列 |

> **注：L1 SRAM 与 Unified Buffer 的区分**
> 
> - **L1 SRAM（512 KB，双 bank）**：每核心本地紧耦合存储，MXU/SFU 直接访问，用于权重的双缓冲与激活值暂存。两个 256 KB bank 各有一组独立读写端口，Ping-Pong 切换以掩盖 DMA 延迟。
> - **Unified Buffer（4 MB scratchpad SRAM）**：每核心独立的中等容量缓冲区，承接 DMA 引擎批量搬移的权重/数据块，为 MXU 提供输入数据流并接收部分和输出。相比 TPUv1 的 24 MB，本设计匹配 64×64 阵列规模，面积效率更高。

---

### 3.3 SFU：Special Function Unit

#### 设计参考
- **中科院（2024）**："Hardware-oriented algorithms for softmax and layer normalization" — 统一 Softmax+LayerNorm 硬件架构
- **Softmax-Hardware-Accelerator**（GitHub）：流水线化 softmax
- **TPUv2 Vector Unit**：可编程向量单元替代 TPUv1 固定函数数据通路

#### 算子清单与硬件方案

| 算子 | 硬件方案 | 流水线级数 | 面积 |
|------|---------|:---:|:---:|
| **Softmax** | 指数查表 + 分段减法归一化 | 8 | ~0.4mm² |
| **LayerNorm** | 并行均值/方差 + 融合乘加 | 6 | ~0.3mm² |
| **GELU** | 分段查表 + 线性插值 | 4 | ~0.2mm² |
| **ReLU** | 组合逻辑（max(0,x)） | 1 | ~0.05mm² |
| **MaxPool 2×2** | 滑动窗口 4 选 1 比较器 | 3 | ~0.1mm² |
| **AvgPool 2×2** | 加法 + 移位（÷4） | 3 | ~0.1mm² |
| **RoPE** | CORDIC 旋转 | 12 | ~0.3mm² |
| **SiLU** | 复用 GELU 查表 | 4 | ~0.05mm² |

总 SFU 面积 ~1.5mm²。非同时激活——算子间共享部分硬件。

#### 中科院统一架构（参考实现）

论文方案将 Softmax 和 LayerNorm 融合为单一硬件通路——先算均值/方差（LN 前半段），再用指数减法归一化（Softmax 后半段），省了中间 buffer。我们在这个基础上加 CNN 算子即可。

---

### 3.4 KV Cache Manager

#### 参考
- **腾讯（2025 专利）**："多轮推理 KV-cache 内存碎片化导致 L2 命中率下降"
- **TPI-LLM**（OpenReview 2025）：多边缘设备 tensor parallelism 的 KV cache 调度
- **HeadInfer**（arXiv 2025）：按 head 卸载 KV cache，单卡 4090 跑 4M token 上下文

#### 设计

```
  ┌─────────────────────────────────────┐
  │            KV Cache Manager          │
  │                                     │
  │  ┌──────────┐   ┌───────────────┐   │
  │  │  页表     │   │  KV SRAM Cache │   │
  │  │  (硬件)   │   │  256 KB        │   │
  │  │  64 条目  │   │  (最近 N token) │   │
  │  └────┬─────┘   └───────┬───────┘   │
  │       │                 │           │
  │  ┌────┴─────────────────┴───────┐   │
  │  │        DMA 搬移逻辑           │   │
  │  │   SRAM ↔ LPDDR5 (KV区)       │   │
  │  └──────────────────────────────┘   │
  └─────────────────────────────────────┘
```

| 参数 | 值 | 说明 |
|------|-----|------|
| KV 存储主位置 | LPDDR5 专用区域 | 2000 token × 3B × 2(K+V) ≈ 96 MB |
| KV Cache SRAM | 256 KB | 缓存最近 16 个 token，命中率 >85% |
| 格式 | INT8 | 省带宽 |
| 页表 | 硬件 64 条目 | RISC-V 维护，DMA 搬移 |
| 地址管理 | 连续分配 | 避免碎片（参考腾讯专利的教训） |

#### KV Cache 为什么不永久放 SRAM

2000 token 的 KV cache 需 96 MB，远超 2 MB L2 SRAM。只能放 LPDDR5。256 KB SRAM 缓存最近 token 足够——Transformer decode 只访问当前 token 和最近几个 token 的 KV。

---

### 3.5 DMA 引擎

#### 功能

- LPDDR5 ↔ L2 SRAM 数据搬移
- 支持 scatter-gather（分散-聚合）描述符链
- 双通道：权重加载通道 + 数据搬移通道
- 与 MXU 运算流水线化——运算时不阻塞 DMA

#### 参数

| 参数 | 值 |
|------|-----|
| 通道数 | 2 |
| 最大 burst | 256 bytes |
| 描述符队列深度 | 16 |

---

### 3.6 RISC-V 主控核

#### 设计参考
- **Coral NPU**（Google）：RISC-V RV32IMF + AXI4 总线 + 4 级流水线

#### 功能

1. **指令解析**：从指令 FIFO 取指，解码，分派到 MXU/SFU/DMA
2. **DMA 描述符管理**：构建描述符链，下发到 DMA 引擎
3. **异常处理**：MXU 溢出、DMA 超时、PCIe 错误
4. **不参与计算**：所有数值运算由 MXU/SFU 执行

#### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| ISA | RV64IMAFD | 64-bit，支持原子操作 |
| 流水线 | 4 级 | 取指-译码-执行-写回 |
| I-Cache | 16 KB | 指令缓存 |
| D-Cache | 16 KB | 数据缓存 |
| 中断 | 16 个外部中断源 | |

---

### 3.7 NPU 指令集架构（ISA）

CISC 风格，一条指令 = 一个完整算子。32-bit 定长指令字。

| OpCode | 指令 | 格式 | 说明 |
|:------:|------|------|------|
| 0x00 | `MMUL wa, ia, oa, N` | 4 字段 | 权重地址 wa，输入地址 ia，输出地址 oa，大小 N。触发 MXU |
| 0x01 | `SOFTMAX sa, da, len` | 3 字段 | Softmax 激活：源地址，目标地址，向量长度 |
| 0x02 | `LAYERNORM sa, da, len` | 3 字段 | Layer Normalization |
| 0x03 | `GELU sa, da, len` | 3 字段 | GELU 激活 |
| 0x04 | `RELU sa, da, len` | 3 字段 | ReLU 激活 |
| 0x05 | `ROPE sa, da, len, theta` | 4 字段 | RoPE 位置编码 |
| 0x06 | `SILU sa, da, len` | 3 字段 | SiLU 激活（复用 GELU 查表，见 §3.3） |
| 0x07 | `MAXPOOL sa, da, H, W` | 4 字段 | Max Pooling 2×2 |
| 0x08 | `AVGPOOL sa, da, H, W` | 4 字段 | Avg Pooling 2×2 |
| 0x09 | `DMA_LD dram, sram, size` | 3 字段 | LPDDR5→SRAM（直接地址模式） |
| 0x0A | `DMA_ST sram, dram, size` | 3 字段 | SRAM→LPDDR5（直接地址模式） |
| 0x0B | `KV_LOAD token_id` | 1 字段 | 将指定 token 的 KV 加载到 SRAM cache |
| 0x0C | `KV_STORE token_id` | 1 字段 | 当前计算的 KV 写入 LPDDR5 |
| 0x0D | `BARRIER` | 0 字段 | 流水线同步栅栏 |
| 0x0E | `NOP` | 0 字段 | 空操作 |
| 0x0F | `VADD sa, da, len` | 3 字段 | 逐元素 INT32 加法（向量单元 §3.8） |
| 0x10 | `VMUL sa, da, len` | 3 字段 | 逐元素 INT32 乘法（向量单元 §3.8） |
| 0x11 | `VRED_MAX sa, da, len` | 3 字段 | 树规约求最大值，log₂(N) 级（向量单元 §3.8） |
| 0x12 | `VRED_SUM sa, da, len` | 3 字段 | 树规约求和，INT32 溢保（向量单元 §3.8） |
| 0x13 | `VCONV sa, da, len` | 3 字段 | INT32→BF16 类型转换，MXU↔SFU 桥梁（向量单元 §3.8） |
| 0x14 | `VRESID sa, da, len` | 3 字段 | 残差连接 da += sa，INT32 饱和（向量单元 §3.8） |
| 0x15 | `DMA_LDD dram, sram, size` | 3 字段 | DMA 加载（描述符链模式） |
| 0x16 | `DMA_STD sram, dram, size` | 3 字段 | DMA 存储（描述符链模式） |

> **OpCode 定义来源**：`sim/engine/isa.py` 第 12-38 行 `OpCode(IntEnum)`，当前共 23 条指令（v2 新增：Vector 7 条 + DMA 描述符 2 条）。指令编码格式见同文件 `NPUEncoder`（第 82-156 行）。

---

### 3.8 Vector Unit（向量处理单元）

向量单元是 MXU（INT32 输出）与 SFU（BF16 输入）之间的桥梁，同时承担逐元素运算和规约操作。它是一个 128-wide SIMD 流水线，与 MXU/SFU 共享 SRAM 带宽。

```
    MXU 输出 (INT32)                SFU 输入 (BF16)
         │                                ▲
         │    ┌───────────────────┐       │
         └───►│    Vector Unit     │───────┘
              │   128-wide SIMD    │
              │                   │
              │  · VADD / VMUL    │
              │  · VRED_MAX/SUM   │
              │  · VCONV          │
              │  · VRESID         │
              └───────────────────┘
```

#### 设计参考
- **TPUv2 Vector Unit**：可编程向量单元，替代 TPUv1 固定函数数据通路
- **本设计**：固定功能向量单元——7 条专用指令，无程序计数器，无分支，面积更小

#### 指令清单

| 指令 | 操作 | 硬件结构 | 延迟 |
|------|------|---------|:---:|
| **VADD** | 逐元素 INT32 加法 | 128 个并行 INT32 加法器 | 1 cycle |
| **VMUL** | 逐元素 INT32 乘法 | 128 个并行 INT32 乘法器 | 1 cycle |
| **VRED_MAX** | 树规约求最大值 | log₂(128)=7 级比较器树 | 7 cycles |
| **VRED_SUM** | 树规约求和（带溢保） | log₂(128)=7 级加法器树 + INT32 饱和逻辑 | 7 cycles |
| **VCONV** | INT32 → BF16 类型转换 | 128 个并行的 INT32→FP32→FP16 截断 + 饱和电路 | 1 cycle |
| **VRESID** | 残差连接 da += sa | 128 个 INT32 加法器 + INT32 饱和逻辑 | 1 cycle |
| **—** | Softmax 分解辅助 | 复用 VRED_MAX + VADD + VRED_SUM | — |

> **VRED_MAX + VRED_SUM 为什么是 7 cycles？** 128 元素的二叉树规约：第 1 级 64 对并行运算 → 第 2 级 32 对 → ... → 第 7 级 1 对。每级 1 cycle。

#### 在 Transformer 中的角色

向量单元承担 Transformer Block 中 SFU 前后的数据搬运和类型转换：

1. **FFN 残差连接**（VRESID）：`hidden_states = hidden_states + FFN_output`，在 INT32 域完成，避免精度损失
2. **Attention 残差连接**（VRESID）：同上
3. **INT32→BF16 转换**（VCONV）：MXU 输出 INT32 累加结果在进入 SFU 前转为 BF16
4. **Softmax 分解**（VRED_MAX + VRED_SUM）：SFU 的 Softmax 硬件用查表算 exp(x)，用向量单元做 max 减法和 sum 规约
5. **RMSNorm / LayerNorm 辅助**：均值/方差计算可用 VRED_SUM 加速

#### 面积估算

| 组件 | 面积 | 说明 |
|------|:---:|------|
| 128×INT32 加法器（VADD/VRESID 复用） | ~0.15mm² | |
| 128×INT32 乘法器（VMUL） | ~0.2mm² | |
| 规约树（VRED_MAX/SUM 共享比较器/加法器） | ~0.1mm² | |
| INT32→BF16 转换器（VCONV） | ~0.05mm² | |
| **合计** | **~0.5mm²** | |

#### 交叉引用

- **OpCode 定义**：`sim/engine/isa.py` 第 29-36 行（VADD=0x0F .. VRESID=0x14）
- **指令编码**：`sim/engine/isa.py` 第 137-145 行（Generic Vector: op + sa + da + len）
- **Bit-accurate 参考模型**：`sim/golden_executor.py` 第 608-704 行 `GoldenVector` 类
  - `add()` (L625), `mul()` (L630), `max_reduce()` (L637), `sum_reduce()` (L645)
  - `conv_i32_to_f16()` (L657), `residual_add()` (L677)

---

## 四、数据流与流水线

### 4.1 LLM Decode 阶段流水线

```
Time ──────────────────────────────────────────────────────→

DMA:   [Load W₀] [Load W₁] [Load W₂] ...
MXU:             [MMUL₀ ] [MMUL₁ ] [MMUL₂ ] ...
SFU:                       [SM/LN₀] [GELU₀] [SM/LN₁] ...
DMA:   [KV-Ld ]                                   [KV-St ]
```

权重加载和矩阵运算流水线重叠——DMA 加载 Layer N+1 权重的同时，MXU 正在算 Layer N。

### 4.2 CNN 推理流水线

```
DMA:   [Load W]  [Load IA]
MXU:             [MMUL (im2col→GEMM)]
SFU:                       [ReLU] [Pool]
```

简单流水线。CNN 层间数据量小，DMA 几乎不占时间。

---

## 五、多核扩展架构（片内 IP 实例化）

### 5.1 核心思想

NPU 作为可参数化 IP 核。SoC 客户在同一个 die 上实例化 1-N 个核，所有外设（LPDDR5/PCIe/RISC-V）共享一套。核间通过片上互联（Crossbar）和专用 FIFO 通道通信，带宽 ~500 GB/s，比 PCIe P2P 快 60 倍。

**与 PCIe 多卡方案的对比**：

| | PCIe 多卡 | **片内多核 IP** |
|------|------|------|
| 核间带宽 | 7.88 GB/s (Gen4 x4) | **~500 GB/s** |
| 核间延迟 | ~1μs | **~10ns** |
| 每增一核算力 | +1 张卡（~$37） | **+14.5mm² die** |
| 共享 LPDDR5 | ❌ 各卡独立 | ✅ 共享物理内存池 |
| 扩展上限 | 受 PCIe 槽数限制 | 受 die 面积限制 |
| 商业模式 | 卖卡给终端用户 | **卖 IP 给 SoC 厂商** |

### 5.2 IP 参数化配置

```verilog
module npu_top #(
    parameter int NUM_CORES      = 1,   // 1, 2, 4, 8
    parameter int MXU_SIZE       = 64,  // 32, 64 (Block engine, 非 systolic)
    parameter int L2_SIZE_KB     = 2048,// 共享 L2
    parameter int LPDDR_DATA_W   = 64,  // 64 / 128
    parameter bit ENABLE_FIFO    = 1    // 使能核间 FIFO
) (
    input  wire        clk,
    input  wire        rst_n,
    // AXI4 主接口 → LPDDR5 控制器
    output wire [31:0] m_axi_awaddr,
    ...
    // AXI4 从接口 ← PCIe / 主机
    input  wire [31:0] s_axi_awaddr,
    ...
    // 中断输出
    output wire [NUM_CORES-1:0] irq
);
```

SoC 客户配 `NUM_CORES=4`，综合工具自动生成 4 核。

### 5.3 三种工作模式

| 模式 | 核间通信 | 每核 L1 | 共享 L2 | 适用场景 |
|------|:---:|------|------|------|
| **独立模式** | 无 | 私有 | 每核独立分区 | 数据并行——N 用户各占一核 |
| **共享内存** | L2 SRAM | 私有 | 权重分片共享 | 单模型跨核——权重拆到 L2 |
| **流水线 FIFO** | 核间 FIFO | 私有 | 激活中转 | 层间流水线——激活值直传 |

### 5.4 核间 FIFO 设计

```
核₀ ──[FIFO 4KB]──→ 核₁ ──[FIFO 4KB]──→ 核₂ ──→ ...
 ←──[FIFO 4KB]──   ←──[FIFO 4KB]──
```

- 双向各 4KB → 每核 8KB FIFO 存储
- 4KB = 512 个 BF16 元素，足够装一层中间激活
- 宽度 256-bit，深度 128。2 cycle 延迟
- 仅在流水线模式使能（`ENABLE_FIFO=1`）。不使能时综合工具优化掉

### 5.5 性能缩放（DSE 验证，基于 Block 64×64）

> Block Engine 零 pipeline fill/drain 开销，M=1 decode 即达到 DRAM 带宽上限，无需 continuous batching 恢复效率。单核 M=1 与 M≥2 性能相同——瓶颈在 DRAM，不在引擎。

| 配置 | 面积 | M=1 decode | 数据并行加速比 |
|------|:---:|------|:---:|
| 1 核 | 28.2mm² | **29.6 tok/s** ✅ | 1.0× |
| 2 核 | 45mm² | 56 tok/s | 1.9× |
| 4 核 | 72mm² | 100 tok/s | 3.4× |
| 8 核 | 125mm² | 157 tok/s | 5.3× |

> **数据并行**：每核独立算不同 batch 的 token。核间无通信开销，仅共享 DRAM 带宽有 5-35% 争用。
> **流水线并行**（大模型）：层分布到多核，核间 FIFO 传递激活。7B ~13 tok/s, 13B ~8 tok/s, 30B ~6 tok/s。

### 5.6 性能瓶颈分析（DSE 验证）

**Block 64×64 实测瓶颈分布（DSE simulator）**：

| 瓶颈 | 占比 | 说明 |
|------|:---:|------|
| DRAM 权重读取 | 86% | INT4 3B 每 token 1.25 GB，43.5 GB/s 有效带宽限制 |
| SFU + 向量单元 | 10% | 类型转换 + 规约 + 激活函数 |
| 其他流水线 | 4% | KV cache 加载、控制流 |

**根因：DRAM 带宽是唯一瓶颈**

Block engine 所有 4,096 个 PE 始终 100% 利用（零流水线开销），每个 decode step 的计算延迟远小于权重加载延迟。实际吞吐由 DRAM 读取全部权重的速度决定：

```
每 token 权重读取 = 2.5B params × 0.5 byte/param (INT4) = 1.25 GB
LPDDR5-6400 有效带宽 = 43.5 GB/s (85% of 51.2)
理论 tok/s 上限 = 43.5 / 1.25 ≈ 35 tok/s
实际可达 ≈ 29.6 tok/s（扣除 KV cache + SFU 开销）
```

**优化路径评估**：

| 方案 | tok/s | 面积 | 代价 |
|------|:---:|:---:|------|
| **当前配置（Block 64×64）** | **29.6** | 28.2mm² | — |
| INT2 量化（待验证） | ~58 | 28.2mm² | 需 INT2 精度验证 |
| 128-bit LPDDR5（100 GB/s） | ~58 | +7mm² PHY | 引脚面积不可接受 |
| 多核数据并行 | ~157 (8核) | 139mm² | 面积线性增长 |

> **结论**：LPDDR5-6400 下 Block 64×64 已达 DRAM 物理上限。要更高吞吐，要么换更高带宽 DRAM，要么降精度（INT2）。多核数据并行是唯一面积线性扩展路径。

### 5.7 `NUM_CORES=2` 流水线并行数据流

```
Time ────────────────────────────────────────────────────────────→

核₀: [Layer 0-15 的 MMUL+SFU] ──→ [FIFO 写激活值]
核₁:                              [FIFO 读] [Layer 16-31 的 MMUL+SFU]

DMA:  [Load W(0-15)→核₀ L1] [Load W(16-31)→核₁ L1]
```

- 权重各自加载到本地 L1，不抢占 L2 带宽
- KV Cache 按层分布到各核本地 SRAM——不需要全局 KV 池

### 5.8 软件侧适配

IREE HAL 后端感知多核：

```
iree_hal_npu_query_info() → {
    num_cores: 4,
    engine: "block_64x64",
    core_l1_kb: 512,
    shared_l2_kb: 4096,
    has_fifo: true
}
```

模型下发时由 Host Runtime 决定分区策略。应用层透明——只看到一个更强的 NPU。

---

## 六、开源复用矩阵

| 模块 | 参考源 | 复用方式 | 改动量 |
|------|--------|---------|:---:|
| **Block Array PE** | OpenTPU (Verilog) | 提取 PE 模块 + 改 broadcast 互联 | 小 |
| **阵列拓扑** | OpenTPU / TPUv1 | 配置 64×64 Block 拓扑 | 小 |
| **MAC 精度** | Gemmini (Chisel) | INT8→INT4 多精度设计参考 | 中 |
| **Unified Buffer** | TPUv1 | 双缓冲架构，容量缩小 | 小 |
| **Softmax+LN 融合** | 中科院 2024 论文 | 直接参考硬件方案 | 中 |
| **RISC-V 集成** | Coral NPU | AXI4 总线连接范式 | 小 |
| **AXI4 矩阵** | Coral NPU | 标准互联 | 低 |
| **指令集设计** | TPUv1 CISC ISA | 五类指令范式 | 小 |
| **PCIe EP** | 商用 IP | 采购 | 无 |
| **LPDDR5 控制器** | 商用 IP | 采购 | 无 |
| **DMA 引擎** | 业界参考设计 | 自研 | 中 |

---

## 七、关键学术/工业参考

| 论文/项目 | 来源 | 年份 | 参考价值 |
|------|------|:---:|------|
| **TPUv1 - In-Datacenter Performance Analysis** | Google, ISCA | 2017 | 架构基线 |
| **OpenTPU** | UCSB ArchLab | 2024 | Verilog 可直接读 |
| **Gemmini** | UC Berkeley | 2019+ | Systolic array 参数化设计方法论 |
| **Hardware-oriented Softmax+LayerNorm** | 中科院 | 2024 | SFU 融合架构 |
| **Calabash: Accelerating Attention on FPGA** | IEEE | 2023 | 双 systolic array 做 attention |
| **TPI-LLM** | OpenReview | 2025 | 多边缘设备跑 70B 模型 |
| **SCALE-Sim v3** | arXiv | 2025 | Cycle-accurate 性能模拟器 |
| **Coral NPU** | Google Research | 2025 | RISC-V+AXI 集成范式 |
| **腾讯 KV Cache 专利** | 腾讯 | 2025 | 内存碎片化问题的教训 |
| **HeadInfer** | arXiv | 2025 | KV cache 按 head 卸载策略 |
| **SystolicAttention** | arXiv | 2025 | Systolic array 上融合 FlashAttention |
| **A Survey on Hardware Accelerators for LLMs** | arXiv | 2024 | 全景参考 |

---

## 八、模拟器验证结果（DSE 更新）

已用自研 DSE simulator（`sim/engine/block_engine.py`）完成性能建模：

1. ✅ 配置 64×64 Block engine（broadcast-based）@ 800 MHz
2. ✅ 注入 Qwen2.5-3B GEMM trace（28 层 × 7 matmuls/层，共 196 matmuls）
3. ✅ **Decode (M=1): 29.6 tok/s**，已达 LPDDR5-6400 有效带宽上限
4. ✅ 瓶颈：DRAM 权重读取占 86%，计算非瓶颈
5. ✅ Block engine 零 pipeline fill/drain 开销，M=1 即满利用率
6. ✅ CV：MobileNetV3-Small 677.9 FPS
7. ✅ 多核数据并行：2核 1.9×、4核 3.4×、8核 5.3×

**关键发现**：
- DRAM 带宽是真天花板，不是 NPU 引擎。Block 64×64 已逼近物理上限
- 换更大阵列不提升 decode 性能，但面积翻倍
- 多引擎对比验证：Block 64×64 在面积/功耗/性能三方面均最优

→ 详细 DSE 报告：`CaduceusCore/docs/Edge_NPU_Architecture_Proposal.md`
→ 引擎模型：`sim/engine/block_engine.py`
→ 设计空间探索器：`sim/design_space_explorer.py`

---

## 九、多引擎设计空间探索（v0.4 新增）

### 9.1 为什么做多引擎对比

2017 年 TPUv1 选 Weight-Stationary Systolic Array 是因为当时 LLM 还不存在，任务以 CNN 推理为主——大 batch、高利用率。2026 年 LLM decode 的场景完全不同：M=1 单 token 推理，DRAM 带宽是主要瓶颈，计算引擎的流水线开销影响被放大。

只押 WS-Systolic 而不验证其他数据流，等于闭着眼睛选架构。本章通过自研 Python simulator 对五种矩阵乘法引擎做统一 INT4 精度下的性能-面积-功耗（PPA）对比。

### 9.2 五种引擎简介

| 引擎 | 数据流 | 参考源 | M=1 特征 |
|------|--------|--------|---------|
| **WS-Systolic** | Weight-Stationary | TPUv1 / OpenTPU | Pipeline fill/drain overhead ~385c/tile |
| **OS-Systolic** | Output-Stationary | Gemmini (UC Berkeley) | 零 pipeline overhead，激活值驻留 |
| **Input-Stationary** | Input-Stationary | Eyeriss (MIT) | 权重广播，适合大 batch |
| **Block Engine** | 2D Block Tiling | TPUv4 VMU | 全并行 MAC，无流水线填充 |
| **Tensor Core** | 16×16 小块并行 | NVIDIA A100 | FP16→BF16，大量小块 DMA |

### 9.3 全局 PPA 对比（INT4 统一精度，DRAM 50GB/s 封顶）

```
DRAM 物理天花板: 35 tok/s (50GB/s, INT4 @ 1.25 GB/token)

┌─ 64×64 阵列配置 ─────────────────────────────────────────────
│ Architecture              tok/s   面积    DRAM%   tok/mm²
│ ────────────────────────────────────────────────────────────
│ ✅ Block 64×64             29.6   28.2mm²   85%    1.05   本设计
│ ✅ OS-Systolic 64×64       29.6   28.2mm²   85%    1.05   Gemmini
│   GMMA 64×64               30.0   30.2mm²   86%    1.00   NVIDIA TC 演进
│   Tensor Core 64×16×16     28     52mm²    80%    0.55   A100 TC
│   Input-Stationary 64×64    6.4   44mm²    18%    0.15   Eyeriss
│   Systolic 64×64           11.2   22.2mm²   32%    0.50   TPUv1
│   WMMA 64×64               0.05   57mm²     0%    0.001   NVIDIA WMMA
└────────────────────────────────────────────────────────────
│
┌─ 更大阵列对比（面积翻倍，性能不变）────────────────────
│ Block (翻倍增)              30.0   52mm²    86%    0.58   TPUv4 VMU
│ Systolic (翻倍增) +WC       21     28mm²    60%    0.75   TPUv1
└────────────────────────────────────────────────────────────
│
┌─ 带宽倍增配置（128-bit LPDDR5, 100GB/s）────────────────
│ Block 64×64 +DMA×2         58     53mm²   100%    1.09   ✅
│ GMMA 64×64 +DMA×2          57     53mm²    98%    1.07
└────────────────────────────────────────────────────────────
```

> **约束说明**：所有配置统一 INT4 权重精度、800 MHz 频率（除标注外）、64-bit LPDDR5-6400（50GB/s）。面积含 MXU+SFU+L1 SRAM+KV Cache。DMA×2 表示 128-bit DRAM 接口（100GB/s）。

### 9.4 核心结论

**结论一：DRAM 是真天花板，不是 NPU 引擎**

所有引擎在 50GB/s DRAM 约束下性能受限于 DRAM 带宽。INT4 × 3B × 1.25 GB/token ÷ 50 GB/s → 物理极限 35 tok/s。扣除 KV cache + SFU 开销 → ~29.6 tok/s。换引擎不改变物理上限。

**结论二：Block 64×64 是端侧最优解**

29.6 tok/s（85% DRAM 利用率），面积 28.2mm²。在 ≤30 mm² 面积约束下性能最高。Systolic 64×64（11.2 tok/s, 22.2mm²）虽面积更小，但 M=1 decode 性能不达标；OS-Systolic（29.6 tok/s, 28.2mm²）性能相同但实现复杂度更高。

**结论三：小阵列（64×64）匹配低带宽**

DRAM 墙下，更大阵列相对 64×64 几乎无性能收益（30 vs 29.6 tok/s，+1%），但面积增加 84%（52 vs 28.2 mm²）。端侧应坚持小阵列。

**结论四：Systolic / WMMA / Input-Stationary 在 M=1 decode 下不可用**

Systolic 的 pipeline fill/drain 开销在 M=1 单 token 下无法摊销（仅 11.2 tok/s）。WMMA 16×16 子块导致 DMA 启动开销爆炸（0.05 tok/s）。Input-Stationary 激活复用率极低（6.4 tok/s）。三者均未达到 20 tok/s 目标。

**结论五：不改 DRAM 的最优参数**

```yaml
mac_engine:
  type: block
  array_height: 64
  array_width: 64
  weight_precision_bits: 4
  frequency_mhz: 800
memory:
  dram_width_bits: 64
  bandwidth_gbps: 51.2
```

→ 29.6 tok/s，85% DRAM 利用率，28.2mm²。Zero change to DRAM subsystem。Block engine 零流水线开销，M=1 即达 DRAM 上限。

**结论六：换 DRAM 换天花板**

如果接受 128-bit LPDDR5 → Block 64×64 + DMA×2 → 58 tok/s（单核 53mm²）。但 LPDDR 扩总线位宽 = 多一倍 PHY 引脚 + PCB 走线，物理成本和面积代价高。当前路线留在 50GB/s 天花板内，用 Block 64×64 逼近此极限。

### 9.5 设计空间搜索器

所有结果由自研 `design_space_explorer.py` 生成：

```bash
cd ~/npu/sim
python3 design_space_explorer.py --quick   # 快速扫描
python3 design_space_explorer.py           # 完整 2550 配置
```

引擎模型位于 `engine/` 目录：
- `systolic_engine.py` — WS-Systolic，含 pipeline fill/drain 建模
- `os_systolic_engine.py` — OS-Systolic（Gemmini 风格）
- `is_systolic_engine.py` — Input-Stationary（Eyeriss 风格）
- `block_engine.py` — 2D Block Tiling 全并行
- `tensor_core_engine.py` — NVIDIA TC 风格 16×16 小块

---

> **文档版本**：v0.6 | **v0.6 变更（DSE 验证）**：(1) MXU 从旧版 Systolic Array 改为 64×64 Block Array (broadcast-based)，零流水线开销；(2) 总面积修正为 ~28.2 mm²（单核），decode 29.6 tok/s（DRAM-bound）；(3) §2.3 模块清单 MXU 行和面积预算表更新；(4) §3.1 MXU 章节重写为 Block broadcast 语义；(5) §5.5-5.6 多核性能缩放和瓶颈分析更新，移除 tiling overhead 相关描述；(6) §8 模拟器结果更新为 DSE 验证数据；(7) §9 PPA 对比更新，Block 64×64 为推荐架构；(8) 移除所有陈旧 systolic tiling 声明 | **v0.5 变更**：(1) 修正 Unified Buffer 容量为 4 MB（与 `golden_executor.py` SRAM_SIZE=4MB 和 `func_model_architecture.md` 统一）；(2) 新增 §3.2 L1 SRAM (512 KB) 与 Unified Buffer (4MB) 区分说明；(3) ISA 表从 14 条扩展至 23 条（新增 SILU + Vector Unit 7 条 + DMA_LDD/DMA_STD），含 OpCode hex 值；(4) 新增 §3.8「Vector Unit」章节，描述 128-wide SIMD 流水线及 7 条向量指令；交叉引用 `sim/engine/isa.py` OpCode 枚举与 `sim/golden_executor.py` GoldenVector 参考模型 | **v0.4 变更**：新增第九章「多引擎设计空间探索」，含五引擎 INT4 统一精度全局 PPA 对比、六条核心结论、推荐参数配置、Simulator 入口说明 | **v0.3 变更**：性能数字从理论估算修正为 v2 tiling-aware simulator 实测；新增 5.6 瓶颈分析章节；第八章替换为实际模拟结果；新增 continuous batching 优化路径 | **v0.2 变更**：NPU 核改为可参数化 IP；多核扩展从 PCIe P2P 改为片内实例化 | **下一步**：软件架构方案更新 + batch scheduler 设计
