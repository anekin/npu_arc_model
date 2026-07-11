# NPU 矩阵乘法引擎架构全景

> 7 种引擎 × 6 级 DRAM，一图看懂架构差异、适用场景与选型逻辑。
> 所有性能数据来自自研 Python simulator，3B LLM decode (M=1)，INT4 精度。
> 本文基于修正后的 Arc Model DSE v2；GMMA TMA 模型和 Systolic Prefill 模型均已修正，确保不违反 DRAM 带宽上限。

---

## 一、总览

```
                     DMA 碎片 ←──────────────────────────→ 面积

  WMMA (0.05)     TensorCore (27.7)    GMMA (30.0)     Block (29.6) ✅
  16×16           64×16×16             64×64+TMA        64×64 广播
       │                │                   │                │
       ▼                ▼                   ▼                ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     面积-性能 Pareto 前沿                        │
  │                                                                  │
  │  22mm² ── Systolic 64×64               (11.2 tok/s)             │
  │  44mm² ── Input-Stationary 64×64       (6.4 tok/s)              │
  │  28mm² ── Block 64×64+WC               (29.6 tok/s) ← ✅ 推荐  │
  │  28mm² ── OS-Systolic 64×64            (29.6 tok/s)             │
  │  30mm² ── GMMA 64×64+TMA               (30.0 tok/s)             │
  │  52mm² ── Tensor Core 64×16×16         (27.7 tok/s)             │
  │  57mm² ── WMMA 16×16                   (0.05 tok/s)  ← ☠️      │
  │                                                                  │
  │  * Block/OS-Systolic/GMMA 均为 DRAM 瓶颈值（~30 tok/s 上限）     │
  │  ═══════ DRAM 51.2 GB/s × 85%效率 = 43.5 GB/s 天花板 ═══════     │
  └──────────────────────────────────────────────────────────────────┘

**核心洞察：** LPDDR5-6400 实际带宽上限约 43.5 GB/s，Block/OS-Systolic/GMMA 均处于 DRAM 瓶颈区（~30 tok/s），三者性能等价。Block 64×64 用最低面积（28.2 mm²）实现该上限性能，且通过全并行广播架构避免了 Systolic 的 pipeline 开销和 Tensor Core 的碎片问题。GMMA 的 TMA 无法突破 DRAM 带宽上限——它只能隐藏 DMA latency，不能减少需要读取的总字节数。引擎选择本质上是 **广播效率 vs 面积** 的 trade-off——Block 用 crossbar 面积换取了零流水线开销和最低的 DMA 碎片。

---

## 二、七种引擎逐个拆解

### 2.1 WS-Systolic Array（TPUv1 风格）

```
           ← weights preloaded on diagonal pipeline

  ┌─────────────────────────────┐
  │ PE₀→PE₁→PE₂→ ... →PE₆₃│  ← data flows left→right
  │  ↓    ↓    ↓         ↓  │
  │ PE₆₄→ ...         →PE₁₂₇│
  │  ↓    ↓             ↓   │
  │ ...        64×64       ...│
  │  ↓    ↓             ↓   │
  │ PE₄₀₃₂→ ...       →PE₄₀₉₅│
  └─────────────────────────────┘
       ↓ partial sums accumulate downward

   每周期: 1 个数据流入，1 个 MAC 运算
   关键瓶颈: M=1 时每个 tile 要等 193 cycles pipeline fill/drain
```

| 特性 | 值 |
|------|-----|
| 数据流 | 权重静态（预加载在 PE 内），激活数据从左向右流 |
| 面积/PE | 极小 — 一个 MAC + 一个寄存器 |
| Pipeline overhead | **193 cycles/tile**（127 fill + 65 drain + 1 compute）|
| 适合场景 | CNN 推理（大 batch）、prefill（M=128）— pipeline overhead 被摊销 |
| 不适合场景 | LLM decode (M=1) — overhead 占 ~99.9% 时间 |
| DRAM 需求 | 低 — 权重加载可双缓冲，激活量极小 |
| 关键优化 | 加宽阵列（64→128）、Weight Cache（PE 双寄存器存 gate+up）|

**一句话：** 面积效率最高（22.2 mm² @ 64×64），但在 M=1 decode 场景被 pipeline 物理开销严重拖累，仅 11.2 tok/s。

---

### 2.2 OS-Systolic（Gemmini 风格）

```
              ← weights flow right
  ┌─────────────────────────────┐
  │ Psum₀ ← Psum₁ ← ...      │  ← partial sums stationary
  │   ↑       ↑                │
  │  PE₀     PE₁      ...      │
  │   ↑       ↑                │
  │ Activation broadcast ↓     │  ← activations flow down
  └─────────────────────────────┘

  输出驻留在 PE 内 → 零 pipeline fill/drain
  代价: 每个 PE 需要 accumulator 存储完整输出行
```

| 特性 | 值 |
|------|-----|
| 数据流 | 输出静态（部分和驻留 PE），权重和激活流入 |
| Pipeline overhead | **0 cycles** — 输出已经在 PE 里，不需要排空 |
| 面积/PE | 大 — 需要 accumulator + 双缓冲（~4× systolic）|
| 适合场景 | LLM decode — 零流水线开销，每 tile 1 cycle |
| 不适合场景 | 面积敏感场景 — 28.2 mm²，与 Block 相同但实现更复杂 |
| DRAM 需求 | 同 systolic — 输出驻留不占 DRAM 带宽 |
| 参考 | UC Berkeley Gemmini (Chisel generator) |

**一句话：** 零 pipeline 开销，性能（29.6 tok/s）与 Block 相同，但面积代价高于 Systolic。实现复杂度高于 Block，故非首选。

---

### 2.3 Block 64×64 Engine（TPUv4 VMU 风格）← ✅ **推荐**

```
  ┌───────────────────────────┐
  │ [MAC] [MAC] ... [MAC] ← 64│  所有 MAC 同时点火
  │ [MAC] [MAC] ... [MAC]     │  → 1 cycle 算完一个 tile
  │  ...   ...   ...  ...     │
  │ [MAC] [MAC] ... [MAC]     │
  └───────────────────────────┘
       ↑                  ↑
  ┌───┴──────────────────┴────┐
  │    Crossbar 广播总线       │  ← 代价：全互连
  │  权重广播到所有 PE         │
  └────────────────────────────┘
```

| 特性 | 值 |
|------|-----|
| 数据流 | 纯空间并行 — 权重+激活广播到所有 MAC |
| 每 tile 时间 | **1 cycle compute + DMA time** |
| 面积 | **28.2 mm² @ 64×64**（含 Weight Cache、im2col、SFU）|
| 瓶颈 | **DRAM** — 算得再快，数据从 DRAM 搬不过来（~30 tok/s 上限）|
| 适合场景 | LPDDR5-6400 端侧 LLM decode — 29.6 tok/s，DRAM 完全占满 |
| CV 性能 | MobileNetV3-Small **677.9 FPS**（64×64 INT4）|
| 功耗 | **~9.6 W** |

**一句话：** ✅ **推荐引擎。** Block 64×64 在 28.2 mm² 实现 DRAM 瓶颈下的最高性能（29.6 tok/s），全并行广播架构消除了 pipeline 开销和 DMA 碎片问题，且 LLM/CV 双栈验证通过。广播效率是核心优势，DRAM 带宽是唯一瓶颈。带宽翻倍性能即可翻倍。

---

### 2.4 Tensor Core（A100 风格）

```
   ┌──────────────────────────────────┐
   │ TC₀   TC₁   TC₂   ...   TC₆₃    │  64 个独立 TC
   │16×16 16×16 16×16       16×16   │
   │ [MAC] [MAC] [MAC]      [MAC]    │
   │...×64 ...×64 ...×64    ...×64   │  ← 64 个子 tile
   └──────────────────────────────────┘
          ↑      ↑      ↑          ↑
        各自独立 DMA（碎片化问题）

  每个 TC 算 16×16×16 小块 → 大量 invocation
  64× 并行掩盖了一部分，但碎片仍比 Block 多
```

| 特性 | 值 |
|------|-----|
| 数据流 | 64 个独立 16×16 TC 并行，各自 DMA |
| 碎片度 | **高** — 比 Block 多 64× 的 DMA 事务 |
| 面积 | 52mm²（~32mm² PE + 30% orchestration）|
| 性能 @ 51.2 GB/s | 27.7 tok/s（略低于 Block 的 29.6）|
| 适合场景 | 需要小块灵活性时（非规则矩阵、稀疏）|
| NVIDIA 差异 | GPU 有 warp scheduler 隐藏 DMA 延迟，单 die NPU 没有 |

**一句话：** Block Engine 的小块版本。灵活性换来了碎片开销，单 die NPU 下不如直接上 Block。在 LPDDR5-6400 下因碎片问题性能反而低于 Block。

---

### 2.5 WMMA — Warp MMA（Volta/Ampere 风格）☠️

```
   ┌─────────────────────────────────────────┐
   │  warp₀  warp₁  warp₂  ...  warp₆₃       │
   │  16×16  16×16  16×16      16×16        │
   │   ↓      ↓      ↓          ↓           │
   │  ┌──┐  ┌──┐  ┌──┐      ┌──┐           │
   │  │RF│  │RF│  │RF│      │RF│ ← 寄存器文件│
   │  └──┘  └──┘  └──┘      └──┘           │
   └─────────────────────────────────────────┘
     每个 warp 从寄存器文件读数据（超低延迟）
     但 M=1 时绝大部分寄存器空间闲置

     16×16 tile × 10c DMA startup × 100K+ invocations
     = 百万级 cycles 纯等待
```

| 特性 | 值 |
|------|-----|
| 性能 @ 50GB/s | **0.05 tok/s** — 比 Block 慢 600× |
| 根因 | DMA 启动开销爆炸（每次启动 10 cycles × 10 万次 = 100 万 cycles 纯等）|
| GPU 怎么解决的 | **数千个 warp 同时跑** — 一个 warp 等 DMA 时，scheduler 切到另一个 warp |
| 单 die NPU 为何不行 | 只有 1 个指令流 — 等 DMA 时 CPU 完全 idle |

**一句话：WMMA 是 GPU 专属架构。单 die NPU 上不能用——这是本报告最重要的发现之一。**

---

### 2.6 GMMA — Group MMA + TMA（Hopper H100 风格）

```
   ┌─────────────────────────────────────────────┐
   │                                             │
   │  ┌───────────┐   ┌──────────────────┐      │
   │  │ TMA 单元   │──→│ Shared Memory    │      │
   │  │(异步 DMA) │   │ (2MB)            │      │
   │  └───────────┘   └─────────┬────────┘      │
   │        ↑                   ↓               │
   │   DRAM ←→ TMA 搬数    ┌──────────┐         │
   │   (不阻塞计算)         │ 64×64   │         │
   │                        │ MAC Array│         │
   │                        └──────────┘         │
   └─────────────────────────────────────────────┘

   TMA 作用: 算 tile N 的同时，TMA 在后台加载 tile N+1
   ⚠️ 关键: 这只能隐藏 DMA latency，不能减少 DRAM 总读取量
   因此 total_time = max(compute, dma) — 仍受物理 DRAM 带宽上限约束
```

| 特性 | 值 |
|------|-----|
| 数据流 | 同 Block + TMA 异步 DMA 引擎 |
| 异步重叠 | DMA 和 compute **可重叠** — 但 DRAM 读取总量不变 |
| 面积 | 30.2 mm²（Block 28.2 + TMA 1 + SharedMem 1）|
| 性能 @ 51.2 GB/s | **30.0 tok/s** — 与 Block 等价，因 DRAM 是瓶颈 |
| 性能 @ 100 GB/s | ~58 tok/s（与 Block 等价，均在 DRAM 上限）|
| 性能 @ 200 GB/s | ~117 tok/s（与 Block 等价，均在 DRAM 上限）|
| TMA 真正价值 | **在 compute-bound 场景（如 HBM）下隐藏 exposed DMA latency** |

**一句话：TMA 只能隐藏 latency，不能突破 DRAM 带宽上限。在 LPDDR5-6400 下 GMMA 与 Block 性能相同（~30 tok/s），但面积和功耗更大（30.2 mm² vs 28.2 mm²），因此不推荐。**

---

### 2.7 Input-Stationary（Eyeriss 风格）

```
              ← weights broadcast (流动)
  ┌─────────────────────────────┐
  │ PE₀₀ ← PE₀₁ ← ...        │  ← activations stationary
  │  ↑       ↑                  │    (驻留在 PE 内)
  │ PE₁₀ ← PE₁₁ ← ...        │
  │  ↑       ↑                  │
  └─────────────────────────────┘

  激活值驻留，权重流入 → 适合大 batch（激活复用）
  M=1 时: 只有一个激活值 → 阵列严重欠利用
```

| 特性 | 值 |
|------|-----|
| 数据流 | 输入静态（激活值驻留），权重广播 |
| 适合场景 | 大 batch prefill、CNN — 激活值可复用 |
| M=1 decode | 6.4 tok/s — 阵列利用率极低 |
| 面积 | 44mm² |
| 参考 | MIT Eyeriss (2016) |

**一句话：** 为 CNN 和 prefill 设计的引擎，decode 场景不适配。在 LPDDR5-6400 下仅 6.4 tok/s，远低于 20 tok/s 目标。

---

## 三、场景速查表

| 场景 | 推荐引擎 | 阵列 | DRAM | tok/s | 面积 |
|------|---------|------|------|:---:|:---:|
| **✅ 推荐配置** | **Block 64×64 + WC** | **64×64** | **LPDDR5-6400 64b** | **29.6** | **28.2 mm²** |
| 备选 — 同性能 | OS-Systolic | 64×64 | LPDDR5-6400 64b | 29.6 | 28.2 mm² |
| 备选 — TMA 测试 | GMMA | 64×64 | LPDDR5-6400 64b | 30.0 | 30.2 mm² |
| 面积最小（性能不足） | Systolic | 64×64 | LPDDR5-6400 64b | 11.2 | 22.2 mm² |
| 中等带宽（～100 GB/s） | Block | 64×64 | LPDDR5X-8533 64b | ~58 | 28.2 mm² |
| 高端（～200 GB/s） | Block | 64×64 | LPDDR5T-9600 64b | ~75 | 28.2 mm² |
| **绝对不要用** | WMMA | 16×16 | 任意 | 0.05 | 57mm² |

---

## 四、选型决策树

```
                    ┌─ DRAM < 100 GB/s? ──┐
                    │ YES                 │ NO
                    ▼                     ▼
            ┌──────────────┐      ┌──────────────┐
            │ 目标 ≥20     │      │ INT2 可用?    │
            │ tok/s?       │      │ +精度已验证?   │
            └──┬────────┬──┘      └──┬────────┬──┘
           YES │        │ NO     YES │        │ NO
               ▼        ▼           ▼        ▼
           Block     Systolic    Block      Block
           64×64+WC  64×64       64×64      64×64
           29.6 tok/s 11.2 tok/s INT2       INT4
           28.2mm²    22.2mm²    ~58 tok/s  ~75 tok/s*
                                 28.2mm²    28.2mm²
```

---

## 五、为什么不选 WMMA — 一图看懂

```
  GPU 上的 WMMA:
  ┌────┬────┬────┬────┐
  │warp│warp│warp│warp│  ← 32 warps/SM × 132 SM = 4224 warps 同时跑
  │ A  │ B  │ C  │ D  │
  │ ═══│DMA │═══ │DMA │  ← A 算的时候 B 等 DMA，C 算的时候 D 等 DMA
  │comp│wait│comp│wait│     scheduler 无缝切换 → 用户看不到等待
  └────┴────┴────┴────┘

  NPU 上的 WMMA:
  ┌────┐
  │唯一│  DMA wait → DMA wait → DMA wait → compute → DMA wait → ...
  │指令│  10c        10c        10c        1c         10c
  │流  │
  └────┘
  10 cycles 等 × 10万次 = 100万 cycles = 1ms 纯浪费
```

---

## 六、关键术语

| 术语 | 全称 | 含义 |
|------|------|------|
| Systolic | 脉动阵列 | 数据像脉搏一样逐周期流过 PE 阵列 |
| WS/OS/IS | Weight/Output/Input Stationary | 哪类数据驻留在 PE 内不动 |
| WMMA | Warp Matrix Multiply Accumulate | 32 线程协作算 16×16 小块 |
| GMMA | Group Matrix Multiply Accumulate | 128+ 线程协作算 128×128 大块 |
| TMA | Tensor Memory Accelerator | H100 的异步 DMA 引擎 |
| BMMA | Block Matrix Multiply Accelerator | 全并行 MAC 阵列（本文的 Block Engine）|

---

> 本文基于自研 Python NPU simulator（dsa_opt.py / npu_sim.py），INT4 + LPDDR5-6400 数据由修正后的 DSE v2 框架生成。
> 数据源：`docs/Edge_NPU_Architecture_Proposal.md` 为架构推理精度规格和 DSE 结果的最新来源。
> GMMA TMA 模型修正说明：`gmma_engine.py` 的 steady-state bottleneck 已修复，不再低于物理 `per_tile_dma`，确保结果不违反 DRAM 带宽上限。
> 代码仓库：`github.com/anekin/CaduceusCore`
> 运行：`cd ~/npu/sim && python3 design_space_explorer.py --top 30`
