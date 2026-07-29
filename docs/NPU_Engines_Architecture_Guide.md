# NPU 矩阵乘法引擎架构全景

> 7 种引擎 × 6 级 DRAM，一图看懂架构差异、适用场景与选型逻辑。
> 所有性能数据来自自研 Python simulator，3B LLM FFN_down GEMM (M=1, K=11008, N=2048)，INT4 精度。
> 性能数据 commit: `02683a9f49bc2df299d31f4af8c1446d99101fce`。
> 本文基于修正后的 Arc Model DSE v2；引擎模型修复后所有 63 个回归测试通过。
>
> ⚠️ **信任声明**: 本指南中的数值与"推荐"结论均为当前模型校准下的估算。部分关键参数（`gmma_pipeline_scale`、`tensor_core_descriptor_overhead`、`block_sparsity_penalty` 等）仍为 T0/T1，尚未达到 `decision-grade` 发布标准。详见 [`docs/model-trust-and-release.md`](docs/model-trust-and-release.md)。

---

## 一、总览

```
                     DMA 碎片 ←──────────────────────────→ 面积

  WMMA (6.9)     TensorCore (2,490)    GMMA (2,540)     Block (2,540) ✅
  16×16           64×16×16             64×64+TMA        64×64 广播
       │                │                   │                │
       ▼                ▼                   ▼                ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     单 GEMM 性能 (FFN_down, M=1)                    │
  │                                                                     │
  │  22mm² ── Systolic 64×64               (946 tok/s)                 │
  │  44mm² ── Input-Stationary 64×64       (6.4 tok/s)                 │
  │  28mm² ── Block 64×64+WC               (2,540 tok/s) ← ✅ 推荐     │
  │  28mm² ── OS-Systolic 64×64            (2,540 tok/s)               │
  │  30mm² ── GMMA 64×64+TMA               (2,540 tok/s)               │
  │  52mm² ── Tensor Core 64×16×16         (2,490 tok/s)               │
  │  ~30mm² ── FSA 64×64                   (1,408 tok/s)               │
  │  57mm² ── WMMA 16×16                   (6.9 tok/s)  ← ☠️          │
  │                                                                     │
  │  * DRAM-bound 三引擎（Block/OS-Systolic/GMMA）性能一致              │
  │  ═══════ DRAM 51.2 GB/s × 85%效率 = 43.5 GB/s 天花板 ═══════       │
  └─────────────────────────────────────────────────────────────────────┘

**核心洞察：** LPDDR5-6400 实际带宽上限约 43.5 GB/s，Block/OS-Systolic/GMMA 均处于 DRAM 瓶颈区（~2,540 tok/s），三者性能等价。Block 64×64 用最低面积（28.2 mm²）实现该上限性能，且通过全并行广播架构避免了 Systolic 的 pipeline 开销和 Tensor Core 的碎片问题。GMMA 的 TMA 无法突破 DRAM 带宽上限——它只能隐藏 DMA latency，不能减少需要读取的总字节数。引擎选择本质上是 **广播效率 vs 面积** 的 trade-off——Block 用 crossbar 面积换取了零流水线开销和最低的 DMA 碎片。Systolic 的 pipeline fill/drain 开销使其降至 946 tok/s，FSA 因 inline Softmax 开销成为 compute-bound。TensorCore 修复了 descriptor 碎片化开销后正确慢于 Block（2,490 vs 2,540 tok/s）。

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
    关键瓶颈: M=1 时每个 tile 要等 192 cycles pipeline fill/drain（修复后公式：per_tile_compute = H×(M+1)+W）
```

| 特性 | 值 |
|------|-----|
| 数据流 | 权重静态（预加载在 PE 内），激活数据从左向右流 |
| 面积/PE | 极小 — 一个 MAC + 一个寄存器 |
| Pipeline overhead | **192 cycles/tile**（H×(M+1)+W = 64×2+64，修复后公式与 MXUModel byte-for-byte 一致）|
| 适合场景 | CNN 推理（大 batch）、prefill（M=128）— pipeline overhead 被摊销 |
| 不适合场景 | LLM decode (M=1) — overhead 占 ~99.9% 时间 |
| DRAM 需求 | 低 — 权重加载可双缓冲，激活量极小 |
| 关键优化 | 加宽阵列（64→128）、Weight Cache（PE 双寄存器存 gate+up）|

**一句话：** 面积效率最高（22.2 mm² @ 64×64），但在 M=1 decode 场景被 pipeline 物理开销严重拖累，单 GEMM 仅 946 tok/s（28 层完整模型约 10.15 tok/s）。

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
| Pipeline overhead | **H cycles K-reduction**（OS 需要 H 次 K 维度累加，修复后公式：self.H + BROADCAST_SYNC + accumulate）|
| 面积/PE | 大 — 需要 accumulator + 双缓冲（~4× systolic，约 2× Block PE area）|
| 适合场景 | LLM decode — K-reduction 在 DMA bound 下不影响瓶颈 |
| 不适合场景 | 面积敏感场景 — PE area ~2× Block PE，相同面积时有效阵列更小 |
| DRAM 需求 | 同 systolic — 输出驻留不占 DRAM 带宽 |
| 参考 | UC Berkeley Gemmini (Chisel generator) |

**一句话：** 修复 K-reduction 深度后（per_tile_compute 加入 self.H=64），OS-Systolic 正确进入 DMA-bound（2,540 tok/s），与 Block 性能一致但 PE area 约 2×。实现复杂度高于 Block，故非首选。

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
| 瓶颈 | **DRAM** — 算得再快，数据从 DRAM 搬不过来（~2,540 tok/s 单 GEMM 上限）|
| 适合场景 | LPDDR5-6400 端侧 LLM decode — 2,540 tok/s（单 GEMM），DRAM 完全占满 |
| CV 性能 | MobileNetV3-Small **677.9 FPS**（64×64 INT4）|
| 功耗 | **~9.6 W** |

**一句话：** ✅ **当前模型下的首选引擎（探索性）。** Block 64×64 在 28.2 mm² 实现 DRAM 瓶颈下的最高性能（2,540 tok/s 单 GEMM / 21.586 tok/s npu_sim 完整模型），全并行广播架构消除了 pipeline 开销和 DMA 碎片问题，且 LLM/CV 双栈验证通过。广播效率是核心优势，DRAM 带宽是当前估算的唯一瓶颈。带宽翻倍性能即可翻倍——但该结论依赖当前 T1 带宽/面积假设，未达 `decision-grade`。

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
| 性能 @ 51.2 GB/s | 2,490 tok/s（修复 descriptor 开销后正确慢于 Block 的 2,540）|
| 适合场景 | 需要小块灵活性时（非规则矩阵、稀疏）|
| NVIDIA 差异 | GPU 有 warp scheduler 隐藏 DMA 延迟，单 die NPU 没有 |

**一句话：** Block Engine 的小块版本。灵活性换来了碎片开销，单 die NPU 下不如直接上 Block。在 LPDDR5-6400 下因每个 sub-tile 的 descriptor 开销（5 cycles/TC/wave），TensorCore 的 DMA cycles 增加 37.8%，正确慢于 Block（2,490 vs 2,540 tok/s）。

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
| 性能 @ 50GB/s | **6.9 tok/s** — 比 Block 慢 370× |
| 根因 | DMA 启动开销爆炸（每次启动 10 cycles × 10 万次 = 100 万 cycles 纯等）|
| GPU 怎么解决的 | **数千个 warp 同时跑** — 一个 warp 等 DMA 时，scheduler 切到另一个 warp |
| 单 die NPU 为何不行 | 只有 1 个指令流 — 等 DMA 时 CPU 完全 idle |

**一句话：WMMA 是 GPU 专属架构。单 die NPU 上不能用（6.9 tok/s，比 Block 低 370×）——这是本报告最重要的发现之一。**

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
| pipeline_scale | **0.05**（未校准 — H100 GMMA 架构假设，可在 YAML 配置中调整）|
| per_tile_compute (M=1) | **7 cycles**（max(1, ceil((H+M+W) × 0.05))，修复后启用缩放）|
| 性能 @ 51.2 GB/s | **2,540 tok/s** — 与 Block 等价，因 DRAM 是瓶颈 |
| 性能 @ 100 GB/s | ~4,950 tok/s（与 Block 等价，均在 DRAM 上限）|
| 性能 @ 460 GB/s (HBM2e) | ~22,824 tok/s（带宽单调性已恢复）|
| total_cycles 下限 | **raw-DMA floor** — TMA overlap 不降低物理字节传输时间 |
| TMA 真正价值 | **在 compute-bound 场景（如 HBM）下隐藏 exposed DMA latency** |

**一句话：TMA 只能隐藏 latency，不能突破 DRAM 带宽上限。GMMA 修复了 pipeline_scale 缩放（per_tile_compute=7）和 raw-DMA floor 后，在 LPDDR5-6400 下与 Block 性能相同（2,540 tok/s），但面积和功耗更大（30.2 mm² vs 28.2 mm²），因此不推荐。带宽单调性已验证：LPDDR5→HBM2e 吞吐提升 9.0×。**

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
| M=1 decode | ~6.4 tok/s — 阵列利用率极低 |
| 面积 | 44mm² |
| 参考 | MIT Eyeriss (2016) |

**一句话：** 为 CNN 和 prefill 设计的引擎，decode 场景不适配。在 LPDDR5-6400 下单引擎仅 ~6.4 tok/s。

---

## 三、场景速查表

| 场景 | 推荐引擎 | 阵列 | DRAM | tok/s | 面积 |
|------|---------|------|------|:---:|:---:|
| **✅ 推荐配置** | **Block 64×64 + WC** | **64×64** | **LPDDR5-6400 64b** | **2,540** | **28.2 mm²** |
| 备选 — 同性能 | OS-Systolic | 64×64 | LPDDR5-6400 64b | 2,540 | 28.2 mm² |
| 备选 — TMA 测试 | GMMA | 64×64 | LPDDR5-6400 64b | 2,540 | 30.2 mm² |
| 面积最小（性能不足） | Systolic | 64×64 | LPDDR5-6400 64b | 946 | 22.2 mm² |
| TensorCore（含 descriptor 开销） | TensorCore | 64×64 | LPDDR5-6400 64b | 2,490 | 52 mm² |
| **绝对不要用** | WMMA | 16×16 | 任意 | 6.9 | 57mm² |

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
            2,540 tok/s 946 tok/s  INT2       INT4
            28.2mm²    22.2mm²     ~4,950     ~2,540*
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
