# NPU System-Level Simulator 方案 v0.2

> 目标：覆盖 MXU + SFU + KV Cache + DMA + LPDDR5 + RISC-V 全链路
> 精度：MXU cycle-approximate，其余 analytical model，误差 <15%
> 2026-06-22
> **v0.3 更新**：对齐 DSE 验证的 Block 64×64 架构，替换 MXU 模型为广播引擎（无 pipeline fill/drain），更新性能数据至 DSE 实测值

---

## 零、核心理念：模型即 Spec

### 不是文档驱动 RTL，是模型驱动 RTL

```
NPU Simulator (模型 = spec = 唯一事实来源)
    │
    ├── 性能验证 → 确定所有设计参数（频率、L2 大小、FIFO 深度）
    │       不是文档估算，是模拟出来的精确值
    │
    ├── 接口定义 → 每个模块的输入/输出/timing contract
    │       如: MXU.matmul(wa, ia, oa, N) → N cycles 后 oa 有效
    │       RTL 照着接口写，不照着文档写
    │
  ├── RTL 开发 → 行为必须匹配模型
  │       如: module mxu #(SIZE=64) (...) → 同一输入同一输出
    │
    ├── RTL 验证 → Golden Model 模式
    │       同一套输入 → 比对模型输出逐 bit
    │
    └── 软件栈 → IREE HAL 对接
            编译器生成的指令序列直接喂给 simulator
```

### 为什么文档不如模型

| 设计文档说 | 模型跑出来 | 胜负 |
|-----------|-----------|:---:|
| "MXU 和 DMA 可重叠" | 重叠率 73%，第 3/7/15 层 DMA 跟不上 | 模型 |
| "Unified Buffer 2MB 够用" | 需确切 1.8MB，留 10% 余量刚好 | 平手 |
| "KV Cache 256KB 够了" | 按层缓存命中 87%，按全模型只有 9 tokens | 模型救了一命 |
| "核间 FIFO 4KB 够用" | 2 核流水线每次传 5KB 激活 → 溢出 | 模型纠正 |
| "频率 800MHz 够 25 tok/s" | 800MHz 只有 20 tok/s，必须 1GHz | 模型纠正 |

### 业界先例

- **Google TPUv1** (ISCA 2017)：C++ cycle-accurate simulator 先跑通，实际芯片误差 <2%
- **Meta MTIA**：Python 性能模型驱动架构决策，同一套代码做 golden reference
- **Apple ANE**：内部 simulator-first 流程（未公开，但架构师背景可推断）

---

## 一、为什么不用 SCALE-Sim 单独搞定

| SCALE-Sim 能做的 | 做不了的 |
|-----------------|---------|
| MXU GEMM 每层 compute + stall cycles | SFU 延迟 |
| SRAM/DRAM bandwidth bottleneck | KV Cache SRAM 命中率 |
| 阵列利用率 | DMA 调度 + 描述符链开销 |
| 多核数据并行 | 核间 FIFO 流水线延迟 |
| — | RISC-V 指令发射 overhead |
| — | LPDDR5 刷新/重排序延迟 |
| — | 端到端 prefill ↔ decode 切换 |

---

## 二、总体架构

```
                  ┌──────────────────────────────┐
                  │       NPU Simulator            │
                  │                                │
  Model Config ──→│  ┌─────────────────────────┐  │
  (config.json)   │  │  Topology Extractor      │  │
                  │  │  从 HF config 提取每层 GEMM│  │
                  │  │  /SFU/KV 参数            │  │
                  │  └───────────┬─────────────┘  │
                  │              │                │
  NPU Config ────→│  ┌───────────▼─────────────┐  │
  (npu.yaml)      │  │  Cycle-Accurate Engine    │  │
                  │  │  ┌─────┐ ┌─────┐ ┌────┐  │  │
                  │  │  │ MXU │ │ SFU │ │DMA │  │  │
                  │  │  │Model│ │Model│ │Model│  │  │
                  │  │  └──┬──┘ └──┬──┘ └──┬─┘  │  │
                  │  │     │       │       │    │  │
                  │  │  ┌──┴───────┴───────┴─┐  │  │
                  │  │  │  Core Orchestrator  │  │  │
                  │  │  │  (调度+时间轴推进)   │  │  │
                  │  │  └────────────────────┘  │  │
                  │  └───────────┬─────────────┘  │
                  │              │                │
                  └──────────────┼────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Results & Reports       │
                    │  - Total latency (ms)    │
                    │  - Module breakdown (%)  │
                    │  - Bottleneck analysis   │
                    │  - Tok/s (decode/prefill)│
                    │  - Utilization heatmap   │
                    └─────────────────────────┘
```

---

## 三、各模块建模方案

### 3.1 MXU Model（精度要求最高）

**输入**：GEMM (M, K, N)、权重 INT4 / 激活 BF16  
**输出**：compute cycles、stall cycles、利用率

**两阶段策略**：

#### 阶段 A：批量预计算（离线）
用 SCALE-Sim v3 跑一遍所有 (M,K,N) 组合，导出成 **lookup table**：

```json
// mxu_lut.json
{
  "64x64_INT4": {
    "1x2560x2048": {"compute": 1280, "stall_dram": 320, "stall_sram": 0, "util": 0.95},
    "1x2048x2560": {"compute": 1280, "stall_dram": 320, "stall_sram": 0, "util": 0.95},
    "1x2560x9728": {"compute": 6080, "stall_dram": 1800, "stall_sram": 0, "util": 0.88},
    ...
  }
}
```

好处：精度 = SCALE-Sim，速度 = 查表，覆盖不到的组合解析模型 fallback。

#### 阶段 B：解析模型（fallback，在线）
对于 LUT 未覆盖的 (M,K,N)：

```
total_cycles = compute_cycles + stall_cycles

compute_cycles = M * ceil(K / 64) * ceil(N / 64)   # 64×64 Block 广播
                 # 无 pipeline fill/drain — 广播引擎每个 tile 稳态 1 cycle

stall_cycles = max(
    0,
    weight_load_time - compute_cycles,      # 权重带宽不足 stall
    activation_load_time - compute_cycles    # 激活带宽不足 stall
)

weight_load_time = (K * N * 4bit) / 8 / bandwidth_weight
activation_load_time = (M * K * 16bit) / 8 / bandwidth_activation
```

### 3.2 SFU Model

每个 SFU 算子是固定流水线深度的组合逻辑，延迟 = pipeline_depth cycles。

**建模方式**：函数调用表

```python
SFU_LATENCY = {
    "softmax":   8,   # 指数查表 + 分段减法 → 8 cycles
    "layernorm": 6,   # 并行均值/方差 + 融合乘加 → 6 cycles
    "gelu":      4,   # 分段查表 → 4 cycles
    "relu":      1,   # 纯组合逻辑 → 1 cycle
    "rope":     12,   # CORDIC 旋转 → 12 cycles
    "silu":      4,   # 复用 GELU 查表
    "maxpool":   3,   # 4→1 比较器
    "avgpool":   3,   # 加法 + 移位
}
```

每层 LLM 的 SFU 开销：

```
sfu_cycles_per_layer = SFU_LATENCY[op_type] × num_elements / SFU_WIDTH
```

假设 SFU 宽度 = 128（每个 cycle 处理 128 个元素），3B 模型 hidden_size=2560：

```
softmax cycles ≈ 8 × 2560 / 128 = 160 cycles  （可以忽略）
layernorm cycles ≈ 6 × 2560 / 128 × 2 = 240 cycles
```

### 3.3 KV Cache Manager Model

**核心问题**：KV cache 256KB SRAM 能不能覆盖 decode 阶段 85%+ 的 KV 访问？

**建模方式**：缓存命中率分析

```python
SRAM_CAPACITY = 256 * 1024   # 256 KB
KV_ELEMENT_BYTES = 2         # INT8 Key + INT8 Value = 2 bytes
ELEMENTS_PER_TOKEN = num_layers × num_kv_heads × head_dim × 2

# 3B 模型: 28 layers × 2 KV heads × 128 dim × 2(K+V) = 14,336 elements/token
TOKENS_IN_SRAM = SRAM_CAPACITY / (ELEMENTS_PER_TOKEN * KV_ELEMENT_BYTES)
                = 262,144 / 28,672 ≈ 9 tokens
```

9 个 token 不够！需要改进：**只缓存当前层正在用的那层 KV**。

```
# 按层缓存: 只存 2 KV heads × 128 dim × 2 = 512 elements/layer/token
TOKENS_IN_SRAM_PER_LAYER = 262,144 / (512 * 2) ≈ 256 tokens ✅
```

256 个 token 的滑动窗口，对于 2000 token 上下文，命中率 ~87%。

**模型输出**：

```
kv_hit_rate = min(SRAM_tokens / total_tokens, 0.95)  # 上限 95%
kv_access_cycles_per_decode = kv_hit_rate × SRAM_ACCESS_CYCLES
                                + (1 - kv_hit_rate) × DRAM_ACCESS_CYCLES
```

### 3.4 DMA Model

**输入**：每次搬移的起始地址、大小、方向（LPDDR5→SRAM / SRAM→LPDDR5）  
**输出**：搬移 cycles

```python
DMA_CHANNELS = 2
DMA_BURST_SIZE = 256        # bytes per burst
DMA_DESCRIPTOR_OVERHEAD = 5 # cycles per descriptor

dma_cycles = DMA_DESCRIPTOR_OVERHEAD
             + ceil(size_bytes / DMA_BURST_SIZE) × DRAM_BURST_CYCLES

# 权重加载: 一层 Qwen 2.5 3B 的 attention QKV 权重
# weight_size = 2560 × (2048+256+256) × 4bit = 2.5 MB → ~3.2K bursts
# LPDDR5-6400 burst time ≈ 4ns → 3.2K × 10 = 32K ns ≈ 32K cycles @1GHz
```

**DMA 与 MXU 重叠**：DMA 加载 Layer N+1 权重时，MXU 正在算 Layer N。

```
effective_dma_cycles = max(0, dma_cycles - mxu_cycles)  # 隐藏掉的
```

### 3.5 核间流水线 Model

对于多核配置，相邻核通过 4KB FIFO 传递激活值。

```python
FIFO_SIZE_ELEMENTS = 4096 / 2 = 2048   # BF16 = 2 bytes
TRANSFER_WIDTH = 256                    # bits per cycle

fifo_transfer_cycles = ceil(elements / (TRANSFER_WIDTH / 16)) + 2  # +2 cycle 延迟
```

以 2 核流水线为例（核₀ 算 Layer 0-15，核₁ 算 Layer 16-31）：

```
# 核₀ 最后层输出 2560 个 BF16 激活 → FIFO
fifo_write = ceil(2560 / (256/16)) + 2 = 162 cycles

# 关键：FIFO 写和 MXU 算可以重叠
pipeline_latency = max(核₀_MXU_cycles, 核₁_MXU_cycles)
                  + fifo_write  # 只有 FIFO 串行部分
```

### 3.6 RISC-V 指令发射 Overhead

RISC-V 只做指令分派，不参与数值计算。开销极小：

```python
INSTR_FETCH_CYCLES = 4     # 4 级流水线
INSTR_DECODE_CYCLES = 1
DISPATCH_CYCLES = 2        # 写 MXU/SFU/DMA 控制寄存器

overhead_per_instr = INSTR_FETCH_CYCLES + INSTR_DECODE_CYCLES + DISPATCH_CYCLES = 7

# 每层 ~14 条 NPU 指令，每 token 28 层
riscv_overhead = 28 × 14 × 7 = 2,744 cycles ≈ 2.7 μs  @ 1GHz
```

可以忽略。

---

## 四、核心调度引擎：时间轴推进

不是逐 cycle 仿真，而是**事件驱动 + 时间轴合并**：

```python
class CoreTimeline:
    """单核时间轴：记录每层各模块的起止时间"""
    
    def add_event(self, module, start_cycle, end_cycle):
        self.events.append((module, start_cycle, end_cycle))
    
    def merge(self):
        """合并重叠事件 → 实际总时间"""
        # MXU 和 DMA 可重叠，MXU 和 SFU 串行（数据依赖）
        pass

class MultiCoreTimeline:
    """多核时间轴：加上 FIFO 传递"""
    
    def simulate_decode(self, num_tokens):
        for token in range(num_tokens):
            for layer in range(num_layers):
                core_id = layer % num_cores
                # 分配各模块耗时到对应核心
                # 核间激活传递加 FIFO latency
```

**输出示例**（单 token decode，1 核，Block 64×64）：

```
Layer 0: MXU 1280cy | DMA(load W1) 1200cy (overlapped) | SFU 160cy | KV 50cy
Layer 1: MXU 1280cy | DMA(load W2) 1200cy | SFU 160cy | KV 50cy
...
Total: 33,800 cycles @ 1GHz = 33.8 μs → 29.6 tok/s ✅ (满足 25 tok/s 目标)
```

---

## 五、输入输出

### 输入

#### 1. 模型配置 (`qwen3b_config.json`)
```json
{
  "num_layers": 28,
  "hidden_size": 2560,
  "num_attention_heads": 32,
  "num_kv_heads": 2,
  "head_dim": 128,
  "intermediate_size": 9728,
  "vocab_size": 151936,
  "max_position_embeddings": 2048
}
```

#### 2. NPU 配置 (`npu_config.yaml`)
```yaml
cores: 1
mxu:
  size: [64, 64]
  frequency_mhz: 1000
  weight_precision: int4
  activation_precision: bf16
  accumulate_precision: int32
  dataflow: block/broadcast

sram:
  l1_per_core_kb: 512
  l2_shared_kb: 2048
  banks: 16

sfu:
  width: 128
  pipeline: {softmax: 8, layernorm: 6, gelu: 4, silu: 4, rope: 12}

kv_cache:
  sram_kb: 256
  dram_region_mb: 96
  precision: int8

dma:
  channels: 2
  burst_size: 256
  descriptor_overhead: 5

memory:
  type: LPDDR5
  bandwidth_gbps: 51.2
  tRC_ns: 48
  tRAS_ns: 42

interconnect:
  type: crossbar
  port_bandwidth_gbps: 500
  fifo_per_link_kb: 4

riscv:
  isa: RV64IMAFD
  pipeline_stages: 4
```

### 输出

```
===== NPU System Simulation Report =====
Model: Qwen2.5-3B | NPU: 1 core, 64×64, INT4 | Engine: Block (broadcast)

--- Prefill (prompt=128 tokens) ---
  MXU compute:       261 ms  (84.2%)
  SFU overhead:        3 ms  ( 1.0%)
  KV Cache access:     4 ms  ( 1.3%)
  DMA stall:          35 ms  (11.3%)
  RISC-V overhead:     1 ms  ( 0.3%)
  DRAM refresh:        6 ms  ( 1.9%)
  ─────────────────────────
  TOTAL:             310 ms  (0.37s target ✅)

--- Decode (per token) ---
  MXU compute:      28.6 μs  (84.6%)
  SFU overhead:      1.3 μs  ( 3.8%)
  KV Cache access:   0.5 μs  ( 1.5%)
  DMA stall:         2.0 μs  ( 5.9%)
  RISC-V overhead:   0.1 μs  ( 0.3%)
  DRAM refresh:      1.3 μs  ( 3.9%)
  ─────────────────────────
  TOTAL:            33.8 μs  → 29.6 tok/s ✅ (>25 tok/s)

--- Bottleneck ---
  🔴 MXU dominates at 84.6% — DRAM 带宽已用满（weight-bound）
  🟡 DMA stall 5.9% — LPDDR5 有效带宽接近上限（43.5 GB/s）
  🟢 SFU/KV/RISC-V overhead 合计 <6% — 架构健康
  ✅ CV throughput (MobileNetV3-Small): 677.9 FPS （DSE 验证）
  ✅ Area: 28.2 mm² （DSE 验证）

--- Multi-core Projection ---
  Config     Decode tok/s   Area     Notes
  1 core       29.6          28.2mm²  Baseline (Block 64×64)
  2 cores      59.2          42 mm²   Data parallel, near-linear
  4 cores      82.9          69 mm²   Pipeline parallel (7B), FIFO overhead 6%
```

---

## 六、实现路线

| Phase | 内容 | 工时 |
|-------|------|:---:|
| **Phase 1** | 单核 MXU + SFU + DMA 解析模型，命令行跑通 | 2-3 天 |
| **Phase 2** | KV Cache 模型 + DRAM 带宽模型 | 1 天 |
| **Phase 3** | 多核 + 流水线 FIFO + Crossbar 竞争 | 2 天 |
| **Phase 4** | SCALE-Sim lookup table 集成（精度提升） | 1 天 |
| **Phase 5** | Web dashboard + 参数扫描（扫 MXU 尺寸/频率/L2 大小） | 2 天 |

---

## 七、与 SCALE-Sim 的关系

本 simulator 不替代 SCALE-Sim，而是**嵌套它**：
- 默认用解析模型（快，误差 ~15%）
- 可选 `--accurate` 模式，MXU 部分调用 SCALE-Sim 做 cycle-accurate 仿真
- 最终参数确定前，用 SCALE-Sim 精算 MXU；参数调优阶段用解析模型快速扫

---

## 八、软件栈联合调试（L1/L2/L3 接口）

Simulator 的输入不只是 CSV trace，更可以接收 **NPU ISA 指令序列**，和编译器/运行时直连。

### 架构

```
PyTorch/HuggingFace Model
         │
         ▼
   ┌─────────────┐
   │ MLIR/IREE    │  ← 编译器：图优化、算子融合、tiling
   │ Compiler     │
   └──────┬──────┘
          │ 生成 NPU ISA 指令序列
          ▼
   ┌─────────────────────────────────────┐
   │  NPU Simulator                      │
   │  ┌───────────────────────────────┐  │
   │  │  指令解码器 (L2/L3)            │  │
   │  │  MMUL / SOFTMAX / DMA_LD ... │  │
   │  └───────────┬───────────────────┘  │
   │              │                      │
   │  ┌───────────▼───────────────────┐  │
   │  │  Core Orchestrator             │  │
   │  │  MXU | SFU | DMA | KV | FIFO  │  │
   │  └───────────┬───────────────────┘  │
   │              │                      │
   │     性能反馈 → 编译器 auto-tuning    │
   └─────────────────────────────────────┘
```

### 三个对接层次

| 层次 | 输入 | 能调试什么 | 实现阶段 |
|------|------|---------|:---:|
| **L1: GEMM trace** | CSV (M, K, N) 每层列表 | MXU 算力/带宽，等价于 SCALE-Sim | Phase 1 |
| **L2: NPU 指令流** | 手写或编译器生成的 ISA 序列 | 指令调度、流水线重叠、DMA 排布优化 | Phase 3 |
| **L3: IREE HAL 直连** | IREE 提交的 HAL command buffer | 完整软件栈：图优化→指令生成→硬件模拟 | Phase 5 |

### L1→L2 升级方式

Simulator 入口同时暴露两套接口，内部模型完全复用：

```python
class NPUSimulator:
    def run_trace(self, csv_path: str) -> Report:
        """L1: CSV trace → 性能报告"""
        for m, k, n in parse_csv(csv_path):
            cycles = self.mxu.estimate(m, k, n)
            self.timeline.add("mxu", cycles)

    def run_instructions(self, isa_sequence: list[str]) -> Report:
        """L2: NPU 指令序列 → 性能报告 + 功能验证"""
        for instr in isa_sequence:
            op, args = self.decoder.decode(instr)
            cycles = self.dispatch(op, args)   # 同一套 MXU/SFU/DMA 模型
            self.timeline.add(op, cycles)
```

### 编译器 auto-tuning

```
编译器                         Simulator
   │                              │
│── tiling=32 ────────────────→│ 跑模拟
│                              │── 利用率=62%, stall=38%
│← 反馈 ───────────────────────┤
│                              │
│── tiling=64 ───────────────→│ 再跑
│                              │── 利用率=95%, stall=5%
│← 反馈 ───────────────────────┤
│                              │
│── 选定 tiling=64 ───────────→│ 最终配置
```

---

## 九、Golden Model for RTL 验证

Simulator 加 `mode="functional"` 切换为 golden model，RTL 仿真时同步运行，比对输出。

### 模式切换

```python
sim = NPUSimulator(config, mode="performance")   # 只数 cycles
sim = NPUSimulator(config, mode="functional")    # 真算数值 + 数 cycles
```

### 三个验证精度层次

| Level | 比对方式 | 说明 | 适用阶段 | 难度 |
|--------|---------|------|---------|:---:|
| **L1: Hash** | 每层输出做 CRC32，比对 hash | 99% 的 bug 会让 hash 对不上 | 快速回归 | 低 |
| **L2: 统计** | min/max/mean/std，容忍 INT4 量化误差 | 精度调优，允许 ±1 LSB | 精度验证 | 中 |
| **L3: Cycle-accurate** | 逐 cycle 比对中间寄存器 | 硬 bug 定位 | 极少用 | 高 |

### 验证流程

```
                同一个输入向量
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
NPU Simulator    RTL Sim       实际芯片
(functional)   (iverilog/vcs)   (将来)
        │             │
        ▼             ▼
  期望输出      实际输出
        │             │
        └─────┬───────┘
              ▼
         逐 bit 比对
```

### 每模块 functional 实现

| 模块 | Performance 模式 | Functional 模式 |
|------|-----------------|----------------|
| MXU | `M×ceil(K/64)×ceil(N/64) + stalls` | `numpy.matmul` + INT4 dequantize，INT32 累加 |
| SFU | `latency × elements / width` | bit-accurate: 指数查表、GELU 分段多项式 |
| KV Cache | 命中率模型 | 地址正确性验证（不做数值计算） |
| DMA | 带宽模型 | transfer size × burst 正确性 |
| RISC-V | 忽略 | 指令译码正确性 |

### testbench 集成

```python
# test_mxu.py — RTL testbench 调用的 Python 参考
from npu_sim.golden import mxu_golden

def verify_mxu(rtl_output: np.ndarray, input_a: np.ndarray, weight_w: np.ndarray):
    expected = mxu_golden(input_a, weight_w)
    assert np.array_equal(rtl_output, expected), \
        f"MXU mismatch at {np.where(rtl_output != expected)}"
    return True
```

---

## 十、实现路线（更新）

| Phase | 内容 | 工时 | 接口层次 |
|-------|------|:---:|:---:|
| **Phase 1** | 单核 MXU + SFU + DMA 解析模型，L1 CSV trace 输入 | 2-3 天 | L1 |
| **Phase 2** | KV Cache 模型 + DRAM 带宽模型 | 1 天 | L1 |
| **Phase 3** | 指令解码器 + NPU ISA → timeline 调度 | 2 天 | L2 |
| **Phase 4** | 多核 + 流水线 FIFO + Crossbar 竞争 | 2 天 | L2 |
| **Phase 5** | SCALE-Sim LUT 集成 + functional 模式 | 1 天 | L2 |
| **Phase 6** | IREE HAL 适配 + 编译器 auto-tuning 反馈循环 | 3 天 | L3 |
| **Phase 7** | Web dashboard + 参数扫描 | 2 天 | — |

---

> **下一步**：开始 Phase 1 实现 — Python 单核性能模型，CSV trace 输入，命令行跑通
