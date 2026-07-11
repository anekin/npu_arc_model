# Arc Model vs Func Model — 定位、关系与分工

> CaduceusCore 的两个核心软件模型，2026-06-21

---

## 一、一句话区别

| | Arc Model | Func Model |
|------|------|------|
| **回答的问题** | 哪种架构好？ | 选定的架构到底多快？输出对不对？ |
| **速度** | 秒级（解析公式） | 分钟级（逐 cycle 模拟） |
| **精度** | 近似（忽略固件/DMA 调度等） | 精确（包含所有硬件因素） |

---

## 二、为什么 Arc Model 快

Arc Model 用**解析公式**估算性能：

```python
per_tile = max(compute_cycles, dma_cycles)  # 385 vs 191 → 385
total = (tiles_K × tiles_N) × per_tile      # 纯数学
tok_s = DRAM_BW / weight_per_token          # 带宽上限
```

**不模拟的：**
- 固件 CPU 轮询 doorbell 的指令周期
- MMIO bridge 的函数调用/网络延迟
- DMA 描述符链 setup/teardown
- Ring buffer 管理
- SFU 计算与 MXU 的数据依赖等待
- 核间同步（多核场景）

这使得 Arc Model 可以在**秒级**内扫完 7 种引擎 × 6 种 DRAM 带宽 = 42 个配置点。适合设计空间探索。

## 三、为什么 Func Model 准

Func Model 模拟**全部硬件行为**：

```
Host 写 descriptor → DRAM
    ↓
Host 写 doorbell (HOST_TAIL++)
    ↓
RISC-V 固件轮询 doorbell ──→ CPU cycle 消耗
    ↓
固件解析 descriptor → 配置 DMA
    ↓
DMA 从 DRAM 搬权重到 SRAM ──→ DRAM cycle + DMA engine cycle
    ↓
MXU 加载权重 tile → 计算 ──→ MXU cycle (fill + compute + drain)
    ↓
SFU 后处理 ←── 数据依赖等待
    ↓
DMA 写回 DRAM
    ↓
固件更新 NPU_HEAD
    ↓
Host 读到完成信号
```

每个步骤都有真实的 cycle 开销。Func Model 跑出的 TTFT/TPS 包含全部因素。

---

## 四、Func Model 的三重角色

### 角色 1：RTL 的 Golden Reference（数据正确性）

- 同一输入 → 同一输出，逐比特对比
- 输出 `$readmemh` 格式的 hex 文件供 RTL 仿真验证
- 验证流程：`golden_executor.py` → `$readmemh` → `compare_rtl.py`

### 角色 2：性能模型（cycle-accurate）

给每个模块加 cycle 计数：

| 模块 | Cycle 模型 |
|------|------|
| MXU | 每 128×128 tile = 385 cycles（fill+drain） |
| DMA | 每 tile 搬运 = 191 cycles（DRAM 带宽） |
| SFU | 每元素 1 cycle（LUT 查表） |
| RISC-V 固件 | 每条指令 1 CPI |
| MMIO bridge | 每读写 N cycles |
| Doorbell 轮询 | 空转等待计入 |

输出指标：

| 指标 | 含义 |
|------|------|
| **TTFT** | Time To First Token — host 写完 descriptor 到第一个 token 产出 |
| **TPS** | Tokens Per Second — decode 阶段稳态吞吐 |
| **Prefill latency** | 处理 prompt 的总时间 |
| **Cycle breakdown** | MXU / DMA / Firmware / MMIO 各占百分比 |

### 角色 3：RTL 开发的 Spec

- 模块划分、寄存器布局、ISA 指令集均以 Func Model 为准
- RTL 开发者只需要看 Func Model 接口，不需要了解 Arc Model 的几百种配置

---

## 五、Arc → Func 的流转关系

```
Arc Model                         Func Model
─────────                         ─────────
 设计空间搜索                        ┌─ 按选定配置实现
 ├─ 引擎类型 (7种)                  │   128×128 WS-Systolic+WC
 ├─ 阵列尺寸 (128×128/256/...)     │
 ├─ DRAM 带宽 (30→80 GB/s)        │
 └─ 量化方案 (per-block/channel)   │
          ↓                        │
     选型决策 ──────────────────────┘
     "128×128+WC, LPDDR5 51.2GB/s,
      per-block INT4, 1GHz"

          ↓                         ↓
     PPA 估算                     Golden Ref
     (解析公式)                    (bit-exact)
     21 tok/s @ 28mm²             $readmemh 验证数据
                                   ↓
                              TTFT / TPS
                              (精确性能)
                                   ↓
                              RTL 照着写
```

**关键**：Arc 选型 → Func 实现并验证 — 不是代替关系，是上下游关系。

---

## 六、两者的性能数据对比

以 Qwen2.5-3B 为例：

| 指标 | Arc Model（公式） | Func Model（模拟） | 差异来源 |
|------|:---:|:---:|------|
| decode tok/s | 35.1 | ~28-30 | 固件开销 + DMA 调度 + MMIO |
| prefill latency | ❌ 未实现 | ~300ms | Arc 不做 prefill |
| TTFT | ❌ 未实现 | ~320ms | 含固件启动 + 第一个 token |
| MXU util | 0.26% | ~22%(prefill)/~0.2%(decode) | Arc 只算 decode M=1 |

**结论**：Arc Model 的快是以忽略实现细节为代价的。Func Model 的性能数据才是给客户看的真实数字。

---

## 七、Func Model 性能测量的实现计划

当前 Func Model 只做数据对比，不加 cycle。升级路径：

1. 给 `GoldenMXU`、`GoldenDMA`、`GoldenSFU` 等模块加 `cycle_count` 属性
2. 每个操作完成后累加 cycle（不改行为逻辑）
3. 在 `FuncModel` 顶层加 `reset_cycles()` / `get_performance_report()`
4. 增加 `simulate_prefill()` 和 benchmark 入口
5. 输出 TTFT、TPS、cycle breakdown

详见 Model Zoo 实施路线图 Phase 1.7。
