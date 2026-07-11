---
date: 2026-07-10
tags: [NPU, architecture, tiny-NPU, CaduceusCore, comparison]
related: [[tiny-NPU五大计算引擎RTL解剖]], [[CaduceusCore架构]]
---

# tiny-NPU vs CaduceusCore — 深度架构对比

## 一、定位差异

| 维度 | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| 设计哲学 | **教学级 RTL 参考** — 展示怎么做对 | **产品级 DSE** — 找最优面积/性能平衡 |
| 目标模型 | 小 Transformer 推理 | 3B-7B LLM + CV/VLM |
| 工艺节点 | 未指定（FPGA/轻量 ASIC 取向） | TSMC 12nm（面积经 TPUv1 die-shot 校准） |
| 内存系统 | 片上 SRAM（文章未详述容量） | LPDDR5-64b 或 3D DRAM 500GB/s，含 DRAM PHY 建模 |
| 验证方法 | RTL 直接实现 | Arc Model → Func Model → E2E → RTL 四阶段 |

---

## 二、计算引擎架构对比

### 2.1 引擎数量与分工

| | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| **MAC 引擎** | 1 种（GEMM 脉动阵列） | **8 种可选**（systolic/block/FSA/GMMA/WMMA/OS/IS/TC） |
| **Softmax** | 专用引擎，3-pass 扫描 | SFU 内 LUT 实现（256 条目） |
| **LayerNorm** | 专用引擎，2-pass + rsqrt LUT | SFU 内实现 |
| **GELU/SiLU** | 专用引擎，查表法双模式 | SFU 内 LUT 实现（64 条目 GELU + SiLU） |
| **向量 ALU** | 专用引擎，4 操作 + COPY2D | Vector 模块，6 操作（ADD/MUL/REDUCE/CONV/RESID） |
| **DMA** | 专用引擎，AXI4 读写 | DMA 模块，双通道 + descriptor chain |
| **RoPE** | ❌ 未提及 | ✅ SFU CORDIC 12 级流水线 |

### 2.2 核心设计差异

| | tiny-NPU | CaduceusCore |
|:---|:---|:---|
| **Softmax 实现** | **流式 3-pass** — 不存全量中间结果，省 SRAM | LUT 查表 + Vector reduce，依赖大 SRAM |
| **SRAM 宽度** | **8-bit** — 面积优先，FP16 需要 2 周期 | 未在架构文档详述（可配置） |
| **精度策略** | INT8/FP16 双精度，FSM 内 `is_fp16` 分支 | INT4 权重 + BF16/FP16 激活，per-block(g=128) 量化 |
| **控制模式** | 微码 + **记分板并发** — 5 引擎可同时跑 | 微码/固件调度，tile-level double-buffer |

---

## 三、控制架构对比

### tiny-NPU：记分板 + 统一命令接口 ⭐

```
微码发射 → cmd_valid/cmd_ready 握手 → 引擎独立运行 → done → 记分板清除
```

- 6-bit 记分板，每引擎 1 bit
- **优势**：控制简单，引擎间天然解耦，适合多引擎并发
- **局限**：只支持 6 个引擎（1 bit 保留），扩展需改记分板宽度

### CaduceusCore：固件调度 + tile-level double-buffer

```
固件 → 写 descriptor → MMIO 桥 → DMA 搬 tile → MXU 计算 → 写回 → 下一 tile
```

- 15-field UINT32 descriptor 协议
- SRAM tile 级调度（K-block × N-tile double-buffer）
- **优势**：灵活，可以处理任意大矩阵，支持 ACCUMULATE 模式
- **局限**：固件复杂度高，RISC-V 轮询 doorbell 有开销

---

## 四、面积效率对比

| | tiny-NPU | CaduceusCore (S1 LPDDR5) |
|:---|:---|:---|
| MAC 阵列 | 1 个 GEMM（尺寸未透露） | block 64×128 或 FSA 128×128 |
| SFU 开销 | 分散到 4 个专用引擎 | 集中 SFU + Vector |
| 总面积 | 未公布 | 61mm² @12nm（含 DRAM PHY 14.7 + PCIe 5.9） |
| **设计倾向** | **面积最小化** — 8bit SRAM、流式 Softmax、查表 GELU | **面积/性能 Pareto** — 8 引擎 DSE 选最优 |

---

## 五、优劣势总结

### tiny-NPU 的优势

1. **流式 Softmax 省面积** — 不需要大 scratch buffer，3-pass 即可
2. **记分板并发模型** — 多引擎自然并行，控制逻辑极简
3. **统一命令接口** — 所有引擎同一套握手协议，扩展新引擎成本低
4. **INT8/FP16 FSM 内分支** — 不额外增加数据通路
5. **工程可读性** — RTL 代码直接展示了每个引擎的完整 FSM，适合学习和参考

### tiny-NPU 的劣势

1. **单 MAC 引擎** — 只有一种 GEMM 架构，无法适应不同带宽场景
2. **没有 DRAM 带宽模型** — 不分析内存墙，无法评估实际吞吐上限
3. **没有量化验证** — 没有像 CaduceusCore 的 cos_sim 精度门
4. **没有软件栈** — 缺少 llama.cpp/ExecuTorch 后端
5. **缺少 RoPE/Residual Add** — Transformer 必需的部分操作未覆盖
6. **扩展性不明** — 记分板 6 bit 硬上限，SRAM 容量/DRAM 接口未讨论

### CaduceusCore 的优势

1. **8 引擎 DSE** — 不是选定一个架构，而是让 Pareto 前沿说话
2. **完整验证链** — Arc(数值精度) → Func(bit-exact) → E2E(llama.cpp 全栈) → RTL
3. **DRAM 墙壁建模** — 75% 效率、BW 天花板、DRAM PHY 面积，不画饼
4. **双场景策略** — LPDDR5 低成本 + 3D DRAM 高性能，一颗 die 不一定打天下
5. **软件栈** — ggml NPU backend + ExecuTorch delegate
6. **面积可溯源** — TPUv1 ISCA 2017 die-shot 为锚，不凭空估面积

### CaduceusCore 的劣势

1. **控制复杂度高** — tile scheduler、descriptor 协议、固件开销（Arc→Func 差 10-20%）
2. **Softmax 依赖大 SRAM** — 不像 tiny-NPU 的流式方案省面积
3. **没有记分板式并发** — 引擎间依赖固件串行调度，不如 tiny-NPU 优雅
4. **工程门槛高** — DSE 需要 4-phase pipeline，跑错一次就推倒重来
5. **面积模型仍在迭代** — PE 面积从 24→4→2mm² 校准了三次才稳定

---

## 六、互相借鉴

| tiny-NPU → CaduceusCore | CaduceusCore → tiny-NPU |
|:---|:---|
| 流式 3-pass Softmax — 省 SRAM 面积 | per-block INT4 量化方案 + cos_sim 验证 |
| 记分板并发模型 — 控制简化 | DRAM 带宽瓶颈分析 — 别忽略内存墙 |
| 统一 cmd 接口 + done 握手 — 引擎解耦 | 多引擎 DSE — 一个 MAC 架构不够 |
| FSM 内 is_fp16 分支 — 双精度零额外硬件 | 软件栈（llama.cpp → ExecuTorch） |

**最大启发**：tiny-NPU 的控制哲学（记分板 + 统一接口 + 流式处理）可以直接移植到 CaduceusCore 的 Func Model，降低固件调度复杂度。CaduceusCore 的 DSE 方法 + 量化验证也可以帮助 tiny-NPU 评估实际部署性能。
