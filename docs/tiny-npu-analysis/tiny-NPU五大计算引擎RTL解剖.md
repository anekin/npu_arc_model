---
date: 2026-07-10
source: "https://mp.weixin.qq.com/s/eDXU3K22MJWZLxy8pmihng"
author: AI驱动IC
tags: [NPU, tiny-NPU, RTL, 计算引擎, CaduceusCore, 微架构]
series: "tiny-NPU 内部揭秘"
related: [[CaduceusCore架构]], [[NPU计算引擎设计]]
research_loop_feedback: "多引擎专用化 + 记分板并发 + 流式Softmax"
---

# tiny-NPU 五大计算引擎 RTL 级解剖

> Day16：继 GEMM 脉动阵列后，拆解 Softmax / LayerNorm / GELU / Vec / DMA 五个专用引擎。

## 一、引擎全景

| 引擎 | 记分板位 | 功能 | 核心硬件 |
|:---|:--|:---|:---|
| Softmax | bit 0 | 注意力概率归一化 | 3 次扫描 + exp LUT + reduce_max/sum |
| LayerNorm | bit 1 | 层归一化 | 2 次扫描 + mean_var_engine + rsqrt LUT |
| GELU | bit 2 | 激活函数 | 查表法，GELU/SiLU 双模式 |
| Vec | bit 3 | 逐元素向量 ALU | 4 操作 ALU + COPY2D 模式 |
| DMA | bit 4 | 片外数据搬运 | AXI4 读写通道 + 突发控制 |

**统一命令接口**：所有引擎用同一套 `cmd_valid/cmd_ready` 握手 + 操作数 + `done` 信号。
**记分板**：6 bit，每引擎 1 bit 忙标志，bit 5 保留。

## 二、Softmax 引擎 — 三次扫描

```
Pass 1: 找 max → 全局最大值
Pass 2: exp(x_i - max) → 累加 sum → 写入 scratch SRAM
Pass 3: scratch × recip(sum) → clamp → 写回 dst
```

- **17 状态 FSM**，覆盖 INT8/FP16 双精度
- FP16：在线跟踪 max，不需要等 reduce_max → 直接进 Pass 2
- INT8：需要 S_P1_WAIT 等 reduce_max 模块出结果
- 中间 scratch SRAM（16 位宽）暂存 exp 结果
- **8 位宽 SRAM**：FP16 需 2 周期读（先低字节再高字节）

## 三、架构启示

1. **专用化 > 通用化**：每种计算原语一个引擎，面积小、可并行
2. **流式处理省面积**：Softmax 不需要全量缓冲区，三趟扫描即可
3. **记分板解耦控制**：微码只管发射，引擎自己管执行
4. **双精度 FSM 分支**：`is_fp16` 信号控制状态跳转路径
5. **SRAM 宽度取舍**：8 位宽 = 面积优先，FP16 吞吐折半

## 四、对 CaduceusCore 的启发

- 控制单元：微码 + 记分板模式，不搞集中式控制器
- 计算引擎：拆成 GEMM/Softmax/LayerNorm/Activation/DMA 五个独立模块
- Softmax：直接借鉴三趟扫描 + scratch SRAM 方案
- 精度：INT8/FP16 双路径，FSM 分支处理
- Model Zoo 推理：INT8 路径可直接参考 tiny-NPU 的量化设计
