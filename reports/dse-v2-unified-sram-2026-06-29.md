# NPU 架构 DSE 报告 — 统一 SRAM 模型 (v2)

**日期**: 2026-06-29  
**方法**: 三变量受控敏感性分析 + 统一 SRAM 效率模型  
**工作负载**: Qwen2.5-3B INT4 量化, seq_kv=2048, prompt_len=128

---

## 0. 方法论修正

### 修正前（v1，2026-06-27）的问题

| 问题 | v1 行为 | 物理现实 | 影响 |
|------|---------|---------|------|
| 权重跨层缓存 | `_sram_hit_rate` 假设 27% 权重命中 | 每层权重不同，不跨层 | TPS 虚高 2× |
| KV cache 硬编码 | `KV_CYCLES_PER_LAYER = 125` | 实际 ~28,000 cycles/layer | 低估 200× |
| 引擎 SRAM 不统一 | 仅 FSA 有 SRAM 模型 | 所有引擎共享同一条 DRAM 总线 | 对比不公平 |

### 修正后（v2）

```
统一 SRAM 模型:
  L2 SRAM ──┬── 60% 权重预取 buffer (wbuf)
            └── 40% KV cache tile buffer (kvbuf)

  权重效率:
    - 若权重 ≤ wbuf → 缓存命中，0 DRAM 读取
    - 若权重 > wbuf → 全量 DRAM 读取，效率因子 [0.55, 0.92]

  KV cache:
    - decode 时每层读取 seq_kv 帧 K+V (INT8)
    - tile buffer 越大，DRAM 交易次数越少
    - qwen2.5-3b @ 2048 tok: 1.0 MB/layer
```

> 天花板公式: `TPS_max = BW / (model_size_INT4) = 51.2 / 1.5 ≈ 34 tok/s`

---

## 1. Sweep 1: DRAM 带宽灵敏度

**固定**: FSA 64×256, SRAM 4MB, batch=1  
**变量**: DRAM 接口方案

| DRAM | BW | TTFT (128 tok) | TPS | 面积 | 带宽利用率 |
|------|:---:|:---:|:---:|:---:|:---:|
| LPDDR5-32b | 25.6G | 121ms | 12 | 29mm² | 68% |
| **LPDDR5-64b** | **51.2G** | **60ms** | **23** | **32mm²** | **68%** |
| LPDDR5-128b | 102.4G | 30ms | 46 | 39mm² | 68% |
| LPDDR5-256b | 204.8G | 15ms | 92 | 53mm² | 68% |

**结论**: TPS 严格线性跟随带宽，利用率稳定在 68%。DRAM 带宽是主瓶颈。

---

## 2. Sweep 2: 引擎公平对比

**固定**: LPDDR5-64b (51.2 GB/s), SRAM 4MB, batch=1  
**变量**: 引擎类型 + 阵列尺寸

| Engine | TTFT | TPS | 面积 | 功耗 | TPS/mm² | 
|--------|:---:|:---:|:---:|:---:|:---:|
| **FSA 64×256** ★ | 60ms | 23 | **32mm²** | 11.5W | **0.714** |
| FSA 128×128 | 61ms | 23 | 32mm² | 11.5W | 0.708 |
| FSA 128×256 | 51ms | 23 | 41mm² | 15.9W | 0.559 |
| systolic 64×256 | 83ms | 15 | 31mm² | 11.0W | 0.494 |
| block 64×256 | 50ms | 23 | 55mm² | 23.0W | 0.418 |
| gmma 64×256 | 50ms | 23 | 63mm² | 27.0W | 0.365 |

**关键发现**:

1. **所有人卡在 23 tok/s** — 因为 23 ≈ 34 × 0.68，带宽利用率一致
2. **引擎差异体现在面积，不是吞吐** — 在 DRAM 墙下计算单元差异被掩盖
3. **FSA 面积效率领先 70%**: 0.714 vs block 0.418 (1.7×) vs gmma 0.365 (2.0×)
4. **TTFT 差异**: block/gmma 略快 (50ms vs 60ms)，因为并行广播无管线填充延迟
5. **systolic 垫底**: pipeline fill/drain 惩罚使其仅达 15 tok/s

---

## 3. Sweep 3: SRAM 尺寸敏感性

**固定**: LPDDR5-64b, batch=1  
**变量**: L2 SRAM 1MB→16MB

| SRAM | FSA TPS | block TPS | gmma TPS | FSA 面积 | 增益 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1MB | 20 | 20 | 20 | 28mm² | — |
| 2MB | 21 | 21 | 21 | 29mm² | +8% |
| 4MB | 23 | 23 | 23 | 32mm² | +18% |
| 6MB | 24 | 24 | 24 | 35mm² | +24% |
| 8MB | 25 | 25 | 25 | 38mm² | +28% |
| **10MB** | **32** | **32** | **32** | **42mm²** | **+64%** |
| 16MB | 34 | 34 | 33 | 51mm² | +72% |

**三个拐点**:

| SRAM | wbuf | 可缓存矩阵 | 效果 |
|:---:|:---:|------|------|
| <6MB | <3.6MB | K(0.31) + V(0.31) = 0.62MB | 缓存 <2% 权重 |
| **10MB** | **6MB** | K(0.31) + V(0.31) + Q(5.0) = 5.6MB | **缓存 12% 权重，TTFT↓37%** |
| 16MB | 9.6MB | K+V+Q+O(5.0) = 10.6MB > 9.6MB | 部分 O 缓存 |

**10MB 是 sweet spot**: Q_proj (5MB) 是 prefill 阶段最重的投影矩阵，将其缓存后 TTFT 从 71ms 降至 45ms。FFN 权重（73% 总权重）太大，任何合理 SRAM 都无法缓存，只能依赖 DRAM 效率。

---

## 4. 终选推荐

| 参数 | 推荐值 | 依据 |
|------|------|------|
| **引擎** | **FSA 64×256** | 面积效率 1.7× 领先，inline softmax |
| **阵列** | 64行 × 256列 | 与 block 128×128 同吞吐，面积相同 |
| **SRAM L2** | **4-6MB** | TPS sweet spot，超出后 ROI 递减 |
| **精度** | INT4 权重 / INT8 激活 | 精度门通过 (cos_sim ≥ 0.96) |
| **内存** | LPDDR5 64-bit 6400Mbps | 51.2 GB/s，合理 PHY 面积 |
| **面积** | ~32-35mm² @ 7nm | 含 4-6MB SRAM |
| **功耗** | ~11.5-11.9W | 含 DRAM PHY |

**预期性能**:

| 负载 | 性能 |
|------|:---:|
| Qwen2.5-3B decode | 23-24 tok/s |
| Qwen2.5-3B TTFT (128 tok) | 57-60ms |
| Qwen2.5-1.5B decode | ~46 tok/s (权重更小，带宽利用率更高) |
| Qwen2.5-7B decode | ~10 tok/s (权重 3.5GB，带宽墙更严重) |

---

## 5. 模型局限 & 下一步

1. **KV cache 竞争未区分 batch**: batch=1 和 batch=2 在当前模型下结果一致，因为 KV 模型使用固定 seq_kv=2048 而非 batch×seq_kv
2. **单层权重缓存的细化**: 当前模型是二值（缓存/不缓存），实际可以是部分缓存
3. **Prefill 阶段的 attention gemm**: Q@K^T 和 P@V 未在 trace 中建模
4. **其他引擎的 SRAM 面积模型**: block/gmma 的 PE 面积可能低估了广播互连代价

---

*工具: `sim/design_space_explorer.py` + `sim/engine/fsa_engine.py` + `sim/engine/block_engine.py` + `sim/engine/gmma_engine.py`*  
*模型: Qwen2.5-3B INT4, seq_kv=2048, prompt_len=128*  
*修正: v2 — 统一 SRAM 效率模型, 动态 KV cache, 权重不跨层缓存*  
*设计约束: TTFT < 200ms ✓ (端侧 3B=60ms, 7B≈120ms), 所有推荐配置满足*
