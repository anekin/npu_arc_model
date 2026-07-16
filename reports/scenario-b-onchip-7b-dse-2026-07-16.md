# 场景 B：On-chip 3D DRAM + 7B 具身智能 DSE 报告

日期：2026-07-16
Arc Model：`v3.8-onchip-bandwidth-cap`
场景：`onchip_7b`
状态：**Raw Exploration PASS；Comparison-ready INFEASIBLE；Product NOT SIGNED**

## 1. 结论

场景 B 的 DSE 数学最优是 **`os_systolic_fused_attention 48×1536, INT2, 1.2 GHz, L2 1 MB`**：
Decode 139.53 TPS、TTFT 175.15 ms、封装占地 100.0 mm²、功耗 46.69 W，满足 TPS≥100、
TTFT≤200 ms、area≤150 mm²。但该 Engine 当前为 M1，融合 overlap、增量 PPA 和 OS 数据流
仍是分析模型；INT2 模型精度也未完成场景 B 验证，因此只能作为研究候选。

同一硬件配置的 **INT4** 点为 116.60 TPS、176.69 ms、100.0 mm²、46.69 W，同样满足硬约束。
本报告建议把 **`os_systolic_fused_attention 48×1536 INT4 @ 1.2 GHz` 作为场景 B 的工程研究基线**；
INT2 保留为性能上界和量化验证分支。当前 PPA 将同一硬件的 INT2/INT4 模式按相同面积和功耗计，
因此两点 PPA 相同；若实现 INT2 专用 packing/datapath，必须另建硬件变体。当前 M2 架构没有可行点，最接近的是
`os_systolic 96×1536 INT2 @ 1.2 GHz`，TPS 132.17 达标，但 TTFT 251.80 ms 超限 25.9%。

## 2. 场景契约

| 项目 | 场景 B 输入 |
|---|---|
| 定位 | 具身智能 / VLM / VLA，实时性优先于低成本 |
| 模型 | Qwen2.5-7B，模型规格 7.62B parameters |
| Workload | 1024-token prompt，128-token output，单请求、decode batch=1，causal attention |
| 精度 | 权重 INT4/INT2 均进入探索；activation INT8；attention/KV FP16 |
| 内存 | 5 GB on-chip 3D DRAM；90% 可用容量为 4.5 GB |
| 带宽 | 额定 500 GB/s；候选有效带宽=`min(logic area×7.5, 500)` GB/s；效率 100% |
| 工艺 | 12 nm |
| 硬约束 | Decode TPS≥100；TTFT≤200 ms；封装面积≤150 mm² |
| 功耗 | 当前没有 `power_w_max`，功耗只参与排序和展示 |
| 排序 | 硬约束过滤后按 `area_mm2, power_w, -tok_s` 字典序 |

TTFT 定义为 1024-token prefill 加首个 decode token；TPS 是 batch=1 的单请求 Decode TPS。

## 3. 搜索空间

- 共 1,710 个候选，1,710 个完成评估，0 个模型异常。
- 阵列：32×1536、48×1536、64×1536、80×1536、96×1536、128×1536。
- 频率：800/1000/1200 MHz；L2：1/2/4/6/8 MB；权重：INT4/INT2。
- 9 类 Engine 映射到该搜索空间；支持 WC 的 Engine 独立搜索 WC OFF/ON，共 13 个硬件变体行。
- paper-faithful `fsa` 未进入本场景表：其因果注意力路径要求 `H=W=head_dim` 的方阵映射，
  与场景 B 的 1536-wide 阵列不兼容；这是 N/A，不是零性能或静默失败。

## 4. 关键候选

| 候选 | 阵列 | 权重 | MHz | L2 KB | Decode TPS | TTFT ms | Area mm² | Power W | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DSE raw best | 48×1536 | INT2 | 1200 | 1024 | 139.53 | 175.15 | 100.0 | 46.69 | PASS / M1 / INT2 精度待验证 |
| 工程研究基线 | 48×1536 | INT4 | 1200 | 1024 | 116.60 | 176.69 | 100.0 | 46.69 | PASS / M1 / 优先校准 |
| M2 最近点 | 96×1536 | INT2 | 1200 | 1024 | 132.17 | 251.80 | 145.7 | 76.53 | FAIL：TTFT 超限 |

42 个候选满足场景硬约束，全部属于 `os_systolic_fused_attention` M1；
Comparison-ready 可行点为 0，Product-qualified 可行点为 0。

## 5. Engine 对比

每个 Engine 显示其最佳可行点；若无可行点，则显示距离硬约束最近的点。

| # | Engine | 状态 | 成熟度 | 阵列 | 权重 | MHz | TPS | TTFT ms | Area mm² | Power W | 失败原因 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `os_systolic_fused_attention` | PASS | M1 | 48×1536 | INT2 | 1200 | 139.53 | 175.2 | 100.0 | 46.7 | 满足硬约束 |
| 2 | `block_fused_attention` | FAIL | M1 | 96×1536 | INT2 | 1200 | 262.47 | 235.3 | 152.6 | 80.3 | TTFT 235.27ms > limit 200.00ms；area 152.60mm2 > limit 150.00mm2 |
| 3 | `os_systolic` | FAIL | M2 | 96×1536 | INT2 | 1200 | 132.17 | 251.8 | 145.7 | 76.5 | TTFT 251.80ms > limit 200.00ms |
| 4 | `block` | FAIL | M2 | 128×1536 | INT2 | 1200 | 252.30 | 327.5 | 184.5 | 97.7 | TTFT 327.48ms > limit 200.00ms；area 184.50mm2 > limit 150.00mm2 |
| 5 | `input_stationary` | FAIL | M1 | 32×1536 | INT2 | 1200 | 8.38 | 494.2 | 100.0 | 34.2 | decode TPS 8.38 < required 100.00；TTFT 494.17ms > limit 200.00ms |
| 6 | `gmma` | FAIL | M1 | 48×1536 | INT4 | 1200 | 5.74 | 608.8 | 131.1 | 68.6 | decode TPS 5.74 < required 100.00；TTFT 608.85ms > limit 200.00ms |
| 7 | `systolic` | FAIL | M2 | 96×1536 | INT2 | 1200 | 10.66 | 860.1 | 100.0 | 49.5 | decode TPS 10.66 < required 100.00；TTFT 860.12ms > limit 200.00ms |
| 8 | `tensor_core` | FAIL | M1 | 48×1536 | INT2 | 1200 | 204.78 | 1224.8 | 100.0 | 44.8 | TTFT 1224.85ms > limit 200.00ms |
| 9 | `wmma` | FAIL | M1 | 32×1536 | INT2 | 1200 | 0.03 | 2315669.0 | 100.0 | 44.8 | decode TPS 0.03 < required 100.00；TTFT 2315668.99ms > limit 200.00ms |

## 6. WC ON/OFF 硬件变体对比

| # | Engine | WC | 状态 | 成熟度 | 阵列 | 权重 | TPS | TTFT ms | Area mm² | Power W | 失败原因 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `os_systolic_fused_attention` | N/A | PASS | M1 | 48×1536 | INT2 | 139.53 | 175.2 | 100.0 | 46.7 | 满足硬约束 |
| 2 | `block_fused_attention` | WC OFF | FAIL | M1 | 96×1536 | INT2 | 262.47 | 235.3 | 152.6 | 80.3 | TTFT 235.27ms > limit 200.00ms；area 152.60mm2 > limit 150.00mm2 |
| 3 | `os_systolic` | N/A | FAIL | M2 | 96×1536 | INT2 | 132.17 | 251.8 | 145.7 | 76.5 | TTFT 251.80ms > limit 200.00ms |
| 4 | `block_fused_attention` | WC ON | FAIL | M1 | 96×1536 | INT2 | 262.47 | 235.3 | 165.0 | 87.1 | TTFT 235.27ms > limit 200.00ms；area 165.00mm2 > limit 150.00mm2 |
| 5 | `block` | WC OFF | FAIL | M2 | 128×1536 | INT2 | 252.30 | 327.5 | 184.5 | 97.7 | TTFT 327.48ms > limit 200.00ms；area 184.50mm2 > limit 150.00mm2 |
| 6 | `block` | WC ON | FAIL | M2 | 96×1536 | INT2 | 252.31 | 379.7 | 157.3 | 82.9 | TTFT 379.66ms > limit 200.00ms；area 157.30mm2 > limit 150.00mm2 |
| 7 | `input_stationary` | N/A | FAIL | M1 | 32×1536 | INT2 | 8.38 | 494.2 | 100.0 | 34.2 | decode TPS 8.38 < required 100.00；TTFT 494.17ms > limit 200.00ms |
| 8 | `gmma` | WC OFF | FAIL | M1 | 48×1536 | INT4 | 5.74 | 608.8 | 131.1 | 68.6 | decode TPS 5.74 < required 100.00；TTFT 608.85ms > limit 200.00ms |
| 9 | `gmma` | WC ON | FAIL | M1 | 48×1536 | INT4 | 5.74 | 608.8 | 136.2 | 71.4 | decode TPS 5.74 < required 100.00；TTFT 608.85ms > limit 200.00ms |
| 10 | `systolic` | WC ON | FAIL | M2 | 96×1536 | INT2 | 10.66 | 860.1 | 100.0 | 49.5 | decode TPS 10.66 < required 100.00；TTFT 860.12ms > limit 200.00ms |
| 11 | `tensor_core` | N/A | FAIL | M1 | 48×1536 | INT2 | 204.78 | 1224.8 | 100.0 | 44.8 | TTFT 1224.85ms > limit 200.00ms |
| 12 | `systolic` | WC OFF | FAIL | M2 | 96×1536 | INT2 | 10.36 | 1263.6 | 100.0 | 44.8 | decode TPS 10.36 < required 100.00；TTFT 1263.57ms > limit 200.00ms |
| 13 | `wmma` | N/A | FAIL | M1 | 32×1536 | INT2 | 0.03 | 2315669.0 | 100.0 | 44.8 | decode TPS 0.03 < required 100.00；TTFT 2315668.99ms > limit 200.00ms |

在场景 B 的高带宽点，Block/BFSA 的 WC ON 性能已被 500 GB/s memory ceiling 吞没，
但面积和功耗仍增加；因此这些代表点不应选择 WC ON。Systolic WC ON 虽改善调度，
整体 TPS/TTFT 仍远离场景要求。

## 7. 物理与容量复核

- ARC-BUG-007 修复前，面积耦合会把 500 GB/s 错误放大到 682.5–1383.75 GB/s；
  首次运行已判无效并被本次结果覆盖。修复提交为 `38409d8`。
- 修复后所有候选带宽均≤500 GB/s；已序列化审计的 1688 个点中，
  完整模型 memory ceiling 越界数为 0。
- INT4 权重为 3.81 GB，理论 Decode ceiling 为 131.23 TPS；工程基线为 116.60 TPS。
- INT2 权重为 1.905 GB，理论 Decode ceiling 为 262.47 TPS；raw best 为 139.53 TPS。
- INT4 部署需要 4.1321 GB，可用 4.5 GB，余量只有 0.3679 GB；INT2 需要 2.2271 GB，
  余量 2.2729 GB。INT4 容量可行但偏紧，扩上下文或并发前必须重算。

## 8. 为什么当前只有 fused-attention 候选达标

场景 B 的主要难点是 1024-token prefill 的 200 ms TTFT，而不只是 Decode memory ceiling。
`os_systolic_fused_attention` 保留 OS projection/FFN 数据流，同时把 QK、Softmax、PV 的
attention 路径融合并建模 overlap，因此 48×1536 已能跨过 TTFT 门槛。基线 OS 虽然 TPS 达标，
但 prefill 路径使 TTFT 停在 251.8 ms。Block/BFSA 的 Decode TPS 更高，却因为 TTFT 和面积
同时超限而失败。这说明场景 B 下一步应优先校准和实现融合 attention，而不是继续堆大阵列。

这个优势目前仍有 35% performance/area、40% power 级别的不确定度标签，且缺少 native RTL trace；
所以“通过硬约束”只表示模型内可行，不等于完成 Engine admission 或产品 Signoff。

## 9. 决策与后续工作

1. 以 `OFSA 48×1536 INT4 @1.2GHz, L2 1MB` 作为场景 B 的研究基线；不放宽 200 ms TTFT。
2. 为 fused attention 建立独立周期 trace、控制/缓冲 PPA 和 RTL/综合证据，把 M1 推进到 M2。
3. 单独验证 Qwen2.5-7B/VLM/VLA 的 INT2 任务精度；通过前不把 INT2 raw best 作为工程默认。
4. 补充场景 B 的功耗/热设计硬上限；当前 46.69 W 只能展示，不能判定产品可接受。
5. 做 48×1536 与 64×1536 的局部敏感性和 guardband；M1 误差收敛前不冻结最终阵列。
6. 继续把 WC OFF/ON 作为不同硬件实现；场景 B 当前代表点不支持为 WC 付出额外 PPA。

## 10. 复现与证据

```powershell
.venv\Scripts\python.exe sim\design_space_explorer.py `
  --scenario onchip_7b --top 20 `
  --output results\onchip_7b_latest_2026-07-16.json

.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe sim\engine_audit.py
```

- 完整结果：`sim/results/onchip_7b_latest_2026-07-16.json`
- 结构化摘要：`sim/results/onchip_7b_dse_summary_2026-07-16.json`
- 完整结果 SHA256：`4bf90cf450f6260dd4bdad33cc24952e52b3bd1664fb53aa2ced3da4efeab5b6`
- 测试：126 passed；Engine audit：10/10 passed。
