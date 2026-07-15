# LPDDR5 应用场景 DSE 报告 — Arc Model v3.6

**运行日期：** 2026-07-15  
**Arc Model：** `v3.6-engine-evidence`  
**范围：** 场景 A 低成本端侧算力扩展，以及 30K Prefix Agent 子场景  
**内存：** 64-bit LPDDR5-6400，物理带宽 51.2 GB/s，覆盖 75%/85%/90% 有效带宽效率角  
**模型：** Qwen2.5-3B，INT4 权重，INT8 激活，FP16 Attention/KV  

## 1. 结论摘要

85% 名义搜索共评估 **1965 个配置，1965 valid、0 invalid**；其中 771 个满足
场景硬约束，343 个属于 M2 comparison-ready，没有 M3/M4 product-qualified
配置。

- 名义成本目标最优是 **OS Systolic 32×64 @1 GHz，L2=1 MB**：20.03 TPS、
  Prefill 667.2 tok/s、TTFT 241.8 ms、面积 42.8 mm²、功耗 8.8 W。
- 这是 **M2 架构比较结论**，不是产品 Signoff。它相对 20 TPS 下限只有约
  **0.15%** 裕量，明显小于 OS 模型声明的不确定度。
- **Block 64×64 @800 MHz** 只增加 1.5 mm²，名义功耗同为 8.8 W，但名义
  Decode 为 28.05 TPS，75% 保守带宽角仍有 24.75 TPS。因此从风险调整后的
  工程角度，Block 是更稳健的下一阶段校准候选。
- Agent 子场景在 75%/85%/90% 三个角均为 **INFEASIBLE**。名义最好的 M1
  研究点也只有 16.22 TPS，没有任何 M2 架构同时满足 TPS 和 TTFT。

建议：保留 **OS 32×64@1GHz** 作为名义最低成本点，同时将
**Block 64×64@800MHz** 作为风险调整后的 shortlist 主候选。在 Block/OS
完成 M3 校准并把 guardband 纳入 product-qualified 判定之前，不冻结产品架构。

## 2. 应用需求

| 项目 | 场景 A | Agent 子场景 |
|---|---|---|
| 工作负载 | 128-token Prompt + 128-token Output | 30,000 cached prefix + 875 append + 214 output |
| Decode batch / 并发 | 1 / 1 | 1 / 1 |
| 精度 | INT4 权重、INT8 激活、FP16 Attention/KV | 相同 |
| Decode TPS 硬下限 | 20 tok/s | 20 tok/s |
| TTFT 目标 / 硬上限 | 500 / 1000 ms | 2000 / 5000 ms |
| Prefill 目标 | 无 | ≥500 tok/s |
| 面积硬上限 | 80 mm² | 80 mm² |
| 容量 | 未指定，容量可行性未 Signoff | 4 GB，90% 可用 |
| 排序目标 | 面积、功耗、TPS | 通过硬约束后按面积、功耗、TPS |

Agent 目标仍属于待本地产品 trace 校准的假设。基础场景没有明确内存容量，虽然
当前推荐点估算占用约 1.81 GB，但容量约束仍是产品 Signoff 缺口。

## 3. 搜索完整性和成熟度

| 搜索 | 配置数 | Valid | Invalid | Raw feasible | M2 feasible | M3/M4 feasible |
|---|---:|---:|---:|---:|---:|---:|
| 场景 A，85% 名义角 | 1965 | 1965 | 0 | 771 | 343 | 0 |
| Agent，85% 名义角 | 1980 | 1980 | 0 | 0 | 0 | 0 |

10 个已登记 Engine 全部进入搜索。Systolic、OS Systolic 和 Block 当前为 M2，
可以进入正式架构横向比较；其余 7 个 Engine 为 M1，只进入 Raw Exploration。
当前没有 M3/M4 Engine。

## 4. 场景 A：全部 Engine 名义结果

| 排名 | Engine | Maturity | 结果/目标 | 最佳配置 | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `os_systolic` | M2 | PASS/MET | 32×64@1000MHz，L2=1MB | 20.03 | 667.2 | 241.8ms | 42.8mm² | 8.8W |
| 2 | `os_systolic_fused_attention` | M1 | PASS/MET | 32×64@1000MHz，L2=1MB | 20.06 | 674.8 | 239.6ms | 42.9mm² | 8.9W |
| 3 | `block` | M2 | PASS/MET | 64×64@800MHz+WC，L2=1MB | 28.05 | 286.4 | 482.7ms | 44.3mm² | 8.8W |
| 4 | `block_fused_attention` | M1 | PASS/MET | 64×64@800MHz+WC，L2=1MB | 28.08 | 288.1 | 479.9ms | 44.4mm² | 8.9W |
| 5 | `systolic` | M2 | PASS/MET | 128×128@1200MHz+WC，L2=1MB | 20.75 | 1433.4 | 137.5ms | 47.2mm² | 12.1W |
| 6 | `fsa` | M1 | PASS/MET | 128×128@1000MHz，L2=1MB | 20.46 | 1718.2 | 123.4ms | 47.9mm² | 11.4W |
| 7 | `gmma` | M1 | PASS/MET | 128×128@1000MHz，L2=1MB | 20.51 | 1673.7 | 125.2ms | 61.9mm² | 18.4W |
| 8 | `tensor_core` | M1 | PASS/MISS | 64×64@1200MHz，L2=1MB | 24.96 | 165.2 | 814.8ms | 44.3mm² | 10.3W |
| 9 | `input_stationary` | M1 | FAIL/MET | 32×64@1200MHz，L2=1MB | 10.60 | 472.6 | 365.2ms | 42.8mm² | 9.4W |
| 10 | `wmma` | M1 | FAIL/MISS | 96×96@1200MHz，L2=1MB | 0.06 | 1.1 | 137264.6ms | 51.2mm² | 14.5W |

`PASS` 表示所列点满足硬约束，`MET/MISS` 表示是否达到软目标。Maturity 与
原始数值排名相互独立，M1 的名义性能优势不能直接变成正式架构推荐。

## 5. 场景 A：带宽效率角

| DRAM 效率 | 有效带宽 | Raw feasible | M2 feasible | 名义目标最优 | OS TPS | Block TPS | Systolic TPS |
|---:|---:|---:|---:|---|---:|---:|---:|
| 75% | 38.40 GB/s | 762 | 334 | `os_systolic` 32×64@1GHz | 20.03 | 24.75 | 20.50 |
| 85% | 43.52 GB/s | 771 | 343 | `os_systolic` 32×64@1GHz | 20.03 | 28.05 | 20.75 |
| 90% | 46.08 GB/s | 771 | 343 | `os_systolic` 32×64@1GHz | 20.03 | 29.69 | 21.21 |

OS 推荐点在三档效率下不变，是因为这个小阵列在约 20 TPS 处主要受自身
compute/dataflow 限制；它对带宽角不敏感，不代表对模型误差也稳健。0.15% 的
TPS 裕量不足以覆盖当前分析模型的不确定度。

Block 更直接地跟随 LPDDR 带宽变化，但在 75% 保守角仍保留约 23.8% TPS
裕量。Systolic 的 Prefill 和 TTFT 更好，但面积和功耗更高，不符合场景 A 的
低成本优先定位。

## 6. 场景 A shortlist

| 定位 | 架构 | Decode TPS | Prefill TPS | TTFT | Area | Power | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| 名义最低成本 | OS Systolic 32×64@1GHz | 20.03 | 667.2 | 241.8ms | 42.8mm² | 8.8W | 面积最低，但 TPS 几乎没有 guardband |
| 风险调整后主候选 | Block 64×64@800MHz | 28.05 | 286.4 | 482.7ms | 44.3mm² | 8.8W | 只增加 1.5mm²，保守角 TPS 裕量明显更好 |
| 时延校准参考 | Systolic 128×128@1.2GHz | 20.75 | 1433.4 | 137.5ms | 47.2mm² | 12.1W | TTFT 更好，但成本匹配较差 |

因此，本轮不能简单地把 OS 作为唯一推荐：OS 是按当前名义目标函数得到的最优，
Block 则是考虑模型不确定度和带宽保守角后的更稳健候选。下一阶段应同时保留两者
进行 Func/RTL trace 校准。

## 7. Agent 子场景：全部 Engine 名义结果

| 排名 | Engine | Maturity | 结果/目标 | 最佳配置 | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `block_fused_attention` | M1 | FAIL/MISS | 64×64@1200MHz，L2=1MB | 16.22 | 179.1 | 4946.8ms | 44.4mm² | 10.4W |
| 2 | `os_systolic_fused_attention` | M1 | FAIL/MET | 128×256@1200MHz，L2=1MB | 14.45 | 2329.4 | 444.8ms | 66.2mm² | 23.5W |
| 3 | `fsa` | M1 | FAIL/MET | 128×256@1200MHz，L2=1MB | 10.39 | 984.3 | 985.2ms | 54.5mm² | 16.4W |
| 4 | `block` | M2 | FAIL/MISS | 128×256@1200MHz+WC，L2=1MB | 14.75 | 141.9 | 6232.2ms | 64.8mm² | 22.7W |
| 5 | `os_systolic` | M2 | FAIL/MISS | 128×256@1200MHz，L2=1MB | 12.69 | 153.5 | 5777.6ms | 64.8mm² | 22.7W |
| 6 | `gmma` | M1 | FAIL/MISS | 192×128@1200MHz，L2=1MB | 9.64 | 154.5 | 5766.2ms | 72.2mm² | 27.1W |
| 7 | `systolic` | M2 | FAIL/MISS | 256×256@1200MHz+WC，L2=1MB | 7.60 | 146.1 | 6120.1ms | 64.8mm² | 22.7W |
| 8 | `tensor_core` | M1 | FAIL/MISS | 128×256@1200MHz，L2=1MB | 13.85 | 81.7 | 10781.9ms | 64.8mm² | 22.7W |
| 9 | `input_stationary` | M1 | FAIL/MISS | 64×256@1200MHz，L2=1MB | 0.76 | 139.8 | 7578.4ms | 53.1mm² | 15.6W |
| 10 | `wmma` | M1 | FAIL/MISS | 32×32@1200MHz，L2=1MB | 0.02 | 0.5 | 1958110.6ms | 42.4mm² | 9.2W |

所有 Engine 至少违反一个硬约束。BFSA/OFSA/FSA 显示了 Attention 优化方向，
但仍为 M1，并且没有解决完整模型的 Decode 瓶颈。名义角距离约束最近的 M2 点是
Block 128×256@1.2GHz：14.75 TPS、TTFT 6232.2 ms，TPS 和 TTFT 均失败。

## 8. Agent：带宽效率角

| DRAM 效率 | 有效带宽 | 可行性 | 最好 Raw 研究点 | Raw TPS / TTFT | 最近 M2 点 | M2 TPS / TTFT |
|---:|---:|---|---|---:|---|---:|
| 75% | 38.40 GB/s | 不可行 | BFSA 64×64@1.2GHz | 14.31 / 4955.6ms | OS 128×256@1.2GHz | 11.97 / 5783.8ms |
| 85% | 43.52 GB/s | 不可行 | BFSA 64×64@1.2GHz | 16.22 / 4946.8ms | Block 128×256@1.2GHz | 14.75 / 6232.2ms |
| 90% | 46.08 GB/s | 不可行 | BFSA 64×64@1.2GHz | 17.17 / 4943.2ms | Block 128×256@1.2GHz | 15.54 / 6228.5ms |

有效带宽从 38.4 提升到 46.08 GB/s 后仍没有可行 Agent 架构。后续需要考虑增加
内存通道/带宽、缩小活跃上下文、改变 KV/Prefix cache 策略、放宽 TPS/TTFT，
或者引入经校准且能够产生实质性提升的新架构。

## 9. Signoff 结论

| Gate | 结果 | 说明 |
|---|---|---|
| 搜索完整性 | PASS | 六次完整搜索全部 valid，10 个 Engine 全覆盖 |
| 带宽角执行 | PASS | 75%/85%/90% 独立运行，没有混入名义排序 |
| 场景 A 架构探索 | PASS | 存在稳定的 M2 shortlist |
| 场景 A 产品推荐 | NOT SIGNED | 无 M3/M4；OS guardband 不足；容量未定义 |
| Agent 架构探索 | INFEASIBLE | 三个效率角均为 0 raw/M2 feasible |
| PPA 产品使用 | NOT SIGNED | 面积和功耗仍是架构阶段分析估算 |

v3.6 测试阶段修复的四个缺陷及其回归测试已由提交 `2897dff` 合入，
并在 `docs/bug-tracker.md` 更新为 `CLOSED`。本轮六次 DSE 没有发现新的
invalid config 或物理 ceiling 失败。

## 10. 结果文件和复现

- 场景 A 名义完整 JSON：`sim/results/lpddr5_3b_latest_2026-07-15.json`
- Agent 名义完整 JSON：`sim/results/lpddr5_3b_agent_latest_2026-07-15.json`
- 六次运行摘要、精确场景定义和源文件 SHA256：
  `sim/results/lpddr5_latest_dse_summary_2026-07-15.json`
- Engine 证据：`sim/config/engine_manifests.yaml`
- Bug 状态：`docs/bug-tracker.md`

名义搜索命令：

```powershell
.venv\Scripts\python.exe sim\design_space_explorer.py --scenario lpddr5_3b --top 50 --output results\lpddr5_3b_latest_2026-07-15.json
.venv\Scripts\python.exe sim\design_space_explorer.py --scenario lpddr5_3b_agent --top 50 --output results\lpddr5_3b_agent_latest_2026-07-15.json
```

75%/90% 角的完整 `scenario_definition` 已保存在摘要 JSON 中；相对名义场景只修改
`memory.dram_efficiency` 和 `memory.effective_bw_gbps`。
