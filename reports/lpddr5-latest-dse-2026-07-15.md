# LPDDR5 应用场景 DSE 报告 — Arc Model v3.7

**运行日期：** 2026-07-15
**Arc Model：** `v3.7-weight-cache-variants`
**范围：** 场景 A 低成本端侧算力扩展，以及 30K Prefix Agent 子场景
**内存：** 64-bit LPDDR5-6400，物理带宽 51.2 GB/s，覆盖 75%/85%/90% 有效带宽效率角
**模型：** Qwen2.5-3B，INT4 权重、INT8 激活、FP16 Attention/KV

## 1. 结论摘要

- 场景 A 的 85% 名义搜索评估 1965 个配置，全部 valid；756 个满足应用硬约束，334 个为 M2 comparison-ready，没有 M3/M4 product-qualified 配置。
- 按低成本排序，名义最优仍为 **OS Systolic 32×64@1GHz、L2=1MB**：20.03 TPS、TTFT 241.8 ms、42.8 mm²、8.84 W。其 TPS 仅比 20 TPS 下限高约 0.15%，仍需保留风险说明。
- **Block 64×64@800MHz 的代表点现在是 WC OFF**：28.04 TPS、44.3 mm²、8.84 W。WC ON 为 28.05 TPS、44.6 mm²、8.96 W；在该强带宽瓶颈点，WC 性能收益几乎被 memory ceiling 吞没，因此不应为它支付硬件成本。
- WS Systolic 对 WC 更敏感。同为 128×128@1.2GHz、L2=1MB，WC OFF 为 16.42 TPS / 47.2 mm² / 12.07 W，WC ON 为 20.75 TPS / 48.1 mm² / 12.60 W。WC 是否值得引入必须按 Engine、阵列和工作负载判断。
- Agent 子场景在三个带宽效率角仍全部 **INFEASIBLE**；WC 没有改变该结论。

## 2. WC ON/OFF 建模与报告契约

`weight_cache` 现在是独立硬件实现变体，不再只是性能模型中的免费开关：

- DSE 搜索对 `systolic`、`block`、`block_fused_attention`、`gmma` 分别生成 WC OFF 和 WC ON；不支持该机制的 Engine 显示 `N/A`。
- WC ON 在 PE array 上计入额外本地 weight 存储/寄存器和选择控制逻辑，并由同一增量面积推导逻辑功耗。
- 当前 PE 面积代理为：WS Systolic +15%，Block/BFSA +10%，GMMA +5%。百分比作用于 PE array，不是整芯片面积。
- 这些参数用于架构阶段排序；在寄存器/SRAM 实现、时钟活动、布局和 leakage 完成独立校准前，不能视为 M3 或产品 PPA signoff 数据。
- WC 调度保留两个独立 GEMM 的合法 fallback，保证启用硬件后同一 shape 不会发生周期回退。
- JSON 保留向后兼容的 `engine_comparison`，并新增 `engine_variant_comparison`，后者是查看硬件实现差异的正式入口。

## 3. 应用需求

| 项目 | 场景 A | Agent 子场景 |
|---|---|---|
| 工作负载 | 128-token Prompt + 128-token Output | 30,000 cached prefix + 875 append + 214 output |
| Decode batch / 并发 | 1 / 1 | 1 / 1 |
| 精度 | INT4 权重、INT8 激活、FP16 Attention/KV | 相同 |
| Decode TPS 硬下限 | 20 tok/s | 20 tok/s |
| TTFT 目标 / 硬上限 | 500 / 1000 ms | 2000 / 5000 ms |
| Prefill 目标 | 无 | ≥200 tok/s |
| 面积硬上限 | 80 mm² | 80 mm² |
| 排序目标 | 面积、功耗、TPS | 通过硬约束后按面积、功耗、TPS |

## 4. 搜索完整性

| 搜索 | 配置数 | Valid | Invalid | Raw feasible | M2 feasible | M3/M4 feasible |
|---|---:|---:|---:|---:|---:|---:|
| 场景 A，85% 名义角 | 1965 | 1965 | 0 | 756 | 334 | 0 |
| Agent，85% 名义角 | 1980 | 1980 | 0 | 0 | 0 | 0 |

## 5. 场景 A：全部 Engine 硬件变体

| 排名 | Engine | WC | Maturity | 结果/目标 | 配置 | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `os_systolic` | **N/A** | M2 | PASS/MET | 32×64@1GHz, L2=1MB | 20.03 | 667.2 | 241.8 ms | 42.8 mm² | 8.84 W |
| 2 | `os_systolic_fused_attention` | **N/A** | M1 | PASS/MET | 32×64@1GHz, L2=1MB | 20.06 | 674.8 | 239.6 ms | 42.9 mm² | 8.89 W |
| 3 | `block` | **WC OFF** | M2 | PASS/MET | 64×64@0.8GHz, L2=1MB | 28.04 | 286.4 | 482.7 ms | 44.3 mm² | 8.84 W |
| 4 | `block_fused_attention` | **WC OFF** | M1 | PASS/MET | 64×64@0.8GHz, L2=1MB | 28.07 | 288.1 | 479.9 ms | 44.4 mm² | 8.91 W |
| 5 | `block` | **WC ON** | M2 | PASS/MET | 64×64@0.8GHz, L2=1MB | 28.05 | 286.4 | 482.7 ms | 44.6 mm² | 8.96 W |
| 6 | `block_fused_attention` | **WC ON** | M1 | PASS/MET | 64×64@0.8GHz, L2=1MB | 28.08 | 288.1 | 479.9 ms | 44.7 mm² | 9.04 W |
| 7 | `fsa` | **N/A** | M1 | PASS/MET | 128×128@1GHz, L2=1MB | 20.46 | 1718.2 | 123.4 ms | 47.9 mm² | 11.40 W |
| 8 | `systolic` | **WC ON** | M2 | PASS/MET | 128×128@1.2GHz, L2=1MB | 20.75 | 1433.4 | 137.5 ms | 48.1 mm² | 12.60 W |
| 9 | `systolic` | **WC OFF** | M2 | PASS/MET | 128×256@1GHz, L2=1MB | 20.38 | 1698.5 | 124.4 ms | 53.1 mm² | 13.98 W |
| 10 | `gmma` | **WC OFF** | M1 | PASS/MET | 128×128@1GHz, L2=1MB | 20.51 | 1673.7 | 125.2 ms | 61.9 mm² | 18.39 W |
| 11 | `gmma` | **WC ON** | M1 | PASS/MET | 128×128@1GHz, L2=1MB | 20.51 | 1673.7 | 125.2 ms | 62.9 mm² | 18.91 W |
| 12 | `tensor_core` | **N/A** | M1 | PASS/MISS | 64×64@1.2GHz, L2=1MB | 24.96 | 165.2 | 814.8 ms | 44.3 mm² | 10.31 W |
| 13 | `input_stationary` | **N/A** | M1 | FAIL/MET | 32×64@1.2GHz, L2=1MB | 10.60 | 472.6 | 365.2 ms | 42.8 mm² | 9.43 W |
| 14 | `wmma` | **N/A** | M1 | FAIL/MISS | 96×96@1.2GHz, L2=1MB | 0.06 | 1.1 | 137264.6 ms | 51.2 mm² | 14.50 W |

`PASS` 表示该行候选满足硬约束；`MET/MISS` 表示是否达到软目标。M1 只进入 Raw Exploration。WC ON/OFF 是分别排序的硬件候选，不再被合并成一个 Engine 行。

## 6. 场景 A：同配置 WC 代价与收益

| Engine / 同配置 | WC | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---|---|---:|---:|---:|---:|---:|
| `systolic` 128×128@1.2GHz, L2=1MB | **OFF** | 16.42 | 1190.8 | 168.4 ms | 47.2 mm² | 12.07 W |
| `systolic` 128×128@1.2GHz, L2=1MB | **ON** | 20.75 | 1433.4 | 137.5 ms | 48.1 mm² | 12.60 W |
| `block` 64×64@0.8GHz, L2=1MB | **OFF** | 28.04 | 286.4 | 482.7 ms | 44.3 mm² | 8.84 W |
| `block` 64×64@0.8GHz, L2=1MB | **ON** | 28.05 | 286.4 | 482.7 ms | 44.6 mm² | 8.96 W |
| `block_fused_attention` 64×64@0.8GHz, L2=1MB | **OFF** | 28.07 | 288.1 | 479.9 ms | 44.4 mm² | 8.91 W |
| `block_fused_attention` 64×64@0.8GHz, L2=1MB | **ON** | 28.08 | 288.1 | 479.9 ms | 44.7 mm² | 9.04 W |
| `gmma` 128×128@1GHz, L2=1MB | **OFF** | 20.51 | 1673.7 | 125.2 ms | 61.9 mm² | 18.39 W |
| `gmma` 128×128@1GHz, L2=1MB | **ON** | 20.51 | 1673.7 | 125.2 ms | 62.9 mm² | 18.91 W |

这张表固定 Engine、阵列、频率、L2、精度与内存，只切换 WC。Block/BFSA/GMMA 在当前 LPDDR 场景中已接近带宽瓶颈，WC 对 TPS 的边际贡献很小；WS Systolic 的双 weight 路径则能明显减少 FFN gate/up 的阵列开销，因此收益更大。

## 7. 场景 A：带宽效率角

| LPDDR 效率 | 有效带宽 | Raw feasible | M2 feasible | 名义最优 | Block WC OFF TPS | Block WC ON TPS | Systolic WC OFF/ON TPS |
|---:|---:|---:|---:|---|---:|---:|---:|
| 75% | 38.40 GB/s | 747 | 325 | `os_s 32x64 INT4 1000MHz B1 LPDDR5-64b` | 24.75 | 24.75 | 20.38 / 20.50 |
| 85% | 43.52 GB/s | 756 | 334 | `os_s 32x64 INT4 1000MHz B1 LPDDR5-64b` | 28.04 | 28.05 | 20.38 / 20.75 |
| 90% | 46.08 GB/s | 756 | 334 | `os_s 32x64 INT4 1000MHz B1 LPDDR5-64b` | 29.69 | 29.69 | 20.39 / 21.21 |

三个效率角的成本最优均为 OS Systolic 32×64@1GHz。Block WC OFF 在 75% 角仍有 24.75 TPS，且比 WC ON 成本低，因此继续作为风险调整后的主 shortlist。Systolic WC ON 能满足 20 TPS，但其成本高于 OS/Block。

## 8. Agent：全部 Engine 硬件变体

| 排名 | Engine | WC | Maturity | 结果/目标 | 配置 | Decode TPS | Prefill TPS | TTFT | Area | Power |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `block_fused_attention` | **WC OFF** | M1 | FAIL/MISS | 64×64@1.2GHz, L2=1MB | 16.22 | 179.1 | 4946.8 ms | 44.4 mm² | 10.42 W |
| 2 | `block_fused_attention` | **WC ON** | M1 | FAIL/MISS | 64×64@1.2GHz, L2=1MB | 16.22 | 179.1 | 4946.8 ms | 44.7 mm² | 10.60 W |
| 3 | `os_systolic_fused_attention` | **N/A** | M1 | FAIL/MET | 128×256@1.2GHz, L2=1MB | 14.45 | 2329.4 | 444.8 ms | 66.2 mm² | 23.50 W |
| 4 | `fsa` | **N/A** | M1 | FAIL/MET | 128×256@1.2GHz, L2=1MB | 10.39 | 984.3 | 985.2 ms | 54.5 mm² | 16.45 W |
| 5 | `block` | **WC ON** | M2 | FAIL/MISS | 128×256@1.2GHz, L2=1MB | 14.75 | 141.9 | 6232.2 ms | 67.2 mm² | 24.06 W |
| 6 | `block` | **WC OFF** | M2 | FAIL/MISS | 128×256@1.2GHz, L2=1MB | 14.75 | 141.9 | 6232.2 ms | 64.8 mm² | 22.65 W |
| 7 | `os_systolic` | **N/A** | M2 | FAIL/MISS | 128×256@1.2GHz, L2=1MB | 12.69 | 153.5 | 5777.6 ms | 64.8 mm² | 22.65 W |
| 8 | `gmma` | **WC OFF** | M1 | FAIL/MISS | 192×128@1.2GHz, L2=1MB | 9.64 | 154.5 | 5766.2 ms | 72.2 mm² | 27.06 W |
| 9 | `gmma` | **WC ON** | M1 | FAIL/MISS | 192×128@1.2GHz, L2=1MB | 9.64 | 154.5 | 5766.2 ms | 73.7 mm² | 27.99 W |
| 10 | `systolic` | **WC ON** | M2 | FAIL/MISS | 256×256@1.2GHz, L2=1MB | 7.61 | 146.1 | 6120.0 ms | 68.4 mm² | 24.77 W |
| 11 | `systolic` | **WC OFF** | M2 | FAIL/MISS | 256×256@1.2GHz, L2=1MB | 7.61 | 144.1 | 6204.7 ms | 64.8 mm² | 22.65 W |
| 12 | `tensor_core` | **N/A** | M1 | FAIL/MISS | 128×256@1.2GHz, L2=1MB | 13.85 | 81.7 | 10781.9 ms | 64.8 mm² | 22.65 W |
| 13 | `input_stationary` | **N/A** | M1 | FAIL/MISS | 64×256@1.2GHz, L2=1MB | 0.76 | 139.8 | 7578.4 ms | 53.1 mm² | 15.60 W |
| 14 | `wmma` | **N/A** | M1 | FAIL/MISS | 32×32@1.2GHz, L2=1MB | 0.02 | 0.5 | 1958110.6 ms | 42.4 mm² | 9.21 W |

所有变体至少违反一项硬约束。Block 的 WC ON/OFF 在 128×256@1.2GHz 上得到相同的 14.75 TPS（显示精度内），但 WC ON 增加约 2.4 mm² 和 1.41 W，说明该点没有采用 WC ON 的架构理由。BFSA 的 WC ON 同样没有带来可见 TPS 收益。

## 9. Agent：带宽效率角

| LPDDR 效率 | 有效带宽 | 可行性 | BFSA WC OFF/ON TPS | Block WC OFF/ON TPS | Systolic WC OFF/ON TPS |
|---:|---:|---|---:|---:|---:|
| 75% | 38.40 GB/s | INFEASIBLE | 14.31 / 14.31 | 13.16 / 13.16 | 7.35 / 7.35 |
| 85% | 43.52 GB/s | INFEASIBLE | 16.22 / 16.22 | 14.75 / 14.75 | 7.61 / 7.61 |
| 90% | 46.08 GB/s | INFEASIBLE | 17.17 / 17.17 | 15.54 / 15.54 | 7.73 / 7.73 |

有效带宽从 38.40 提升到 46.08 GB/s 后仍无可行 Agent 架构。后续需要从带宽/通道、Prefix/KV 策略、工作负载约束或经校准的新 Engine 入手，不能依赖 WC 单点优化解决。

## 10. Signoff 状态

| Gate | 结果 | 说明 |
|---|---|---|
| WC 变体搜索与展示 | PASS | WC ON/OFF 独立进入搜索，JSON/CLI/Markdown 均分别展示 |
| WC PPA 非零成本 | PASS | 面积与功耗共享同一 WC-adjusted PE area；定向回归覆盖 4 个支持 Engine |
| 自动化回归 | PASS | 125 tests passed |
| 场景 A 架构探索 | PASS | 存在 M2 shortlist |
| 场景 A 产品推荐 | NOT SIGNED | 无 M3/M4；OS guardband 不足；容量仍未完成产品定义 |
| Agent 架构探索 | INFEASIBLE | 75%/85%/90% 均无 raw/M2 feasible 配置 |
| WC 产品 PPA | NOT SIGNED | 当前增量为架构阶段 proxy，尚无独立 RTL/综合/布局校准 |

本轮登记 `ARC-BUG-005`：旧版本 WC 性能收益未计硬件成本，且报告按 Engine 合并了 WC ON/OFF。修复已通过全量回归和六次 DSE，当前状态为 `VERIFIED`；提交合入后再更新为 `CLOSED`。 同时登记 `ARC-BUG-006`：Systolic WC pair 缺少两次独立 GEMM fallback；修复后 Agent FFN shape 满足 WC 单调性。

## 11. 结果文件与复现

- 场景 A 名义完整 JSON：`sim/results/lpddr5_3b_latest_2026-07-15.json`
- Agent 名义完整 JSON：`sim/results/lpddr5_3b_agent_latest_2026-07-15.json`
- 六次运行摘要、完整场景定义、Engine 与硬件变体表、源文件 SHA256：`sim/results/lpddr5_latest_dse_summary_2026-07-15.json`
- WC PPA 参数来源与局限：`references/area_sources.md`
- Bug 状态：`docs/bug-tracker.md`

名义搜索命令：

```powershell
.venv\Scripts\python.exe sim\design_space_explorer.py --scenario lpddr5_3b --top 50 --output results\lpddr5_3b_latest_2026-07-15.json
.venv\Scripts\python.exe sim\design_space_explorer.py --scenario lpddr5_3b_agent --top 50 --output results\lpddr5_3b_agent_latest_2026-07-15.json
```
