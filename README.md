# NPU Arc Model

CaduceusCore 的架构设计与空间搜索（DSE）独立仓库。

> Arc Model 回答「选什么架构」，CaduceusCore 回答「怎么实现」。

---

## 一、设计方法学

### 1.1 三模型体系

```
┌─────────────────────────────────────────────────────────────────┐
│                      NPU 设计验证三模型                            │
│                                                                  │
│  Arc Model (本仓库)           Func Model            RTL           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │ 精度门 + DSE     │ → │ bit-exact 仿真    │ → │ Verilog     │  │
│  │ 量化验证          │    │ MMIO + 固件调度    │    │ 综合 + 布局  │  │
│  │ 8 引擎 PPA       │    │ tile scheduler   │    │             │  │
│  │ 带宽/面积建模     │    │ E2E llama.cpp    │    │             │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
│       ↑                        ↑                     ↑          │
│   回答「对不对」           回答「多准」          回答「多快多小」   │
└─────────────────────────────────────────────────────────────────┘
```

详见 [`docs/arc_vs_func.md`](docs/arc_vs_func.md)。

### 1.2 DSE 四阶段方法论

```
Phase -1: 需求澄清    → 检查关键参数（seq_len/TTFT/model/memory/process）
                        绝不允许静默使用默认值

Phase 0:  预检分析    → 瓶颈预测（BW vs compute ceiling）
                        TTFT 约束覆盖 BW 瓶颈推荐
                        组件清单验证（PHY/PCIe/TSV）

Phase 1:  扫描+敏感度 → 全维度参数扫描
                        自动敏感度分析（ΔTPS%, ΔArea%）
                        零敏感度参数标记为降成本候选

Phase 2:  交叉校验    → 对比已知产品（TPUv1/RK1828/Eyeriss）
                        异常检测（>2× 偏差标记）
```

入口脚本及详细说明见 [`sim/dse_scenario.py`](sim/dse_scenario.py)。

### 1.3 关键技术决策

| 决策 | 结论 | 依据 |
|:---|:---|:---|
| **PE 面积校准** | TPUv1 ISCA 2017 die-shot 为主锚点 | [`references/area_sources.md`](references/area_sources.md) |
| **DRAM 效率** | 0.85 — conservative sequential decode baseline (per-bank refresh ~3.6%, extra from controller scheduling/command bus/bank conflicts) | 详见 contracts/hardware.py MemoryConfig |
| **量化方案** | per-block (g=128) INT4 | cos_sim > 0.99，比 per-channel 稳定 0.014 |
| **SRAM 灵敏度** | LPDDR5 场景：4-6MB sweet spot（**探索性结论，待 T2+ 证据**）；3D DRAM 场景：512KB 足够（**理想驻留假设**） | DSE 扫描结果 |
| **过程节点** | TSMC 12nm（面积 = 7nm 基线 × 2.70，**TSMC 12FFC 密度比，源自 bitcell 查表**）；各节点绑定物理可行频率范围（详见"频率-节点绑定"行） | [`sim/contracts/bitcell.py`](sim/contracts/bitcell.py) `BitcellTable` + [`references/calibration/parameters.yaml`](references/calibration/parameters.yaml)；频率约束见 [`sim/config/dse_axes.yaml`](sim/config/dse_axes.yaml) |
| **频率-节点绑定** | 各节点绑定物理可行频率范围：7nm 800–2000 MHz, 12nm 800–1200 MHz, 22nm 400–800 MHz, 28nm 200–600 MHz。block 引擎 BW-bound 不受频率影响（20.8 tok/s 恒定），FSA compute-bound 在老旧节点因频率上限而大幅落后 | [`sim/config/dse_axes.yaml`](sim/config/dse_axes.yaml)（frequency-bound constraints）；频率感知跨节点数据见 [`.omo/evidence/investigate-fsa-cross-node-freq.md`](.omo/evidence/investigate-fsa-cross-node-freq.md) |
| **SRAM bitcell 溯源** | TSMC HD bitcell 面积查表（7/12/22/28nm），peripheral overhead 可配，外部参考（TPUv1/RK1828）偏差 <30% | [`sim/contracts/bitcell.py`](sim/contracts/bitcell.py) |
| **DRAM 效率模式化** | sequential decode baseline 0.90，random KV 访问 0.50（固定延迟惩罚 40 cycles） | [`docs/model-trust-and-release.md`](docs/model-trust-and-release.md) §DRAM 效率模式化方法 |
| **BW-面积耦合** | On-chip 3D DRAM BW = area × 7.5 GB/s/mm²（**基于 RK1828 的外推，未绑定硅实现**） | RK1828 验证 |

### 1.4 双场景技术路线（探索性估算）

> 以下数值为 Arc Model 在**当前校准假设**下的估算，包含 T0/T1 参数；在 `decision-grade` 证据补齐前，不应用作产品承诺。

| | Scenario A (低成本) | Scenario B (高性能) |
|:---|:---|:---|
| 内存 | LPDDR5-64b (51.2 GB/s) | On-chip 3D DRAM (500 GB/s) |
| 模型 | Qwen2.5-3B INT4 | Qwen2.5-7B INT4 |
| seq_len | 128 (chat) | 1024 (VLM/VLA) |
| 引擎 | block 64×128 (8.2 TOPS) | block 32×1536 (49 TOPS) |
| 面积 @12nm | 61mm²（估算） | 66mm²（估算） |
| Decode | 23 tok/s（估算） | 148 tok/s（估算） |
| TTFT | 45ms（估算） | 160ms（估算） |
| 跨节点验证结论 | block 引擎面积单调 7→28nm（99→261mm² @lpddr5_3b）；全 8 引擎 × 4 节点排名矩阵揭示 os_systolic 在跨所有节点和场景中均占绝对领先（31.8 tok/s @lpddr5, 310.9 tok/s @onchip），GMMA 在高 BW 场景中作为第二名具竞争力（203.5 tok/s @7nm onchip）；WMMA/GMMA PE 校准已升级 T1（PE 面积 4.5/5.5 mm² @7nm、WMMA 片段序列化 120 cycles、GMMA pipeline，源自 H100 SM die/Volta 架构分析），WMMA 全模型 tok/s 仍低（~0.5 tok/s）但较校准前提升 ~10×（per-FFN_down-GEMM 6.9→67.6 tok/s）；决策级状态：**FAIL（频率-节点绑定为探索性结论，多节点覆盖不完全）** | [`全引擎跨节点排名矩阵`](.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md)；频率感知数据：[`.omo/evidence/investigate-all-engines-cross-node-freq.md`](.omo/evidence/investigate-all-engines-cross-node-freq.md) |
| 详细报告 | [`reports/arch-report-A-lpddr5-3b.md`](reports/arch-report-A-lpddr5-3b.md) | [`reports/arch-report-B-3ddram-7b.md`](reports/arch-report-B-3ddram-7b.md) |

---

## 二、目录结构

```
npu_arc_model/
├── README.md                          # 本文件
├── sim/
│   ├── arc_model.py                   # 精度门 + 性能模型入口
│   ├── design_space_explorer.py       # 8 引擎 DSE 扫描引擎
│   ├── dse_scenario.py                # 四阶段 DSE 方法论
│   ├── fsa_ref.py                     # FSA 内联 Softmax 参考
│   ├── npu_sim.py                     # 主仿真入口
│   ├── golden_executor.py             # MXU/SFU/Vector/DMA 黄金模型
│   ├── quantize.py                    # INT4 量化实现
│   ├── validate_quant.py              # 量化精度验证
│   ├── overnight_loop.py              # 自动化健康检查
│   ├── sw_overhead_eval.py            # 软件开销评估
│   ├── bottleneck_analysis.py         # 瓶颈分析
│   ├── qwen25_forward.py              # Qwen2.5 前向传播模型
│   ├── qwen25_l3.py                   # Qwen2.5 Layer-3 详细分析
│   ├── eval_models.py                 # 模型评估
│   ├── eval_model_zoo.py              # Model Zoo 批量评估
│   ├── model_zoo_report.py            # Model Zoo 报告生成
│   ├── model_specs.py                 # 模型规格定义
│   ├── weight_cache_eval.py           # 权重缓存评估
│   ├── hw_bottleneck_chain.py         # 硬件瓶颈链分析
│   ├── hw_levels.py                   # 层级分析
│   ├── dma_improvement_eval.py        # DMA 改进评估
│   ├── config/
│   │   ├── scenarios.yaml             # 场景定义
│   │   ├── design_space.yaml          # DSE 扫描范围
│   │   └── npu_config.yaml            # 硬件参数
│   ├── engine/                        # 8 种 MAC 引擎
│   │   ├── mac_engine.py              # 工厂 + ABC
│   │   ├── systolic_engine.py         # Weight-stationary 脉动
│   │   ├── block_engine.py            # 全并行 MAC
│   │   ├── gmma_engine.py             # Group MMA
│   │   ├── wmma_engine.py             # Warp-level MMA
│   │   ├── fsa_engine.py              # FSA 内联 Softmax
│   │   ├── os_systolic_engine.py      # Output-stationary
│   │   ├── is_systolic_engine.py      # Input-stationary
│   │   ├── tensor_core_engine.py      # Tensor Core 风格
│   │   ├── ppa_model.py               # 面积/功耗模型
│   │   ├── compiler.py                # 模型→ISA 编译器
│   │   ├── timeline.py                # 事件驱动时序
│   │   ├── isa.py                     # NPU ISA 定义
│   │   └── multicore.py               # 多核扩展
│   ├── models/                        # 性能模型
│   │   ├── mxu.py                     # MXU 计算模型
│   │   ├── sfu.py                     # SFU 延迟模型
│   │   ├── vector.py                  # Vector 模型
│   │   ├── dma.py                     # DMA 模型
│   │   ├── dram.py                    # DRAM 带宽模型
│   │   ├── kv_cache.py                # KV Cache 模型
│   │   ├── pcie.py                    # PCIe 模型
│   │   ├── crossbar.py                # Crossbar 互联
│   │   ├── noc.py                     # NoC 模型
│   │   └── sw_overhead.py             # 软件开销模型
│   └── cv/                            # CV 模型支持
│       ├── cv_sim.py                  # CV 仿真
│       ├── cv_trace.py                # CV 算子 trace
│       ├── conv_mapper.py             # Conv→GEMM 映射
│       ├── onnx_importer.py           # ONNX 导入
│       ├── validate_onnx.py           # ONNX 验证
│       └── traces/
│           ├── vit_trace.py           # ViT-B/16
│           ├── qwen_vl_vit_trace.py   # Qwen2.5-VL ViT
│           ├── yolov8n_trace.py       # YOLOv8n
│           ├── resnet18_trace.py      # ResNet-18
│           ├── resnet50_trace.py      # ResNet-50
│           └── sd_unet_trace.py       # Stable Diffusion UNet
├── scripts/                           # 性能分析与校准脚本
│   ├── analyze_perf.py                # 综合性能分析
│   ├── analyze_sfu_perf.py            # SFU 性能分析
│   ├── analyze_vector_perf.py         # Vector 性能分析
│   ├── calibrate_mxu_model.py         # MXU 模型校准
│   ├── compare_mxu_perf.py            # MXU 对比
│   ├── compare_sfu.py                 # SFU 对比
│   ├── run_mxu_perf_case.py           # MXU 性能案例
│   ├── run_sfu_perf_case.py           # SFU 性能案例
│   ├── run_vector_perf_case.py        # Vector 性能案例
│   ├── generate_mobilenetv3_ppa.py    # MobileNetV3 PPA
│   └── export_mobilenetv3_onnx.py     # ONNX 导出
├── docs/                              # 架构设计文档
│   ├── tiny-npu-analysis/             # tiny-NPU 对比分析
│   └── ...
├── reports/                           # DSE 分析报告
├── sim/results/                       # 扫描与评估结果
│   ├── engine_eval_v3.md              # 引擎评估 v3
│   ├── param_sweep.json               # 参数扫描结果
│   └── param_sweep_v2.json            # 参数扫描 v2
└── references/                        # 数据溯源
```

---

## 三、文档索引

### 3.1 方法论

| 文档 | 内容 |
|:---|:---|
| [`docs/arc_vs_func.md`](docs/arc_vs_func.md) | Arc Model vs Func Model 角色划分 |
| [`docs/design-methodology.md`](docs/design-methodology.md) | 设计方法论：DSE 流程、决策准则 |
| [`docs/Edge_NPU_Architecture_Proposal.md`](docs/Edge_NPU_Architecture_Proposal.md) | 边缘 NPU 架构提案 |

### 3.2 架构设计

| 文档 | 内容 |
|:---|:---|
| [`docs/NPU硬件详细架构设计v0.1.md`](docs/NPU硬件详细架构设计v0.1.md) | 硬件微架构详细设计（670 行） |
| [`docs/NPU_Engines_Architecture_Guide.md`](docs/NPU_Engines_Architecture_Guide.md) | 8 引擎对比、Pareto 前沿、决策树 |
| [`docs/NPU系统级模拟器方案v0.1.md`](docs/NPU系统级模拟器方案v0.1.md) | 系统级模拟器架构方案 |
| [`docs/NPU软件架构方案v0.1.md`](docs/NPU软件架构方案v0.1.md) | 软件栈方案 v0.1 |
| [`docs/NPU软件架构方案v0.2.md`](docs/NPU软件架构方案v0.2.md) | 软件栈方案 v0.2（两阶段：llama.cpp → ExecuTorch） |
| [`docs/端侧NPU协处理器产品需求方案v0.1.md`](docs/端侧NPU协处理器产品需求方案v0.1.md) | MRD + PRD |

### 3.3 Model Zoo & CV

| 文档 | 内容 |
|:---|:---|
| [`docs/model_zoo.md`](docs/model_zoo.md) | Model Zoo 规划（LLM 10 + CV 9，B/C 分类） |
| [`docs/CaduceusCore Model Zoo 实施路线图.md`](docs/CaduceusCore Model Zoo 实施路线图.md) | 实施路线图 + 优先级 |
| [`docs/cv_gantt.md`](docs/cv_gantt.md) | CV 模型开发甘特图 |
| [`docs/qwen25-3b-forward-spec.md`](docs/qwen25-3b-forward-spec.md) | Qwen2.5-3B 前向传播规格 |
| [`docs/ttft_gantt.md`](docs/ttft_gantt.md) | TTFT 分析甘特图 |

### 3.4 性能校准

| 文档 | 内容 |
|:---|:---|
| [`docs/mxu-perf-calibration.md`](docs/mxu-perf-calibration.md) | MXU 性能模型校准 |
| [`references/area_sources.md`](references/area_sources.md) | PE 面积数据溯源（TPUv1/Eyeriss/RK1828/M4） |

### 3.5 架构对比分析

| 文档 | 内容 |
|:---|:---|
| [`docs/tiny-npu-analysis/tiny-NPU五大计算引擎RTL解剖.md`](docs/tiny-npu-analysis/tiny-NPU五大计算引擎RTL解剖.md) | tiny-NPU 5 引擎 RTL 解剖笔记 |
| [`docs/tiny-npu-analysis/tiny-NPU vs CaduceusCore 深度对比.md`](docs/tiny-npu-analysis/tiny-NPU%20vs%20CaduceusCore%20深度对比.md) | 引擎/控制/面积全面对比 |
| [`docs/tiny-npu-analysis/Streaming vs LUT Softmax 深度对比.md`](docs/tiny-npu-analysis/Streaming%20vs%20LUT%20Softmax%20深度对比.md) | 两种 Softmax 量化对比 |
| [`docs/tiny-npu-analysis/tiny-NPU vs CaduceusCore PPA对比.md`](docs/tiny-npu-analysis/tiny-NPU%20vs%20CaduceusCore%20PPA对比.md) | LLM+CV 需求满足度矩阵 |
| [`docs/控制面调度改进方案.md`](docs/控制面调度改进方案.md) | 记分板+统一接口改造方案（含 mermaid 时序） |

| 文档 | 内容 |
|:---|:---|
| [`docs/mxu-perf-calibration.md`](docs/mxu-perf-calibration.md) | MXU 性能模型校准 |
| [`references/area_sources.md`](references/area_sources.md) | PE 面积数据溯源（TPUv1/Eyeriss/RK1828/M4） |

---

## 四、报告索引

| 报告 | 日期 | 内容 |
|:---|:---|:---|
| [`reports/arch-dse-three-scenarios.md`](reports/arch-dse-three-scenarios.md) | 2026-07 | 三场景 DSE 最终报告 |
| [`reports/arch-report-A-lpddr5-3b.md`](reports/arch-report-A-lpddr5-3b.md) | 2026-07 | Scenario A: LPDDR5+3B, block 64×128, 61mm² |
| [`reports/arch-report-B-3ddram-7b.md`](reports/arch-report-B-3ddram-7b.md) | 2026-07 | Scenario B: 3D DRAM+7B, block 32×1536, 66mm² |
| [`reports/dse-fsa-eval-2026-06-27.md`](reports/dse-fsa-eval-2026-06-27.md) | 2026-06 | FSA 引擎评估（2,016 配置） |
| [`reports/dse-v2-unified-sram-2026-06-29.md`](reports/dse-v2-unified-sram-2026-06-29.md) | 2026-06 | 统一 SRAM 模型 DSE |
| [`reports/dse-7b-3d-dram-2026-06-29.md`](reports/dse-7b-3d-dram-2026-06-29.md) | 2026-06 | 7B + 3D DRAM 分析 |

---

## 五、快速开始

```bash
cd ~/npu_arc_model

# 1. 精度门 — 量化方案验证
PYTHONPATH=. python3 sim/arc_model.py \
  --model ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --scheme both

# 2. DSE 扫描（快速模式：3 引擎 × 关键配置）
python3 sim/design_space_explorer.py --quick

# 2a. All-engine smoke test（验证 DSE 引擎覆盖）
python3 -m pytest sim/tests/test_dse_coverage.py sim/tests/test_engine_instantiate.py -v

# 3. 全量 DSE（8 引擎 × 全维度扫描）
python3 sim/design_space_explorer.py

# 4. 场景需求检查 + 预检（Phase -1 + 0）
python3 -c "
from sim.dse_scenario import check_requirements, preflight, print_preflight
from sim.config.npu_config import load_config
config = load_config()
rc = check_requirements('lpddr5_3b', config)
pf = preflight('lpddr5_3b', config)
print_preflight(pf)
"

# 5. 软件开销评估
python3 sim/sw_overhead_eval.py

# 6. Overnight 健康检查
python3 sim/overnight_loop.py
```

---

## 六、与 CaduceusCore 的关系

```
npu_arc_model (本仓库)              CaduceusCore (~/npu)
─────────────────────────           ──────────────────────
Arc Model: 精度门 + DSE            Func Model: bit-exact 仿真
Golden Executor: 量化验证           MMIO Bridge: 寄存器级接口
Engine models: 8 引擎 PPA          Tile Scheduler: 固件调度
CV traces: 模型拓扑                 RTL: Verilog 实现
Config: 场景/参数定义               Firmware: RISC-V 微码
                                    Spike: RISC-V 模拟器
         ↓                                ↓
    选什么架构                          怎么实现
         ↓                                ↓
    DSE 结论 ──── 输入 ────→ 产品决策 → 硬件实现
```

两仓库**独立演进**，共享 Golden Executor 的量化验证标准。Arc Model 的 DSE 结论通过报告形式输入 CaduceusCore 的产品决策。

---

## 七、关键洞见（探索性结论）

> 本节洞见来自当前 Arc Model 的解析估算，部分驱动参数仍处于 T0/T1。详见 [`docs/model-trust-and-release.md`](docs/model-trust-and-release.md) 了解信任等级与发布边界。

1. **SRAM 不是常数，是 DSE 中最关键的性能杠杆** — 在 LPDDR5 内存墙下，FSA 省下 PE 面积可全部转为 SRAM（**192 tok/s 为估算**），block PE 过大无空间加 SRAM（**41 tok/s 为估算**）

2. **TTFT 约束可覆盖 BW 瓶颈的阵列建议** — 即使 decode 是 BW 瓶颈，长 seq_len 的 prefill 可能是 compute 瓶颈。seq_len=1024 需要 **估算** 72 TOPS，BW 逻辑推荐 16 TOPS 是错的

3. **On-chip 3D DRAM 下 SRAM 零性能影响** — 权重常驻、无 K-tiling、KV 4µs/layer（**理想假设**）。512KB 即可，8MB 是 LPDDR5 习惯的延续

4. **FSA 在低 BW 赢，block 在高 BW 赢** — 引擎选择不取决于引擎本身，取决于带宽场景。LPDDR5→FSA（面积优先），3D DRAM→block（宽阵列无 pipeline 惩罚）

5. **面积必须可溯源** — PE 基线来自 TPUv1 ISCA 2017 die-shot，不可凭经验猜测

6. **os_systolic 是全场景最优引擎** — 全 8 引擎 × 4 节点跨节点 DSE 揭示：os_systolic（输出驻留脉动阵列，Gemmini 风格）在低 BW (51.2 GB/s) 下保持 31.8 tok/s，在高 BW (500 GB/s) 下达到 310.9 tok/s — 在两种极端带宽条件下均超越包括 block、GMMA、input_stationary 在内的所有其他引擎。GMMA（Hopper H100 风格异步 DMA）在高 BW 场景中作为第二名具竞争力（203.5 tok/s @7nm），但在老旧节点（28nm）因频率上限和面积膨胀被 os_systolic 大幅拉开（97.7 vs 224.1 tok/s）。此项发现来自探索性 DSE，待 T2+ 参数校准后验证。（[`详细排名矩阵`](.omo/evidence/task-4-cross-node-all-engines-dse-matrix.md)）

---

## 八、历史基线与适用范围声明

> ⚠️ **重要**: 以下"63 passed"与 2026-07-28 postfix 证据**仅证明历史 bug 集合已修复** — 不证明跨频率、batch、内存层级或高利用率场景下的物理正确性。

已知未验证/待修复问题领域：
- **频率传播**: `--freq` flag 目前的覆盖不改变输出，多频点行为待 Todo 6 修复
- **Batch 边界**: M=2→3 等递增场景的 total latency 反向下降（Systolic）、OS M scaling 错误 — 待 Todo 5 修复
- **内存层级**: 3D DRAM 未进入搜索空间，capacity>0 全驻留假设不成立 — 待 Todo 10/11 修复
- **利用率上限**: 当前引擎不保证 `utilization <= 1` — 待 Todo 3/5 修复

### 8.1 自动复现环境 (uv)

```bash
python3 --version  # >= 3.10, < 3.13

# 安装 uv（如未安装）
pip install uv

# 从 lock 文件精确复现环境
uv sync --frozen

# 运行全部测试
uv run pytest -q
```

### 8.2 Clone 与复现

```bash
git clone git@github.com:anekin/npu_arc_model.git
cd npu_arc_model
uv sync --frozen

# 1. 一键运行全部回归测试
uv run pytest -q
# 期望: 63 passed, 0 failed (历史 bug 集 — 非物理正确性证明)

# 2. 快速 DSE 回归（36 configs）
uv run python sim/design_space_explorer.py --quick --output /tmp/dse_quick.json
# 期望: exit 0, errors=0

# 3. 全量 DSE 回归（13,440 configs）
uv run python sim/design_space_explorer.py --output /tmp/dse_full.json
# 期望: exit 0, errors=0

# 4. 端到端 CLI 基准
uv run python sim/npu_sim.py --engine systolic --json | python3 -c "import sys,json; print(json.load(sys.stdin)['decode']['tok_per_s'])"
# 期望: ~10.15 tok/s
uv run python sim/npu_sim.py --json | python3 -c "import sys,json; print(json.load(sys.stdin)['decode']['tok_per_s'])"
# 期望: ~21.59 tok/s
```

### 8.3 基线溯源

以下信息记录了 `arc-model-scenario-driven-dse-development` 计划的基线点：

| 项目 | 值 |
|:---|:---|
| **计划** | `.omo/plans/arc-model-scenario-driven-dse-development.md` |
| **基线 Commit** | `94ac75159dbc4a3bcb5c3bc1715923eec7e6ad05` (chore(research): add embodied AI source materials) |
| **lock digest** | `ce33a246885fb5620a61ecd2b5d8aa773d4503c3a99a9c5b1ecb56fabfe8b288` |
| **config digest** | `d3ad177cd825b7ef6342bc0f53402e61e5d4438267ae4112d7e6aca041a08217` (`sim/config/npu_config.yaml`) |
| **process_node** | 12nm (TSMC 12FFC) |
| **node_scale** | 2.70× = TSMC 12FFC 密度比（非几何 (12/7)²）— **已修正**，详见 [`sim/contracts/bitcell.py`](sim/contracts/bitcell.py) `_node_scale_factor(12)` → 2.70 |
| **dram_efficiency** | 0.85 — conservative baseline；README 中 75% 声称已被证伪（待 Todo 6 统一清理） |
| **Python** | >=3.10,<3.13 (via `pyproject.toml`) |
| **复现命令** | `uv sync --frozen && uv run pytest -q` |

### 8.3 复现证据链

| 资产 | 路径 | 内容 |
|------|------|------|
| 可执行修复计划 | `.omo/plans/dse-engine-model-bug-fix.md` | 11 条 todo + F1-F4 终审 |
| Boulder 工作记录 | `.omo/boulder.json` | 子 Agent 任务会话与耗时 |
| 过程学习/问题 | `.omo/notepads/dse-engine-model-bug-fix/*.md` | learnings, issues, problems, decisions |
| 验证证据 | `.omo/evidence/task-*.txt/json` | 每条 todo 与终审的实测输出 |
| 原始 dated 报告 | `reports/dse-engine-model-bugs-2026-07-27.md` | **只读**，SHA256 未变 |
| 修复后报告 | `reports/dse-engine-model-bugs-postfix-2026-07-27.md` | before/after 对照表 |

### 8.4 关键验证命令

```bash
# 原始报告 SHA256 校验（必须与计划一致）
sha256sum reports/dse-engine-model-bugs-2026-07-27.md
# 61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3

# 7-engine FFN_down 基准（与修复后报告一致）
python3 -c "
import json
with open('.omo/evidence/task-10-engine-ffn-down.json') as f:
    d = json.load(f)
for r in d['results']:
    print(f\"{r['engine_type']:12s} {r['tok_per_s']:8.1f} tok/s  {r['bottleneck']}\")
"
```

---

## 九、贡献

Arc Model 维护与 CaduceusCore 同步：
- `sim/golden_executor.py` — 量化方案变更时同步
- `sim/engine/ppa_model.py` — 面积模型变更时同步（含 `area_sources.md`）
- `sim/config/scenarios.yaml` — 新增产品场景时更新
- 报告 → CaduceusCore 产品决策输入
