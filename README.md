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
| **DRAM 效率** | 75%（非 85%） | JEDEC 实测：刷新 5.4% + 行冲突 4.5% + 总线 1-2% |
| **量化方案** | per-block (g=128) INT4 | cos_sim > 0.99，比 per-channel 稳定 0.014 |
| **SRAM 灵敏度** | LPDDR5 场景：4-6MB sweet spot；3D DRAM 场景：512KB 足够 | DSE 扫描结果 |
| **过程节点** | TSMC 12nm（面积 = 7nm 基线 × 2.94） | 用户指定 |
| **BW-面积耦合** | On-chip 3D DRAM BW = area × 7.5 GB/s/mm² | RK1828 验证 |

### 1.4 双场景技术路线

| | Scenario A (低成本) | Scenario B (高性能) |
|:---|:---|:---|
| 内存 | LPDDR5-64b (51.2 GB/s) | On-chip 3D DRAM (500 GB/s) |
| 模型 | Qwen2.5-3B INT4 | Qwen2.5-7B INT4 |
| seq_len | 128 (chat) | 1024 (VLM/VLA) |
| 引擎 | block 64×128 (8.2 TOPS) | block 32×1536 (49 TOPS) |
| 面积 @12nm | 61mm² | 66mm² |
| Decode | 23 tok/s | 148 tok/s |
| TTFT | 45ms ✓ | 160ms ✓ |
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
├── docs/                              # 架构设计文档
├── reports/                           # DSE 分析报告
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

## 七、关键洞见

1. **SRAM 不是常数，是 DSE 中最关键的性能杠杆** — 在 LPDDR5 内存墙下，FSA 省下 PE 面积可全部转为 SRAM（192 tok/s），block PE 过大无空间加 SRAM（永远 41 tok/s）

2. **TTFT 约束可覆盖 BW 瓶颈的阵列建议** — 即使 decode 是 BW 瓶颈，长 seq_len 的 prefill 可能是 compute 瓶颈。seq_len=1024 需要 72 TOPS，BW 逻辑推荐 16 TOPS 是错的

3. **On-chip 3D DRAM 下 SRAM 零性能影响** — 权重常驻、无 K-tiling、KV 4µs/layer。512KB 即可，8MB 是 LPDDR5 习惯的延续

4. **FSA 在低 BW 赢，block 在高 BW 赢** — 引擎选择不取决于引擎本身，取决于带宽场景。LPDDR5→FSA（面积优先），3D DRAM→block（宽阵列无 pipeline 惩罚）

5. **面积必须可溯源** — PE 基线来自 TPUv1 ISCA 2017 die-shot，不可凭经验猜测

---

## 八、贡献

Arc Model 维护与 CaduceusCore 同步：
- `sim/golden_executor.py` — 量化方案变更时同步
- `sim/engine/ppa_model.py` — 面积模型变更时同步（含 `area_sources.md`）
- `sim/config/scenarios.yaml` — 新增产品场景时更新
- 报告 → CaduceusCore 产品决策输入
