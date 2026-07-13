# NPU Arc Model

面向应用需求的 NPU 架构设计空间搜索（DSE）工具。

Arc Model 回答“针对这个应用场景，应选择什么 NPU 架构”；CaduceusCore 中的 Func Model 和 RTL 回答“选定架构如何准确实现”。

## 模型边界

| 层次 | 输入 | 输出 | 本仓库是否负责 |
|---|---|---|---|
| Arc Model | 模型、上下文长度、吞吐/TTFT、面积、功耗、内存和工艺约束 | 可行架构候选、指标分解、约束证据和推荐顺序 | 是 |
| Func Model | 已冻结的架构规格、算子和量化语义 | bit-exact 功能结果 | 否 |
| RTL | 已冻结的微架构和接口 | 可综合实现及实测 PPA/时序 | 否 |

Func Model、RTL 或硅后测量结果可以作为 Arc Model 的校准证据，但不能成为 DSE 的运行时依赖。

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements-dev.txt
# Linux/macOS
# .venv/bin/python -m pip install -r requirements-dev.txt

# 内置场景快速搜索
.venv\Scripts\python sim/design_space_explorer.py --scenario lpddr5_3b --quick --top 3

# 自定义应用需求
.venv\Scripts\python sim/design_space_explorer.py \
  --requirements sim/config/application-requirements.example.yaml \
  --quick --top 3

# 回归测试
.venv\Scripts\python -m pytest sim/tests -q
```

Linux/macOS 请将 `.venv\Scripts\python` 换成 `.venv/bin/python`。
## 内置场景定位

- **场景 A：低成本端侧算力扩展**（`lpddr5_3b`）。Qwen2.5-3B、INT4 only、
  64-bit LPDDR5-6400，标称有效带宽效率 85%（75%/90% 保守/乐观角）；
  Decode TPS ≥20，TTFT 设计目标 ≤500ms、硬上限 ≤1000ms，目标达标点
  优先按面积和功耗排序。
- **场景 A 的 Agent 子场景**（`lpddr5_3b_agent`）。使用缓存前缀之后的
  875-token 增量 append、214-token 输出和 32K 最大上下文；保持 batch=1，
  4GB LPDDR 容量必须容纳 INT4 权重、FP16 KV 和运行时预留。TTFT≤2s 与
  Prefill TPS≥500 是待本地产品 trace 校准的设计目标，TTFT≤5s 是暂定硬上限。
- **场景 B：具身智能**（`onchip_7b`）。高带宽内存和更长上下文，保留明确的
  TTFT ≤200ms 实时约束；性能和时延达标优先于低成本。

场景 A 与场景 B 使用不同的约束契约，不应直接复用同一套推荐排序结论。

## 应用需求输入

`--requirements` 接受 YAML。关键字段包括：

```yaml
name: edge_chat_3b
model: qwen2.5-3b
seq_len: 128
process_nm: 12
memory:
  type: lpddr5
  bandwidth_gbps: 51.2

  dram_efficiency: 0.85
constraints:
  tps_min: 20
  ttft_ms_max: 1000
  area_mm2_max: 80
  power_w_max: 20
targets:
  ttft_ms_max: 500
objectives: [area_mm2, power_w, -tok_s]
```

性能优先场景可在 `workload` 中提供 `prompt_tokens`、`output_tokens`、
`max_context_tokens`、`concurrent_requests`、`decode_batch_size`、`kv_bits` 和
`runtime_reserve_mb`。Arc Model 分别输出单请求 Decode TPS、Aggregate TPS、
Prefill TPS、TTFT、ITL 和 E2E latency。若配置 `memory.capacity_gb`，
权重、最大上下文 KV Cache、激活和运行时预留必须装入可用容量，否则候选
会作为物理不可行点淘汰。

每次DSE还会为所有参与搜索的Engine输出统一对比表；无可行配置的Engine
显示距离约束最近的候选及失败原因。JSON结果保存在 `engine_comparison` 字段。

DSE 先执行硬约束过滤，再优先选择满足 `targets` 的候选，最后按
`objectives` 做字典序排序。若没有目标达标点，会在硬约束可行范围内选择
目标距离最近的候选；若没有硬约束可行点，则给出违反硬约束距离最小的候选，
不会把不可行方案伪装成推荐结果。

## 核心目录

- `sim/design_space_explorer.py`：正式 DSE CLI。
- `sim/dse/`：工作负载、单位换算、约束、候选评估与结构化结果。
- `sim/engine/`：计算引擎和 PPA 分析模型。
- `sim/config/scenarios.yaml`：内置应用场景。
- `sim/config/application-requirements.example.yaml`：自定义需求示例。
- `sim/tests/`：物理模型与仓库边界回归测试。
- `references/`：面积等模型参数的来源。
- `reports/`：历史搜索报告；历史数值不等于当前版本的可复现结论。

`sim/arc_model.py`、Golden Executor、量化前向和 ISA/调度相关代码是迁移时保留的旧版校准/研究工具，不是 DSE 主入口。其中 GGUF 精度验证需要额外的 `q4_dequant` 适配器。

## 当前评估原则

- 物理带宽统一以 GB/s 输入，并按候选频率换算为 bytes/cycle。
- Decode 与 Prefill 分开建模。
- Attention 显式计入 QK、Softmax、PV 和 KV 流量。
- On-chip memory 带宽与逻辑 die 面积耦合。
- 输出记录场景、配置、模型版本哈希和分项周期，便于复核。
- PPA 仍是分析估算；在获得 Func/RTL/综合/硅后数据后应通过版本化校准数据更新，而不是复制实现代码。

迁移结论和已知遗留项见 `docs/migration-audit.md`。
