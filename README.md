# NPU Arc Model

CaduceusCore 的架构设计与空间搜索（DSE）独立仓库。

## 定位

- **Arc Model**：8 引擎 DSE + 量化精度验证 + 性能建模
- **不包含**：Func Model（bit-exact 仿真）、RTL 实现、固件调度

## 目录

```
sim/
├── arc_model.py              # 精度门 + 性能模型入口
├── design_space_explorer.py  # 8 引擎 DSE 扫描
├── dse_scenario.py           # 场景需求检查 + 预检 + 交叉校验
├── fsa_ref.py                # FSA 内联 Softmax 参考模型
├── npu_sim.py                # 主入口
├── golden_executor.py        # MXU/SFU/Vector/DMA 黄金模型
├── quantize.py               # INT4 量化方案
├── validate_quant.py         # 量化精度验证
├── config/
│   ├── scenarios.yaml        # 场景定义（模型/seq_len/内存/约束）
│   ├── design_space.yaml     # DSE 扫描范围
│   └── npu_config.yaml       # 硬件参数
├── engine/                   # 8 种 MAC 引擎 + PPA 模型
├── models/                   # 性能模型（MXU/SFU/Vector/DMA/DRAM）
└── cv/                       # CV 模型 trace + 验证
```

## 快速开始

```bash
# 精度验证
python3 sim/arc_model.py --model ~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --scheme both

# DSE 扫描
python3 sim/design_space_explorer.py --quick

# 场景分析
python3 -c "
from dse_scenario import check_requirements, preflight
check_requirements('lpddr5_3b', config)
preflight('lpddr5_3b', config)
"
```

## 与 CaduceusCore 的关系

- Arc Model（本仓库）：负责「选什么架构」
- CaduceusCore（`~/npu`）：负责「怎么实现」— Func Model + RTL + 固件
- 两仓库独立演进，Arc 的 DSE 结论输入 CaduceusCore 的产品决策
