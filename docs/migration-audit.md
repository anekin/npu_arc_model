# Arc Model 迁移审计

## 结论

初始 `npu_arc_model` 仓库包含大部分历史 Arc 文件，但不能独立复现：DSE 不支持 README 中声明的场景参数，`arc_model.py` 在启动时依赖未迁移的 `q4_dequant`，且没有依赖清单和测试。

本次改进后，场景驱动 DSE 的必要闭包已完整迁入并可独立测试：

- 场景和自定义需求输入；
- LLM Decode/Prefill 工作负载；
- 八类计算引擎与 PPA 分析模型；
- 带宽单位、3D memory 带宽/面积耦合；
- TPS、TTFT、面积、功耗硬约束；
- 可行解排序、无可行解诊断、结构化结果与 provenance；
- 物理模型和 CLI smoke tests。

## 边界分类

### DSE 核心

`design_space_explorer.py`、`dse_scenario.py`、`dse/`、`engine/*_engine.py`、`engine/ppa_model.py`、分析型 `models/`、`model_specs.py`、`config/`、`cv/`、`references/`。

### 可选校准适配器

量化精度检查、ONNX 导入、与实测性能比较脚本。这些工具可以消费外部证据，但 DSE 不应在运行时依赖 CaduceusCore。

### 迁移时遗留、非 Arc 核心

`golden_executor.py`、`npu_sim.py`、`engine/isa.py`、`engine/compiler.py`、`models/crossbar.py`、`models/pcie.py`、Qwen bit-exact forward 等实现级代码。这些文件目前为保留历史研究结果而存在；新功能不应继续建立对它们的依赖。后续可在确认没有校准脚本使用后移至 `legacy/` 或删除。

## 已知缺口

- GGUF 精度工具需要外部 `q4_dequant`，不属于核心安装依赖。
- PPA 参数仍需用综合、RTL 或硅后数据持续校准，并记录数据来源、工艺和版本。
- 历史 `reports/` 的数字来自旧模型，必须重新运行后才能作为当前决策证据。
- 当前搜索是枚举加 Pareto/约束排序；更大的空间后续可增加分层搜索或贝叶斯优化，但不应先于模型校准。

## 仓库间接口

Arc Model 向 CaduceusCore 输出版本化架构规格和评估证据；CaduceusCore 反馈版本化校准数据。两边不复制运行时代码，也不依赖对方工作区中的 Python 模块。
