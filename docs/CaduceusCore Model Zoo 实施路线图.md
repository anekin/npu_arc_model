# CaduceusCore Model Zoo 实施路线图 v2.0

> 详细规划见 `model_zoo.md`，本文档只跟踪执行进度。
>
> 基于硬件需求分析重新排序，2026-06-22
> 原则：最少新代码验证最多硬件 → 最快展示竞争力 → 最难问题前置

---

## 三、实施阶段

### Phase 0（已完成）
✅ LLM 基础：Qwen2.5-1.5B/3B GGUF→INT4→MXU GEMM→cos_sim
✅ SFU：GELU, Softmax, RoPE
✅ E2E：Spike + firmware MMIO 全链路

---

### Phase 1：CV 快速验证 — im2col + 最小新硬件
**目标**：用最少工作量跑通第一个 CV 模型

| 顺序 | 模型 | 新增代码 | 验证的硬件 |
|:--:|------|------|------|
| 1.1 | **MobileNetV3-Small** ✅ 已完成 | ONNX 加载器 + im2col + ReLU/Swish(SFU) + Pool2D | im2col, Conv SFU, Pool |
| 1.2 | Arc Model 适配 | CV 精度门：im2col→GEMM 的 cos_sim | 量化精度 |
| 1.3 | Func Model 适配 | Conv2D golden reference (im2col+MXU+ReLU+Pool) | bit-exact 参考 |

**理由**：2.5M 参数最小，Depthwise Conv 测试 im2col 极端情况（1×1 膨胀到 3×3 滑动窗口）。

---

### Phase 2：LLM 竞品对齐
**目标**：补齐 Llama/Phi，让 LLM 列表覆盖全球标杆

| 顺序 | 模型 | 工作量 | 价值 |
|:--:|------|:--:|------|
| 2.1 | **Llama 3.2-1B** | GGUF 加载路径（tensor 名映射） | 海外客户评估必问 |
| 2.2 | **Llama 3.2-3B** | 同上，规模递增 | 3B 甜点位全球标杆 |
| 2.3 | **Phi-3.5-mini** | GGUF 加载路径 | 对标 Hailo-10H |
| 2.4 | **DeepSeek-R1-1.5B** | GGUF 加载 + KV cache 验证 | RTL 验证目标 |

**理由**：这些模型不需要新硬件，只需适配 GGUF tensor 名映射。可以并行推进。

---

### Phase 3：快速竞争优势 — ViT-B
**目标**：展示独家 Transformer CV 能力

| 顺序 | 模型 | 新增代码 | 验证的硬件 |
|:--:|------|------|------|
| 3.1 | **ViT-B/16** | PyTorch→INT4 转换 + Patch Embedding | **0 新硬件** |
| 3.2 | Arc Model | ViT 精度 + 性能评估 | Self-Attn 复用 |
| 3.3 | Func Model | ViT golden reference | 仅 MXU + SFU |

**理由**：ViT-B 全部是矩阵乘法（Self-Attention + MLP），不需 im2col、不需 Conv SFU、不需 Pool。**纯软件工作量**，却能展示 Hailo-8/Coral 完全做不到的能力。这是最快 C 类胜利。

---

### Phase 4：CNN 压力测试
**目标**：验证全部 Conv 硬件子系统

| 顺序 | 模型 | 新增硬件 | 新验证维度 |
|:--:|------|:---:|------|
| 4.1 | **ResNet-18** | ResAdd | 残差连接 + BN folding |
| 4.2 | **YOLOv8n** | Concat, Upsample | 完整检测流水线 |
| 4.3 | **EfficientNet-B0** | —(同 MobileNet) | 现代 CNN 架构 |
| 4.4 | **EfficientDet-Lite0** | —(同前) | BiFPN 多尺度 |

---

### Phase 5：规模扩展
**目标**：推送 NPU 至极限

| 顺序 | 模型 | 意义 |
|:--:|------|------|
| 5.1 | **ResNet-50** | MLPerf 基准 |
| 5.2 | **YOLOv8s** | 300MB 激活压力 |
| 5.3 | **YOLOv8s-seg** | 分割任务 |
| 5.4 | **Mistral-7B** | SWA/GQA 不同 attention |
| 5.5 | **Llama 3.1-8B** | 全球 8B 标杆 |

---

## 四、关键决策点

### 为什么 ViT 提前到 Phase 3（原规划 Phase 4）

| 原规划 | 新规划 | 原因 |
|------|:--:|------|
| 先跑 B1-B3 CNN | 先跑 ViT-B | ViT 零新硬件，CNN 需 im2col+SFU+Pool |
| CV 全链路后展示 | 最快展示独家能力 | Hailo/Coral 跑不了 ViT，早出牌早占位 |

### 为什么 LLM 扩展提前到 Phase 2

- Llama/Phi 的 GGUF 加载只需 tensor 名映射，与 CV im2col 完全正交
- 可以并行推进：Phase 1(CV) 和 Phase 2(LLM) 互不阻塞

---

## 五、阶段产出

| Phase | 时间估算 | 里程碑产出 |
|:--:|:--:|------|
| 1 | 2-3 天 | MobileNetV3 通过 cos_sim gate → 首个 CV 模型验证通过 |
| 2 | 1-2 天 | Llama 3.2-1B/3B + Phi-3.5 + DeepSeek-R1 加载通过 |
| 3 | 1 天 | ViT-B 精度验证 → "Transformer CV"能力声明 |
| 4 | 3-4 天 | ResNet-18 + YOLOv8n + EfficientNet-B0 + EfficientDet-Lite0 |
| 5 | 2-3 天 | ResNet-50 + YOLOv8s + YOLOv8s-seg + Mistral + Llama-8B |

**总计**：~10-13 天跑通全部 19 个模型。
