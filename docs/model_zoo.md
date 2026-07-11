# CaduceusCore Model Zoo

> 基于竞品调研 + MLPerf 行业基准 + 硬件需求分析 + Arc Model DSE 数据，2026-06-22
> v3.0：合并规划 + 实施路线图 + Arc Model DSE 数据

---

## 一、分层逻辑

| 层级 | 含义 | 选取标准 |
|:----:|------|------|
| **B类** | Baseline — 入场券 | 竞品都支持、MLPerf 基准、客户评估必问 |
| **C类** | Competitive — 差异化 | 竞品跑不了或跑不好的，展示 CaduceusCore 独有架构优势 |

---

# Part 1: LLM Model Zoo

## 1.1 竞品对比

同形态产品：Hailo-10H（LLM专用芯片，与Hailo-8 CV芯片分开）、算能 BM1684X。

| 模型 | Hailo-10H | 算能 BM1684X | Jetson Orin Nano | Apple M5 | CaduceusCore |
|------|:---:|:---:|:---:|:---:|:---:|
| Llama 3.2-1B | ✅ | ✅ | ✅ | ✅ | 待定 |
| Llama 3.2-3B | ✅ | ✅ | ✅ | ✅ | 待定 |
| Qwen2.5-1.5B | ✅ | ✅ | ✅ | ✅ | ✅ 已有 |
| Qwen2.5-3B | ⚠️ | ✅ | ✅ | ✅ | ✅ 已有 |
| DeepSeek-R1-1.5B | ✅ | ❌ | ✅ | ✅ | 待定 |
| Phi-3.5-mini (4B) | ✅ | ❌ | ✅ | ✅ | 待定 |
| Gemma 2 2B | ✅ | ✅ | ✅ | ✅ | 待定 |
| Llama 3.1-8B | ❌ | ✅ | ⚠️ | ✅ | 待定 |
| Mistral-7B | ❌ | ⚠️ | ✅ | ✅ | 待定 |
| Mixtral 8×7B | ❌ | ❌ | ❌ | ⚠️ | 待定 |

**关键发现**：Hailo 需要两颗芯片（Hailo-8 CV + Hailo-10H LLM），CaduceusCore 一颗搞定。新增 Jetson Orin Nano（GPU 方案）和 Apple M5（统一内存方案）补全竞品全景。但 10 个模型覆盖 Qwen/Llama/DeepSeek/Phi/Mistral/Gemma/Mixtral，架构多样性远超竞品。

## 1.2 分层规划

### B类 (Baseline) — 必须支持

| # | 模型 | 参数量 | hidden | layers | 选取理由 |
|:--:|------|:---:|:---:|:---:|------|
| B1 | **Llama 3.2-1B** | 1.2B | 2048 | 16 | 全球最通用 1B，Hailo-10H 必选 |
| B2 | **Qwen2.5-1.5B** ✅ | 1.5B | 1536 | 28 | 中文市场标杆 |
| B3 | **Llama 3.2-3B** | 3.2B | 3072 | 28 | 3B 甜点位，MRD 目标体量 |
| B4 | **Phi-3.5-mini** | 3.8B | 3072 | 32 | 微软端侧标杆，4B 接近 7B 效果 |

### C类 (Competitive) — 差异化

| # | 模型 | 参数量 | hidden | layers | 选取理由 |
|:--:|------|:---:|:---:|:---:|------|
| C1 | **Qwen2.5-3B** ✅ | 3.1B | 2560 | 28 | 已有 |
| C2 | **DeepSeek-R1-Distill-1.5B** | 1.5B | 1536 | 28 | RTL 验证目标，推理链模型 |
| C3 | **Qwen3-8B** ✅ | 8.2B | 4096 | 32 | 最新 Qwen 旗舰小模型 |
| C4 | **Llama 3.1-8B** | 8B | 4096 | 32 | 全球 8B 标杆，对标 GPT-J |
| C5 | **Gemma 4-12B** ✅ | 12B | 4096 | 40 | 最大规模压力测试 |
| C6 | **Mistral-7B** | 7.3B | 4096 | 32 | 欧洲旗舰，SWA 架构 |

## 1.3 LLM 变更汇总

| 动作 | 模型 | 原因 |
|:--:|------|------|
| ➕ 新增 | Llama 3.2-1B, Llama 3.2-3B, Phi-3.5-mini | 补齐全球覆盖 |
| ➕ 新增 | DeepSeek-R1-1.5B, Llama 3.1-8B, Mistral-7B | 架构多样性 + RTL 目标 |
| ➖ 移除 | Qwen2.5-7B | 由 Llama 3.1-8B 替代（同体量全球标杆） |

**最终：10 个 LLM（B类 4 + C类 6）**

---

# Part 2: CV Model Zoo

## 2.1 竞品对比

| 领域 | Hailo-8/8L (M.2) | Coral Edge TPU (USB/M.2) | CaduceusCore |
|------|:---:|:---:|:---:|
| 图像分类 | YOLOv8-cls, ResNet-18/50 | MobileNetV1/V2 | 待定 |
| 目标检测 | YOLOv5/v8/v11, EfficientDet | EfficientDet-Lite, SSD | 待定 |
| 实例分割 | YOLOv8-seg | DeepLabV3 | 待定 |
| **Transformer 视觉** | ❌ | ❌ | **✅ 独家** |

CaduceusCore 核心差异：systolic array 天然支持 im2col→GEMM（CNN）**和** self-attention（ViT）。Hailo-8/Coral 的数据流架构做不了 Transformer 视觉。

## 2.2 分层规划

### B类 (Baseline)

| # | 模型 | 任务 | 参数量 | 选取理由 |
|:--:|------|------|:---:|------|
| B1 | **MobileNetV3-Small** | 分类 | 2.5M | Coral 基准，最轻量端侧验证 |
| B2 | **ResNet-18** | 分类 | 11.7M | MLPerf + 所有竞品支持 |
| B3 | **YOLOv8n** | 检测 | 3.2M | Hailo + 算能主推 |
| B4 | **EfficientDet-Lite0** | 检测 | 3.2M | Coral + Hailo 双选 |

### C类 (Competitive)

| # | 模型 | 任务 | 参数量 | 选取理由 |
|:--:|------|------|:---:|------|
| C1 | **ViT-B/16** | 分类 | 86M | 🔑 Transformer 视觉，竞品跑不了 |
| C2 | **ResNet-50** | 分类 | 25.6M | MLPerf 工业基准 |
| C3 | **YOLOv8s** | 检测 | 11.2M | 检测压力测试 |
| C4 | **EfficientNet-B0** | 分类 | 5.3M | 现代 CNN 架构标杆 |
| C5 | **YOLOv8s-seg** | 分割 | ~12M | 分割赛道 |

**最终：9 个 CV（B类 4 + C类 5）**

---

# Part 3: 硬件需求分析

## 3.1 LLM：DRAM 带宽决定 tok/s

LLM decode 每 token 必须从 DRAM 读一遍全部权重。KV cache 占 DRAM 流量 <0.1%。

| 级别 | 模型 | 权重 | DRAM/tok | 峰值 tok/s | Arc Model DSE 实测 tok/s (≤12W, ≤40mm²) |
|:--:|------|:---:|---|:--:|:--:|
| 1B | Llama 3.2-1B | 0.4 GB | 384 MB | **108** | N/A |
| 1.5B | Qwen2.5-1.5B | 0.6 GB | 625 MB | **66** | **124.7** |
| **3B** | **Llama 3.2-3B** | **1.3 GB** | **1344 MB** | **31** ← MRD 甜点位 | N/A |
| 4B | Phi-3.5-mini | 1.5 GB | 1512 MB | **27** | N/A |
| 7B | Mistral-7B | 3.3 GB | 3328 MB | **12** | N/A |
| 8B | Llama 3.1-8B | 3.3 GB | 3328 MB | **12** | N/A |
| 12B | Gemma-4-12B | 4.7 GB | 4800 MB | **8.6** ← 带宽墙 | **16.3** |

## 3.2 CV：计算 + 激活内存主导

CV 权重只读一次（非自回归），瓶颈在中间激活膨胀和总 MACs。

| 级别 | 模型 | 权重 | MACs | 峰值激活 | FPS (估) | Arc Model DSE FPS |
|:--:|------|:---:|:---:|:---:|:--:|:--:|
| 微 | MobileNetV3-Small | 10MB | 56.5M | 8MB | **825** | **497.6** (best systolic) / **1243.3** (best tensor_core) |
| 小 | ResNet-18 | 47MB | 1.8G | 30MB | **193** | estimated |
| 中 | YOLOv8n | 13MB | 8.7G | 200MB | **70** | estimated |
| 中 | EfficientNet-B0 | 21MB | 0.4G | 15MB | **500** | estimated |
| 大 | YOLOv8s | 45MB | 28.4G | 300MB | **43** | estimated |
| 大 | ViT-B/16 | 344MB | 17.6G | 80MB | **35** | estimated |

## 3.3 三类模型硬件特征

| | LLM（自回归） | CNN（im2col→GEMM） | ViT（Self-Attn） |
|------|:---:|:---:|:---:|
| DRAM 压力 | 🔴 极高 | 🟢 低 | 🟡 中 |
| 瓶颈 | 带宽 | 计算 + 激活膨胀 | 计算 |
| MXU 效率 | 🟡 M=1 利用率低 | 🟢 im2col 后 K 大 | 🟢 全 GEMM |
| SRAM 需求 | 8KB/tile | tiled im2col | 8KB/tile |
| KV cache | 需要 | 不需要 | 不需要 |

**核心洞察**：ViT-B **零新硬件**——全 GEMM 路径复用 LLM 的 Self-Attention 和 MLP。是最快能展示 C 类竞争力的模型。

---

# Part 4: 实施路线图

原则：**最少新代码验证最多硬件 → 最快展示竞争力 → 最难问题前置**

## 硬件覆盖矩阵

| 模型 | im2col | Conv SFU | Pool2D | ResAdd | Concat | Upsample | BN Fold | Self-Attn | **新增硬件** |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ViT-B/16 | — | — | — | — | — | — | — | ✅(已有) | **0** |
| MobileNetV3 | ✅ | ✅ | ✅ | — | — | — | ✅ | — | 3 |
| ResNet-18 | ✅ | — | ✅ | ✅ | — | — | ✅ | — | 3 |
| EfficientNet-B0 | ✅ | ✅ | ✅ | — | — | — | ✅ | — | 3 |
| YOLOv8n | ✅ | ✅ | — | — | ✅ | ✅ | ✅ | — | 4 |
| EfficientDet-Lite0 | ✅ | ✅ | ✅ | — | ✅ | — | ✅ | — | 4 |

## Phase 0（已完成）
- ✅ LLM 基础：Qwen2.5-1.5B/3B GGUF→INT4→MXU GEMM→cos_sim
- ✅ SFU：GELU, Softmax, RoPE
- ✅ E2E：Spike + firmware MMIO 全链路

## Phase 1：CV 基础 — im2col + 最小新硬件

| 顺序 | 任务 | 新增代码 | 新增硬件 |
|:--:|------|------|:--:|
| 1.1 | ONNX 模型加载器 | ONNX→numpy 权重提取 | — |
| 1.2 | im2col 引擎 | 滑动窗口展开为 GEMM 矩阵 | im2col |
| 1.3 | Conv SFU 算子 | ReLU, SiLU, Swish | SFU 扩展 |
| 1.4 | Pool2D | MaxPool, AvgPool | Pool 单元 |
| 1.5 | **MobileNetV3-Small** ✅ | 首个完整 CV 模型 ✅ | 全链路验证 ✅ |
| 1.6 | Arc Model CV 精度门 | im2col→GEMM cos_sim | — |
| 1.7 | Func Model CV golden ref | Conv2D bit-exact 参考 | — |

**里程碑**：MobileNetV3-Small ✅ 完成 → 首个 CV 模型验证通过，Arc Model DSE 评估完成

## Phase 2：LLM 竞品对齐（可与 P1 并行）

| 顺序 | 模型 | 工作量 | 新硬件 |
|:--:|------|:--:|:---:|
| 2.1 | **Llama 3.2-1B** | GGUF tensor 名映射 | 0 |
| 2.2 | **Llama 3.2-3B** | 同上 | 0 |
| 2.3 | **Phi-3.5-mini** | GGUF 加载路径 | 0 |
| 2.4 | **DeepSeek-R1-1.5B** | GGUF + KV cache 验证 | 0 |

**里程碑**：LLM Model Zoo 覆盖中国(Qwen/DeepSeek) + 全球(Llama/Phi)

## Phase 3：快速 C 类胜利 — ViT-B

| 顺序 | 任务 | 新硬件 |
|:--:|------|:--:|
| 3.1 | PyTorch→INT4 转换 + Patch Embedding | **0** |
| 3.2 | Arc Model ViT 精度评估 | **0** |
| 3.3 | Func Model ViT golden reference | **0** |

**理由**：ViT-B 全部是矩阵乘法，**零新硬件**，却展示 Hailo-8/Coral 完全做不到的能力。最快 C 类胜利。

## Phase 4：CNN 完善

| 顺序 | 模型 | 新硬件 | 验证维度 |
|:--:|------|:--:|------|
| 4.1 | ResNet-18 | ResAdd | 残差连接 + BN folding |
| 4.2 | YOLOv8n | Concat, Upsample | 完整检测流水线 |
| 4.3 | EfficientNet-B0 | — | 现代 CNN 架构 |
| 4.4 | EfficientDet-Lite0 | — | BiFPN 多尺度 |

## Phase 5：规模扩展

| 顺序 | 模型 | 意义 |
|:--:|------|------|
| 5.1 | ResNet-50 | MLPerf 基准 |
| 5.2 | YOLOv8s + YOLOv8s-seg | 300MB 激活压力 + 分割 |
| 5.3 | Mistral-7B | SWA 不同 attention |
| 5.4 | Llama 3.1-8B | 全球 8B 标杆 |

---

# Part 5: 完整总览

| | LLM | CV | 合计 |
|------|:---:|:---:|:---:|
| B类 | 4 | 4 | **8** |
| C类 | 6 | 5 | **11** |
| **总计** | **10** | **9** | **19** |

### 状态追踪

| 模型 | 类型 | 状态 |
|:-----|:---:|:----:|
| MobileNetV3-Small | Arc Model DSE 评估 | ✅ 完成 |
| MobileNetV3-Small | Func Model golden ref | ⏳ 进行中 |
| MobileNetV3-Small | 硬件验证 (im2col+SFU+Pool) | ⏳ 进行中 |

### 覆盖维度

| 维度 | 覆盖 |
|------|------|
| 任务类型 | 文本生成/推理链/分类/检测/分割 |
| 架构多样 | Dense FFN / SwishGLU / GQA / SWA / CNN / Transformer |
| 市场覆盖 | 中国(Qwen/DeepSeek) + 全球(Llama/Phi/Mistral/Gemma) |
| 规模梯度 | 2.5M (MobileNetV3) → 12B (Gemma-4) |
| MLPerf 对齐 | ResNet-18/50 + GPT-J + BERT(待评估) |

### 竞品覆盖对比

| | Hailo-8+10H | 算能 BM1684X | CaduceusCore |
|------|:---:|:---:|:---:|
| 芯片数 | 2 (分 CV/LLM) | 1 | **1** |
| LLM 模型数 | ~6 | ~5 | **10** |
| CV 模型数 | ~10 | ~10 | **9** |
| Transformer CV | ❌ | ❌ | **✅** |
| 国产模型 | Qwen, DeepSeek | Qwen, ChatGLM | Qwen, DeepSeek |
| 海外模型 | Llama, Phi, Gemma | Llama | Llama, Phi, Mistral, Gemma |
