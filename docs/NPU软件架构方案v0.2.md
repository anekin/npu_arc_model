# NPU 软件架构方案 v0.2

> 2026-06-18
> **v0.2 重大更新**：放弃 IREE 单一路线，改为两阶段方案（llama.cpp → ExecuTorch）
> 根因：IREE 的 Coral NPU HAL 后端未开源，原 22-33 周估算偏乐观。
> 重新调研后，llama.cpp 和 ExecuTorch 均有可参考的开源 NPU 后端。
> **目标硬件**：Block 64×64 NPU（28.2 mm²，~9.6 W，INT4 + LPDDR5-6400），详参 `Edge_NPU_Architecture_Proposal.md`

---

## 一、为什么从 IREE 切换到 llama.cpp + ExecuTorch

| 维度 | IREE（原方案 v0.1） | llama.cpp + ExecuTorch（新方案 v0.2） |
|------|:---|:---|
| NPU 参考实现 | ❌ Coral HAL 未开源 | ✅ NXP eIQ Neutron / Intel NPU 开源后端 |
| 自定义后端复杂度 | HAL 8 API（C） | Delegate 2 API（C++）/ ggml 5 函数（C） |
| 当前可用性 | 需等 22-33 周 | **阶段1: 4-8 周** / 阶段2: 6-12 周 |
| LLM 支持 | StabiliHLO → Linalg | **GGUF 原生** / PyTorch 生态 |
| 模型格式 | 需导出 StableHLO | **已有 GGUF** / PyTorch 原生 |
| 适合芯片量产后 | ✅ | ✅（ExecuTorch Delegate） |

**原方案 v0.1 的问题**：IREE 的 Coral NPU 工具链是 Google + Synaptics 闭源合作，开源仓库（`google-coral/coralnpu`）只包含 TFLite Micro + C++ 模拟器，不含 IREE HAL 后端。22 周的估算建立在「有 Coral HAL 可以 fork」的假设上，这个假设不成立。

---

## 二、两阶段方案总览

```
┌─ 阶段 1: llama.cpp NPU 后端 (4-8 周) ─────────────────────┐
│                                                             │
│  GGUF 模型 ─→ llama.cpp ─→ ggml NPU backend ─→ Python Model │
│  (已有)       (开源)        (5个C函数,自研)      (已有)      │
│                                                             │
│  目标: LLM decode 全链路跑通, 验证 ISA + 性能模型            │
│  产出: ggml NPU backend + 性能对比报告                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼  RTL 完成后
┌─ 阶段 2: ExecuTorch Delegate (6-12 周) ────────────────────┐
│                                                             │
│  PyTorch ─→ torch.export ─→ ExecuTorch NPU Delegate ─→ NPU  │
│  模型        (开源)          (C++ Delegate, 自研)    硬件    │
│                                                             │
│  目标: 产品级软件栈, 通用模型支持, 对标 Intel/NXP NPU        │
│  产出: NPU Delegate + 多核调度 + 量化管线                    │
└─────────────────────────────────────────────────────────────┘
```

**关键设计决策**：阶段 1 的 ggml backend 代码**不扔掉** — ExecuTorch Delegate 的 C++ 执行层可以直接复用 NPU ISA 生成 + DMA 描述符编排逻辑。

---

## 三、阶段 1 详细设计：llama.cpp NPU 后端

### 3.1 为什么选 llama.cpp

1. **模型已就绪**：用户硬盘上已有 Qwen GGUF 模型，零转换开销
2. **GGUF 是端侧 LLM 事实标准**：HuggingFace、Ollama、LM Studio 均支持
3. **backend 接口极简**：5 个 C 函数即可接入，比 IREE 的 8 个 HAL API 更简单
4. **LLM 推理链路完整**：tokenizer → context → sampling → decode，不需要自己搭
5. **社区验证**：CUDA/Metal/Vulkan/SYCL 后端均跑通，架构成熟

### 3.2 ggml backend 接口

```c
// 5 个必须实现的接口
struct ggml_backend_npu_context {
    // 1. 初始化 NPU 设备
    ggml_backend_t ggml_backend_npu_init(const char* device_path);
    
    // 2. 分配/释放 NPU 内存
    ggml_backend_buffer_t ggml_backend_npu_alloc_buffer(size_t size);
    void                  ggml_backend_npu_free_buffer(ggml_backend_buffer_t buf);
    
    // 3. 核心：提交计算
    //    graph 包含所有待执行的算子 (ggml_op)
    //    NPU backend 遍历 graph, 生成 NPU ISA, 下发执行
    bool ggml_backend_npu_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph *graph
    );
    
    // 4. 同步等待
    void ggml_backend_npu_synchronize(ggml_backend_t backend);
};
```

### 3.3 graph_compute 内部流程

```
ggml_cgraph (GGUF 计算图)
    │
    ▼
算子遍历: MUL_MAT → SOFTMAX → ROPE → ...
    │  提取 (M,K,N) + 权重地址
    ▼
NPUCompiler.compile_decode(trace)
    │  生成 NPU ISA 指令序列
    ▼
DMA 描述符链构建
    │  权重地址 → DRAM 物理地址映射
    ▼
NPUSimulator.run_instructions(program)
    │  虚拟 NPU 执行, 返回周期数
    ▼
读回输出 buffer → 返回给 llama.cpp
```

### 3.4 需要自研的部分

| 模块 | 工作量 | 说明 |
|------|:---:|------|
| ggml backend 框架 (init/alloc/free/sync) | 1 周 | 参考 Metal/Vulkan backend |
| graph_compute: 算子遍历 → trace | 1 周 | 映射 ggml_op → (M,K,N) |
| NPU ISA 生成 + DMA 描述符 | 1 周 | **复用现有 NPUCompiler** |
| Python model 对接 (C→Python bridge) | 1-2 周 | pybind11 或 Unix socket IPC |
| 集成测试 + 性能对比 | 1-2 周 | vs CPU Metal backend |
| **阶段 1 合计** | **4-8 周** | |

### 3.5 C → Python Model 通信方案

```
┌─────────────┐     Unix Socket / pybind11      ┌──────────────┐
│  llama.cpp   │ ───────────────────────────────→│  Python Model │
│  (C/C++)     │  ISA program (binary)           │  (虚拟 NPU)   │
│              │ ←───────────────────────────────│              │
│              │  cycles + output buffer          │              │
└─────────────┘                                  └──────────────┘
```

两种实现：
- **pybind11**：Python model 编译为 C++ 可调用库，零通信开销
- **Unix Socket**：独立进程，易于调试，性能损耗 <1%（传输 ISA 二进制 <10KB）

---

## 四、阶段 2 详细设计：ExecuTorch NPU Delegate

### 4.1 为什么是 ExecuTorch 而不是 IREE

| | ExecuTorch | IREE |
|---|---|---|
| 可参考的开源 NPU | ✅ Intel NPU / NXP eIQ Neutron | ❌ Coral HAL 未开源 |
| 后端接口 | Delegate (init + execute) | HAL (8 API) |
| 量化框架 | ✅ 内置 INT4/BF16 | 需额外集成 |
| PyTorch 生态 | ✅ 原生 | 需 torch-mlir 桥接 |
| LLM 部署流程 | ✅ 有文档和示例 | ❌ 需自己摸索 |
| 芯片厂商采用 | Intel / NXP / ARM / Qualcomm | Google Pixel TPU (闭源) |

### 4.2 Delegate 接口

```cpp
// ExecuTorch NPU Delegate — 核心只需2个函数
class NPUBackend : public BackendDelegate {
public:
    // 1. 初始化: 加载 .pte 模型, 分配 NPU buffer
    Result<int> init(
        const void* processed,   // 编译好的 NPU 子图
        size_t processed_size,
        FreeableBuffer* input_buffers,
        FreeableBuffer* output_buffers
    );
    
    // 2. 执行: 提交 NPU ISA → 等待完成
    Result<int> execute();
};
```

**对比 IREE HAL**：IREE 需要实现 8 个 API（driver/device/buffer/command_buffer/executable/semaphore/submit/query），ExecuTorch 只需 2 个。

### 4.3 需要自研的部分

| 模块 | 工作量 | 说明 |
|------|:---:|------|
| Delegate 框架 (init/execute) | 1-2 周 | 参考 Intel NPU Delegate |
| AoT 编译器 (PyTorch → NPU 子图) | 2-3 周 | torch.export + partitioner |
| NPU ISA 代码生成 | 1-2 周 | **复用阶段 1 成果** |
| 多核调度 | 1-2 周 | **复用 MultiCoreTimeline** |
| 量化管线 (INT4/BF16) | 1-2 周 | 参考 llama.cpp 量化实现 |
| 集成测试 | 1-2 周 | |
| **阶段 2 合计** | **6-12 周** | |

---

## 五、与 Python Model 的配合

两个阶段共享同一套虚拟 NPU 接口，只是通信方式不同：

```
阶段 1: llama.cpp (C) ──→ ggml backend ──→ pybind11/socket ──→ Python Model
阶段 2: ExecuTorch (C++) ──→ NPU Delegate ──→ AXI4 寄存器 ──→ 真实 NPU
                                                    │
                                          Python Model (验证阶段)
```

Python Model 只需要暴露一个接口：

```python
# models/npu_device.py
class NPUDevice:
    def execute(self, isa_program: bytes) -> Tuple[int, bytes]:
        """执行 NPU ISA 程序, 返回 (cycles, output_buffer)"""
        ...
```

阶段 1 通过 socket/pybind11 调用此接口，阶段 2 直接操作硬件寄存器 — 接口签名不变。

---

## 六、工作量汇总

| 阶段 | 模块 | 工作量 | 优先级 |
|------|------|:---:|:---:|
| **阶段1** | ggml NPU backend 框架 | 1 周 | P0 |
| | ggml_op → trace 映射 | 1 周 | P0 |
| | ISA 生成 + DMA 编排 | 1 周 | P0 |
| | C→Python bridge | 1-2 周 | P0 |
| | 集成测试 | 1-2 周 | P1 |
| | **阶段1 小计** | **4-8 周** | |
| **阶段2** | ExecuTorch Delegate 框架 | 1-2 周 | P1 |
| | AoT 编译器 (partitioner) | 2-3 周 | P1 |
| | ISA 代码生成 (复用阶段1) | 1-2 周 | P1 |
| | 多核调度 | 1-2 周 | P1 |
| | 量化管线 | 1-2 周 | P1 |
| | 集成测试 | 1-2 周 | P2 |
| | **阶段2 小计** | **6-12 周** | |
| | **合计** | **10-20 周** | |

对比原方案：**22-33 周 → 10-20 周**，且阶段 1 产出（llama.cpp 后端）RTL 出来之前就能用。

---

## 七、与原方案 v0.1 的差异总结

| | v0.1 (IREE) | v0.2 (llama.cpp + ExecuTorch) |
|---|---|---|
| 时间段 | 22-33 周 | **10-20 周** |
| 阶段 1 可用性 | 无（需等 IREE 全栈） | **4-8 周**（llama.cpp + Python model） |
| 可参考的开源 NPU 后端 | 无 | **Intel NPU / NXP eIQ / 多个** |
| LLM 验证 | 需先搭 Torch-MLIR | **GGUF 直接可用** |
| 最终产品栈 | IREE HAL | **ExecuTorch Delegate** |
| PyTorch 生态 | 需要 torch-mlir 桥接 | **原生 PyTorch** |

---

> **文档版本**：v0.2 | **变更**：IREE 路线 → llama.cpp (阶段1) + ExecuTorch (阶段2)
> **下一步**：实现 ggml NPU backend，跑通 Qwen2.5-3B GGUF → ISA → Python model 全链路
