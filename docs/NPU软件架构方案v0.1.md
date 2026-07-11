# NPU 软件架构方案 v0.1

> ⚠️ **DEPRECATED — 本方案已由 v0.2 替代**
> 请参阅 `docs/NPU软件架构方案v0.2.md`。
> v0.2 将策略从 IREE/MLIR（本方案）调整为两阶段方案：
>   阶段 1：GGUF → llama.cpp → ggml NPU backend → Python Model
>   阶段 2：PyTorch → ExecuTorch NPU Delegate → NPU 硬件
> 本文件保留仅供历史参考。
>
> 2026-06-17
> 配套硬件架构 v0.2：NPU 作为 IP 核，支持单核/多核实例化

---

## 一、为什么选 MLIR/IREE 而不是 TVM 或自研

| 维度 | TVM | 自研编译器 | **IREE** |
|------|:---:|:---:|:---:|
| 成熟度 | 高（Apache 顶级） | — | 高（Google/LLVM 社区） |
| 自定义后端 | BYOC 机制 | 全力 | **HAL 接口** |
| Transformer LLM 优化 | 弱 | — | **强**（Coral NPU 已验证） |
| 多设备支持 | 有 | — | **原生多设备 HAL** |
| RISC-V 后端 | 弱 | — | **有**（LLVM RISC-V 后端） |
| NPU 参考实现 | 无 | — | **Coral NPU 完整栈** |
| 开源许可 | Apache 2.0 | — | Apache 2.0 |
| 上手周期 | 3-6 月 | 12-18 月 | **2-3 月**（有 Coral 参考） |

**选 IREE 的理由**：
1. Coral NPU 已经在 IREE 上跑通——Google 和 Synaptics 2025 年发布的完整栈
2. HAL 抽象层天然支持自定义加速器后端——我们只需实现 HAL 接口
3. MLIR 多级方言允许在合适的层级做优化——Linalg 层做算子融合，HAL 层做内存管理
4. 多核扩展天然支持——`iree_hal_device_query_info()` 枚举核心数

---

## 二、整体软件栈

```
┌────────────────────────────────────────────┐
│            用户模型 (PyTorch/HuggingFace)    │
│            3B LLM / YOLOv8 / ResNet         │
└──────────────────┬─────────────────────────┘
                   │ torch.export / ONNX
                   ▼
┌────────────────────────────────────────────┐
│  Layer 1: Torch-MLIR / StableHLO            │
│  · PyTorch FX Graph → MLIR 方言              │
│  · 输入：PyTorch nn.Module                   │
│  · 输出：StableHLO / TOSA MLIR              │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Layer 2: IREE Compiler (iree-compile)      │
│  · StableHLO → Linalg → Vector → HAL        │
│  · 图优化：算子融合、常量折叠、死代码消除      │
│  · 量化：INT4 权重 + BF16 激活 代码生成       │
│  · 输入：StableHLO MLIR                      │
│  · 输出：.vmfb (IREE VM Flat Buffer)         │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Layer 3: IREE HAL (Hardware Abstraction)   │
│  · 设备发现 & 能力查询                        │
│  · 命令缓冲区 (Command Buffer) 录制            │
│  · 内存管理 (Buffer Alloc/Map)                │
│  · 多核调度                                  │
│  · 输入：.vmfb                               │
│  · 输出：HAL 命令流 → NPU ISA 指令序列          │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Layer 4: NPU Driver (Kernel/Userspace)      │
│  · PCIe 通信                                  │
│  · DMA 描述符链下发                           │
│  · 中断处理                                  │
│  · 输入：HAL 命令流                           │
│  · 输出：AXI4 寄存器读写 → NPU 芯片             │
└──────────────────┬─────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────┐
│  Layer 5: NPU Firmware (RISC-V 核上)         │
│  · 指令 FIFO 监控                            │
│  · DMA 状态轮询                              │
│  · 异常上报                                  │
│  · 输入：HAL 下发的 NPU ISA 指令               │
│  · 输出：MXU/SFU/DMA 控制信号                  │
└────────────────────────────────────────────┘
```

---

## 三、逐层详细设计

### 3.1 Layer 1：模型导入（Torch-MLIR）

**做什么**：把 PyTorch 模型转成 MLIR 中间表示

**输入**：
```python
model = AutoModelForCausalLM.from_pretrained("Qwen2.5-3B")
# 或
model = torch.hub.load('ultralytics', 'yolov8n')
```

**输出**：StableHLO MLIR 文本
```mlir
module {
  func.func @main(%arg0: tensor<1x4096xbf16>) -> tensor<1x4096xbf16> {
    %0 = stablehlo.dot_general %arg0, %weight ...
    %1 = stablehlo.add %0, %bias ...
    return %1
  }
}
```

**复用**：完全开源。Torch-MLIR 是 LLVM 孵化项目，PyTorch 基金会支持。**我们的工作量为零。**

---

### 3.2 Layer 2：IREE 编译（iree-compile）

**做什么**：MLIR 多级优化，生成目标平台的 VM 字节码

**优化 pipeline**：
```
StableHLO → Linalg-on-Tensors (算子融合)
         → Vector (向量化)
         → SPIR-V / HAL (硬件抽象)
         → VM Bytecode (.vmfb)
```

**关键优化**：
- **算子融合**：QKV 投影 3 个矩阵乘 → 1 个融合 GEMM
- **INT4 编解码插入**：权重解包 + 反量化 → BF16 的逻辑自动插入
- **内存规划**：L1/L2/LPDDR5 三级缓冲分配

**命令行示例**（用户侧）：
```bash
iree-compile \
  --iree-hal-target-backends=npu \
  --iree-npu-cores=4 \
  --iree-npu-l1-kb=512 \
  --iree-npu-l2-kb=4096 \
  model.stablehlo.mlir \
  -o model.vmfb
```

**复用**：IREE 编译器本体上游开源。**我们的工作**：实现 `--iree-hal-target-backends=npu` 插件。

**工作量：~4-6 周**

---

### 3.3 Layer 3：HAL 后端（核心自研）

**做什么**：把编译好的 VM 字节码翻译成 NPU ISA 指令序列，管理设备生命周期

#### HAL 接口（参考 Coral NPU 实现）

| 接口 | 功能 | 实现要点 |
|------|------|---------|
| `iree_hal_npu_driver_create()` | 设备发现 | PCIe 枚举 NPU 设备 |
| `iree_hal_npu_device_query_info()` | 能力查询 | 返回核数、L1/L2 大小、FIFO 有无 |
| `iree_hal_npu_buffer_allocate()` | 内存分配 | LPDDR5 上的输入/输出/权重 buffer |
| `iree_hal_npu_buffer_map()` | 内存映射 | CPU 可访问的 DMA 缓冲区 |
| `iree_hal_npu_command_buffer_begin()` | 命令录制开始 | 创建 ISA 指令序列 |
| `iree_hal_npu_executable_create()` | 可执行对象 | 将 .vmfb 中的 kernel 绑定到 NPU ISA |
| `iree_hal_npu_semaphore_create()` | 信号量 | 多核同步 |
| `iree_hal_npu_submit_and_wait()` | 提交执行 | DMA 下发指令 → 等待中断 |

#### 核心流程

```
submit_and_wait(command_buffer):
    1. 将 ISA 指令序列打包为 DMA 描述符链
    2. 通过 PCIe BAR 空间写入 NPU 指令 FIFO
    3. 写 RISC-V 控制寄存器 → 触发执行
    4. 等待中断 (irq_handler)
    5. 读回输出数据 (通过 PCIe DMA)
```

#### 多核支持（透明）

```c
// 始终只创建一个设备——核数对应用透明
iree_hal_device_t* dev;
iree_hal_npu_create_device(NULL, &dev);

// 内部自动查询核数并分区
// 运行 3B 模型 → 数据并行，N 核并发处理 N 个请求
// 运行 7B 模型 → 流水线并行，层自动分配到各核

iree_hal_device_submit_and_wait(dev, cmd);  // 一行代码，HAL 处理所有多核细节

// 高级调试接口（可选）
iree_hal_npu_query_info(dev, &info);
// info.num_cores = 4
// info.compute_tops = 26.4  ← 4 核算力 (Block 64×64)
```

**工作量：~8-12 周**（HAL 后端是最大工作量）

---

### 3.4 Layer 4：NPU Driver

**做什么**：PCIe 通信、DMA 管理、中断处理

| 模块 | 功能 | 实现 |
|------|------|------|
| PCIe 驱动 | BAR 空间映射、MSI-X 中断 | Linux kernel module |
| DMA 引擎 | 描述符链下发、传输状态轮询 | 操作 MMIO 寄存器 |
| 用户态库 | ioctl 接口、buffer 管理 | libnpu.so |

**复用**：参考 Coral NPU 的开源 Linux 驱动（`gasket-dkms`）。PCIe 通信层是标准化程度最高的部分。

**工作量：~3-4 周**

---

### 3.5 Layer 5：NPU Firmware

**做什么**：运行在 RISC-V 核上的轻量固件

```c
void main() {
    while (1) {
        // 检查指令 FIFO 是否有新指令
        if (reg_read(INSTR_FIFO_STATUS) & FIFO_NOT_EMPTY) {
            instr = reg_read(INSTR_FIFO_DATA);
            dispatch(instr);  // 分派到 MXU/SFU/DMA
        }
        // 检查 DMA 完成中断
        if (irq_pending & IRQ_DMA_DONE) {
            update_dma_state();
            clear_irq(IRQ_DMA_DONE);
        }
        // 检查 MXU 完成中断
        if (irq_pending & IRQ_MXU_DONE) {
            signal_host();  // 通知主机
            clear_irq(IRQ_MXU_DONE);
        }
    }
}
```

**工作量：~2-3 周**（固件极简，不做复杂调度）

---

## 四、总线数据流（端到端）

以一次 LLM decode 为例：

```
主机 CPU:
  ① tokenizer.encode("今天天气") → [101, 791, 1921, 3698]
  ② iree_hal_npu_buffer_map(output_buffer)  // 准备输出
  ③ iree_hal_npu_command_buffer_begin()
  ④ 写 ISA 指令：DMA_LOAD(W_attn) → MMUL → SOFTMAX → ... → DMA_STORE → IRQ
  ⑤ iree_hal_npu_submit_and_wait()

  ↓ PCIe DMA 下发给 NPU ↓

NPU (RISC-V 核):
  ⑥ dispatch(DMA_LOAD) → 启动 DMA
  ⑦ dispatch(MMUL)     → 触发 MXU
  ⑧ dispatch(SOFTMAX)  → 触发 SFU
  ⑨ dispatch(DMA_STORE)→ 结果写回 LPDDR5
  ⑩ IRQ → 主机

  ↑ PCIe 中断 ↑

主机 CPU:
  ⑪ iree_hal_npu_buffer_map(output_buffer)
  ⑫ tokenizer.decode(output) → "不错"
```

完整一次 decode 的主机→NPU→主机 往返延迟 < 100μs（不含计算）。

---

## 五、模型部署流水线

### 5.1 开发者视角

```bash
# Step 1: 导出 PyTorch 模型为 MLIR
python export.py --model Qwen2.5-3B --quant int4 --output model.stablehlo.mlir

# Step 2: IREE 编译为 NPU 字节码
iree-compile --iree-hal-target-backends=npu model.stablehlo.mlir -o model.vmfb

# Step 3: 部署到 NPU
iree-run-module --device=npu --module=model.vmfb --input="今天天气怎么样？"
```

### 5.2 生产部署

```c
// C API，嵌入式集成
iree_hal_device_t* device;
iree_hal_npu_driver_create(iree_allocator_system(), &driver);
iree_hal_driver_create_device(driver, &device);

// 加载编译好的模型
iree_vm_module_t* module;
iree_vm_bytecode_module_create(device, vmfb_data, vmfb_size, &module);

// 推理循环
while (1) {
    iree_hal_buffer_view_t* input = tokenize(user_input);
    iree_hal_buffer_view_t* output;
    iree_vm_invoke(module, "generate", input, &output);
    printf("%s\n", detokenize(output));
}
```

---

## 六、多核软件支持

### 6.1 设计原则：多核 = 统一设备，更高算力

多核对软件透明。应用永远只看到一个 NPU 设备对象。核数增加只改变算力容量，不改变编程模型。

| 核数 | 应用层看到的 | 行为 |
|:---:|------|------|
| 1 | 1 设备，~6.6 TOPS | 3B: 29.6 tok/s |
| 2 | 1 设备，~13.2 TOPS | 3B 数据并行 59.2 tok/s / 7B 流水线 ~26 tok/s |
| 4 | 1 设备，~26.4 TOPS | 3B 数据并行 118.4 tok/s / 7B 流水线 ~52 tok/s |

**应用代码零改动。** 单核怎么用，多核就怎么用。跟 GPU 的 SM 数量一样——用户不关心里面有几个 SM。

### 6.2 HAL 层自动分区

| 策略 | 何时用 | HAL 如何做 |
|------|------|-----------|
| **按层均分** | 流水线并行 | `layers_per_core = total_layers / num_cores` |
| **按内存均分** | 大模型分片 | 每核加载 ≤ L1 SRAM 大小的权重分片 |
| **全复制** | 数据并行 | 每核加载完整模型 |

### 6.3 核间通信（HAL 自动管理）

```
应用层：    iree_hal_device_submit_and_wait(dev, cmd)  // 一行

HAL 自动：
  Core₀ ← Layer 0-15          ─→ FIFO ─→ Core₁ ← Layer 16-31
  Core₂ ← 另一个请求 Layer 0-31  独立运行（数据并行）

应用层完全无感知核间 FIFO 的存在。
```

---

## 七、工作量汇总

| 模块 | 复用源 | 工作量 | 优先级 |
|------|------|:---:|:---:|
| Torch-MLIR 适配 | 上游开源 | 0 周 | — |
| IREE 编译器插件 | IREE 上游 | 4-6 周 | P1 |
| **HAL 后端** | Coral NPU 参考 | **8-12 周** | **P0** |
| NPU Driver | Gasket 驱动 | 3-4 周 | P1 |
| NPU Firmware | 自研 | 2-3 周 | P2 |
| 模型量化工具 (INT4) | llama.cpp/llm-awq 参考 | 3-4 周 | P1 |
| 端到端集成测试 | — | 2-4 周 | P2 |
| **软件栈合计** | | **22-33 周** | |

与硬件 Phase 2（RTL 16-24 周）完全并行。

---

## 八、与 Coral NPU 软件栈的复用关系

| Coral NPU 组件 | 我们的复用方式 | 改动 |
|------|------|:---:|
| **IREE 编译器流程** | 完全相同 | 无 |
| **Coral HAL 后端** | Fork 后修改寄存器地址、设备查询 | 中 |
| **Coral ISA 编码** | 替换为我们的 32-bit NPU ISA | 中 |
| **Gasket PCIe 驱动** | 修改 Vendor ID / Device ID | 小 |
| **RISC-V Firmware** | 参考调度循环，替换指令集 | 中 |
| **模型部署脚本** | 相同 | 无 |

> Coral NPU 的 IREE 工具链在 Synaptics Torq 上已量产验证——我们直接站在谷歌的肩膀上。

---

## 九、下一步

1. 搭建 IREE 开发环境 → Fork Coral NPU HAL 后端 → 改为我们的 ISA
2. 在 QEMU RISC-V 上验证 Firmware 调度循环
3. 等 RTL 完成后，HAL 后端对接真实 NPU 寄存器

---

> **文档版本**：v0.1 | **下一步**：SCALE-Sim v3 性能建模（硬件）+ IREE HAL 后端原型开发（软件）
