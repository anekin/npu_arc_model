# 具身智能与物理 AI 场景的 NPU 协处理器需求

日期：2026-07-23
状态：市场调研后的需求基线，供场景建模和 DSE 使用

## 1. 结论

当前应用需求不应继续用一个“7B VLM/VLA、10–20 个输出 token、
10 FPS”的场景同时代表具身智能和物理 AI。

市场方案至少形成四类不同负载：

1. 紧凑型 VLA：约 0.45B，多图像输入、连续动作块、异步补充动作队列。
2. 通用连续动作 VLA：约 3.3B，2–3 路图像、H=50 动作块、
   10 次 flow-matching 迭代。
3. 人形机器人双速率策略：7B 慢速语义模型与 80M、200 Hz 快速策略并行。
4. AMR/工业/车辆物理 AI：多个 10–30 FPS 周期性视觉任务并发，
   重点是 P99、抢占、带宽隔离、ECC 和故障恢复。

本项目中的 NPU 是主控 CPU 的协处理器。摄像头、雷达、激光雷达、
编码器、ISP、传感器同步以及最终电机控制接口均不属于 NPU 硬件需求。
NPU 接收的是主控已经准备好的、带时间戳和期限的 tensor job。

因此，DSE 的一级约束应从单一 TOPS/FPS 改为：

- 完整的子模型/算子图；
- tensor 输入输出大小和到达周期；
- 策略推理频率、动作执行频率和动作块长度；
- P50/P95/P99/WCET、输入新鲜度和超时行为；
- 多任务并发、优先级、抢占粒度和资源隔离；
- 权重、激活、缓存、workspace、DMA buffer 的完整容量；
- 精度、稀疏性、功耗、温度和任务成功率约束。

## 2. 系统边界

### 2.1 主控 CPU 负责

- 摄像头和其他传感器接口、采集、时间同步、标定选择；
- ISP/编解码/ROI 选择，以及未下沉到 NPU graph 的预处理；
- 模型生命周期、graph 选择、buffer pool 注册和 admission control；
- 任务依赖、绝对期限、健康监控、结果校验和降级策略；
- 轨迹融合、复杂分支逻辑、最终 actuator gating；
- 与独立安全 MCU/实时控制器协同。

### 2.2 NPU 负责

- 执行离线编译、shape 有界的 tensor graph；
- 对注册的共享 buffer 做 DMA 和计算；
- 维护有界的异步任务队列；
- 按期限、优先级和 criticality 调度；
- 返回 completion fence、执行时间戳和错误状态；
- 提供 context/core/device 级 watchdog、隔离和复位能力。

生产 NVIDIA DRIVE、Qualcomm、TI Jacinto 以及 ROS 2/Autoware 软件栈都采用
类似的“主控为控制平面、加速器为异步数据平面”模式。公开机制包括
预注册共享内存、显式 fence、有界 FIFO 或 latest-wins 队列，以及独立安全
监督。[NVIDIA DriveWorks CGF](https://developer.nvidia.com/docs/drive/drive-os/6.0.8/public/driveworks-nvcgf/nvcgf_html/cgf_execution.html)、
[Qualcomm FastRPC](https://github.com/qualcomm/fastrpc)、
[TI TIOVX 内存管理](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/10_01_00_04/exports/docs/tiovx/docs/user_guide/TIOVX_MEMORY_MANAGEMENT.html)、
[ROS 2 zero-copy 设计](https://design.ros2.org/articles/zero_copy.html)。

## 3. 具身智能与物理 AI 对 NPU 需求的区别

| 维度 | 具身智能/VLA | 广义物理 AI |
|---|---|---|
| 核心任务 | 视觉、语言、状态到动作 | 感知、定位、检测、规划辅助、状态估计 |
| 典型模型 | VLM + action expert；AR、flow 或 diffusion | CNN/Transformer/BEV/点云等多模型 DAG |
| 时间结构 | 动作块、异步 refill、慢思考+快策略 | 多个 10–30 FPS 周期任务并发 |
| 主要容量压力 | 0.45B–7B 权重和多模态缓存 | 多个驻留模型、激活和中间 tensor |
| 主要调度压力 | 1–5 ms 快路径不能被 7B 慢任务阻塞 | 多 criticality job 的 P99 和抢占 |
| 关键正确性指标 | task success、动作误差、稳定性 | 检测/分割/定位精度、漏检、过期率 |
| 关键安全边界 | NPU 输出动作目标，MCU 决定执行 | NPU 输出感知/规划信息，安全域做最终裁决 |

具身智能是物理 AI 的一个重要子集，但两者不能使用同一个平均 FPS 场景。

对 Figure、Digit、Unitree、Fourier、UBTECH、PAL 等产品/SDK 的一手资料
对比后，公开频率大致落在三个不同层级：语义/VLA 约 3–15 Hz，learned
joint-target/reactive policy 约 50–250 Hz，状态/总线/actuator loop 约
500 Hz–2 kHz。最后一层通常属于运动主控或 MCU，不能倒推成 NPU 必须完成
500–2000 次完整 VLA 推理。[Figure Helix](https://www.figure.ai/news/helix)、
[UBTECH Tienkung](https://docs.ubtrobot.com/walker-tienkung/en/docs/user-guide/6/)、
[PAL TALOS](https://pal-robotics.com/datasheet/talos/)。

## 4. 市场方案给出的边界

### 4.1 模型与运行时

- SmolVLA 为 450M，运行时把每帧压缩为 64 个视觉 token，action expert
  约 100M；当前公开配置使用 512×512、H=50 和 10 次 flow step，
  异步运行时在当前动作块执行期间预测下一块。
  [Hugging Face SmolVLA](https://huggingface.co/blog/smolvla)
- π0 使用 3B PaliGemma 和 300M action expert，总计 3.3B；输入为
  2–3 路 RGB、语言和本体状态，H=50，做 10 次 flow-matching 迭代，
  并缓存 observation prefix 的 KV。对于 50 Hz 机器人，论文每执行
  25 个动作，即每 0.5 秒做一次推理，而不是每秒做 50 次完整推理。
  [π0 论文](https://www.physicalintelligence.company/download/pi0.pdf)
- Figure Helix 的 7B System 2 运行在 7–9 Hz，80M System 1 运行在
  200 Hz，二者异步共享最新 latent。Helix 02 又增加 10M、1 kHz 的
  System 0，但最终关节电流/位置/安全环仍应放在实时控制器。
  [Figure Helix](https://www.figure.ai/news/helix)、
  [Helix 02](https://www.figure.ai/news/helix-02)
- OpenVLA 为 7B，原始版本没有动作块，官方仓库建议约 5–10 Hz
  数据/控制频率；后续 OFT/FAST 路径正是为降低自回归动作延迟。
  [OpenVLA](https://github.com/openvla/openvla)

这组证据说明必须把 `policy_inference_hz`、`action_execution_hz` 和
`action_chunk_horizon` 分开。

### 4.2 协处理器和平台

| 方案 | 官方公开值 | 对本项目的意义 |
|---|---|---|
| Hailo-10H | 20 INT8 / 40 INT4 TOPS，2.5 W typical，LPDDR4/4X，PCIe/USB host co-processor | 最接近本项目边界的离散 NPU 参考 |
| Cambricon MLU220 M.2 | 8 INT8 TOPS，PCIe 3.0 x2，8.25 W passive | 国内低功耗 PCIe 加速卡参考 |
| Axera AX8850 M.2 | 24 INT8 TOPS，M.2 2242/2280，低于 8 W | 具身视觉/运动控制定位的国内参考 |
| Qualcomm IQ-9075 | 50/100 dense INT8 TOPS，最高 36 GB ECC LPDDR5，双 tensor processor | 中高端并发机器人平台参考 |
| TI AM69A | 32 INT8 TOPS，4 个 C7x/MMA，最高 68 GB/s，ECC | 多周期视觉任务和 AMR 参考 |
| Jetson AGX Orin 64GB | GPU 85 dense INT8 TOPS；整个平台另有 275 sparse INT8 TOPS 口径，204.8 GB/s，15–60 W | 说明 precision、sparsity、scope 必须随 TOPS 保存 |
| Jetson Thor T5000 | 2070 sparse FP4 TFLOPS，128 GB，273 GB/s，40–130 W | 高端 physical-AI 平台，不能与 dense INT8 直接比较 |

来源：
[Hailo-10H product brief](https://hailo.ai/files/hailo-10h-product-brief-en/)、
[MLU220 M.2](https://www.cambricon.com/index.php?a=lists&c=index&catid=57&m=content)、
[Axera AX8850](https://www.axera-tech.com/en/news/2989.html)、
[Qualcomm IQ9 brief Rev. G](https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_G_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf)、
[TI AM69A](https://www.ti.com/product/AM69A)、
[Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)、
[Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)。

所有市场 TOPS 必须存成：

```yaml
peak_compute:
  value: 20
  unit: TOPS
  precision: INT8
  sparsity: dense
  scope: npu_only
```

不能把 FP4 sparse、INT8 dense、`effective/equivalent TOPS` 混成一个数排序。

## 5. 建议的四个 DSE 场景

以下“市场固定项”来自公开方案；“验收建议”是本项目的工程基线，
不是厂商性能承诺。最终架构应由完整 workload trace 推导，不应仅按
候选 TOPS 区间选型。

### 5.1 E0：紧凑型 VLA

市场固定项：

- 参考模型：SmolVLA 450M；
- 多 RGB 图像 + 语言 + robot state；
- 每帧 64 visual tokens；
- 约 100M flow-matching action expert；
- 当前公开 checkpoint/config 的 H=50、flow steps=10；
- 异步 action-chunk refill。

验收建议：

| 项目 | 建议 |
|---|---|
| 权重精度 | INT4 和 INT8 都进入 DSE，激活 INT8/FP16 |
| policy trigger | 基线为动作队列 low-watermark 触发；等效 1/2/5/10 Hz 扫描，10 Hz 为压力点 |
| action sweep | 20/30/50 Hz，与 policy rate 分开 |
| deadline | `P99 <= 0.75 * refill_window`，hard timeout 不超过队列耗尽时间 |
| 本地内存 | 从 4/8 GB 扫描；公开参考运行时约需 2 GB，完整 footprint 仍须至少 20% 余量 |
| NPU 功耗搜索域 | 3–10 W |
| 计算搜索域 | 8–24 dense INT8 TOPS，仅用于生成候选 |

公开 LeRobot 异步部署文档给出的参考推理显存约为 SmolVLA 2 GB、
π0 14 GB；它们是相应软件栈的运行时参考，不是本项目定制 INT4 NPU 的
固定值。[LeRobot async inference](https://huggingface.co/docs/lerobot/async)

### 5.2 E1：通用连续动作 VLA

市场固定项：

- 参考模型：π0 3.3B；
- 2–3 路 RGB + language + proprioception；
- H=50；
- 10 次 flow-matching action forward；
- 50 Hz 动作执行时，公开实现每 0.5 秒补充一次动作块；
- RTX 4090 论文参考：image encoder 14 ms、observation forward 32 ms、
  10 次 action forward 合计 27 ms，总模型时间 73 ms。该值不是 NPU 保证。

验收建议：

| 项目 | 建议 |
|---|---|
| policy rate | 基线 2 Hz，并扫描更积极的 5/10 Hz 闭环模式 |
| action rate | 20/50 Hz |
| chunk | H=50；记录实际执行多少 action 后 refill |
| deadline | 2 Hz 基线下 P99 ≤250 ms、hard ≤500 ms；均为工程 SLO |
| 本地内存 | 8/16/24 GB 扫描；公开 LeRobot 参考运行时约需 14 GB。若自研 INT4 图要落到 8 GB，必须由实测 footprint 和精度证明 |
| 计算搜索域 | 20–100 dense INT8 TOPS |
| 关键优化 | prefix KV/cache、action suffix 重算、flow-step fusion |

### 5.3 E2：人形机器人双速率策略

市场固定项：

- S2：7B，7–9 Hz；
- S1：80M，200 Hz；
- latest-latent 异步通信；
- 可选 S0：10M，1 kHz；只有 NPU 能证明 1 ms WCET 时才允许进入 NPU，
  否则下沉到独立实时 AI/控制域。

验收建议：

| 项目 | 建议 |
|---|---|
| S1/action output | 输出 period=5 ms；公开资料未披露 S1 action chunk 长度。只有确认每 tick 都完整推理时，才采用 P99 ≤4 ms、hard=5 ms |
| S2 | period=111–143 ms，9 Hz 点 P99 ≤100 ms |
| 抢占 | 低优先级最大 non-preemptible segment ≤1 ms |
| 隔离 | 双 engine/partition 优先；共享 engine 必须做 response-time analysis |
| mailbox | S2→S1 为 latest-value，不允许 FIFO 积压 |
| 权重驻留 | S1 永久驻留；S2 不得通过换模阻塞 S1 |
| 本地内存 | 扫描 8/16/24/32 GB；5 GB 仅作为激进 INT4 可行性点，不能在没有完整 footprint 时作为产品基线 |
| 计算搜索域 | 50–150 dense INT8 TOPS |

80M S1 在 200 Hz 下，按每个参数每次至少使用一次估算，下界为
16 GMAC/s；若每次从本地 DRAM 读取全部 INT8 权重，则仅权重流量下界
就是 16 GB/s。容量、驻留和最坏干扰比平均 TOPS 更关键。

### 5.4 P0：持续感知物理 AI

TI 的公开 AM69A 参考负载包括：

- 12 路 2 MP、30 FPS 输入，12 FPS 推理，约 12 aggregate TOPS；
- 3 路 8 MP、30 FPS 输入，ROI 推理 10–30 FPS，约 24 TOPS；
- 8 路 2 MP、30 FPS 输入，30 FPS 检测，约 24 TOPS。

这些是 SoC 级参考，不应把摄像头接口复制到 NPU。应转换为 CPU 输出的
tensor DAG、`bytes/job`、到达周期和期限。
[TI AM69A workload white paper](https://www.ti.com/lit/wp/spradb4/spradb4.pdf)

验收建议：

| 项目 | 建议 |
|---|---|
| 驻留模型 | 4–8 |
| inflight jobs | ≥16 |
| critical perception | period=33.3 ms，P99 ≤25 ms，hard ≤33.3 ms |
| localization/fusion | period=50 ms，P99 ≤37.5 ms |
| inspection/auxiliary | period=83–100 ms，P99 ≤62.5–75 ms |
| 抢占 | 最大 non-preemptible segment ≤1–2 ms |
| 带宽准入 | 初始不超过实测 sustained BW 的 70–75%，用 peak bytes 而非 average |
| 可靠性 | ECC/parity、IOMMU 隔离、per-context watchdog、分级复位 |
| 计算搜索域 | 20–100 dense INT8 TOPS |
| 本地内存 | 4/8/16 GB 扫描，按完整多模型 footprint 判定 |

上述 P99 是工程验收基线。公开平台通常不给出混合工作负载的 P99 保证，
必须在目标芯片上用长时间并发测试闭环。

## 6. CPU 到 NPU 的 workload contract

### 6.1 初始化期

```yaml
contract:
  version: 1
  model_id: ...
  model_hash: ...
  compiled_target: ...
  quantization_profile: ...
  tensors:
    - name: image_0
      direction: input
      dtype: fp16
      shape: [1, 3, 512, 512]
      layout: NCHW
      bytes: 1572864
  buffer_pool:
    slots: 4
    alignment_bytes: 4096
    memory_domain: dma_buf
    cache_policy: coherent_or_explicit_sync
```

### 6.2 每个任务

```yaml
job:
  stream_id: ...
  frame_id: ...
  capture_time_ns: ...
  submit_deadline_ns: ...
  completion_deadline_ns: ...
  stale_after_ns: ...
  priority: ...
  criticality: ...
  queue_policy: mailbox_latest  # 或 fifo
  input_buffers: [...]
  output_buffers: [...]
  acquire_fences: [...]
```

必须保证：

- 传递 handle/offset/fence，不跨进程传裸指针；
- 每个 accepted job 最终返回唯一 terminal completion；
- cancel 不等于 buffer 已安全释放；
- camera/live perception 默认 latest-wins，stateful history 才用 FIFO；
- 过期输入在执行前拒绝，不能形成无界 backlog；
- CPU 验证输出 shape、范围、finite、时间戳和业务不变量后才能发布。

NPU 可用期限为：

`D_npu = D_sensor_to_result - T_cpu_pre - T_host_transfer - T_cpu_post - T_safety_guard`

对 action chunk：

`D_refill <= executed_actions / action_execution_hz - T_transfer - T_guard`

## 7. DSE 必须新增的字段

四类场景的机器可读基线见
[embodied-physical-ai-requirements.example.yaml](../sim/config/embodied-physical-ai-requirements.example.yaml)。
文件显式标记为 `executable_by_current_dse: false`，避免当前 loader 静默忽略
新字段后仍输出一个看似有效的架构结论。

```yaml
scenario:
  class: embodied_vla | physical_ai_multijob
  provenance:
    source_urls: [...]
    assumptions: [...]

  graph:
    stages:
      - id: vision_encoder
        model: ...
        params_b: ...
        weight_bits: ...
        activation_bits: ...
        trace: ...
      - id: language_backbone
      - id: action_expert
        policy_type: flow_matching
        action_dim: ...
        action_horizon: 50
        integration_steps: 10
    dependencies: [...]

  temporal:
    host_tensor_arrival_hz: ...
    policy_inference_hz: ...
    action_execution_hz: ...
    chunk_refill_low_watermark: ...
    p50_ms_max: ...
    p95_ms_max: ...
    p99_ms_max: ...
    wcet_ms_max: ...
    max_input_age_ms: ...

  transfer:
    host_link: pcie
    bytes_h2d_per_job: ...
    bytes_d2h_per_job: ...
    copies: ...
    max_inflight: ...
    queue_policy: mailbox_latest

  scheduling:
    priority: ...
    criticality: ...
    preemptible: true
    max_nonpreemptible_ms: ...
    partition: shared | dedicated

  memory:
    physical_total_gb: ...
    npu_reservable_gb: ...
    system_reserved_gb: ...
    weights_gb: ...
    activations_peak_gb: ...
    kv_or_feature_cache_gb: ...
    workspace_peak_gb: ...
    dma_buffers_gb: ...
    usable_fraction: ...
    reserve_fraction_min: 0.20

  reliability:
    ecc: required
    iommu_isolation: required
    context_watchdog: required
    recovery_scope: [context, core, device]

  quality:
    task_success_min: ...
    perception_metric_min: ...
    action_error_max: ...
    quantization_regression_max: ...

  constraints:
    npu_power_w_max: ...
    power_scope: chip | module | board | soc | system
    area_mm2_max: ...
    sustained_temperature_c_max: ...

  product_evidence:
    exact_sku: ...
    document_revision: ...
    deployment_status: production_bom | shipping_option | devkit | reference_design | demo | roadmap
```

每个数值必须标记为：

- `market_source`：公开模型/产品给出的事实；
- `engineering_slo`：本项目的验收建议；
- `profile_required`：需要真实模型 trace；
- `target_measurement`：必须在目标芯片上实测；
- `unknown`：不能从 TOPS 或同系列产品推断。

## 8. 对当前 S3 的审查

当前文档 [arch-dse-three-scenarios.md](../reports/arch-dse-three-scenarios.md)
把 S3 定义成 Qwen2.5-7B、ViT、1024 token、10–20 个动作 token 和
10 FPS pipeline；可执行 [scenarios.yaml](../sim/config/scenarios.yaml)
中的 `onchip_7b` 则只有 7B token workload。

存在以下问题：

1. 10–20 个文本 token 只适用于离散、自回归 action tokenizer，
   不能代表 π0/SmolVLA/GR00T 类连续 action chunk。
2. 当前 10 FPS 同时混淆了视觉帧率、策略刷新率和动作执行率。
3. Qwen-VL trace 没有计入 QK^T 和 attention×V，四 crop 至少少
   0.344 TMAC，即最低低估 11.76%。
4. `layer_norm`、`softmax` 和 `gelu` 未被 CV simulator 计时。
5. 单一共享 engine 串行时，现有 10-token 组合约 7.05 FPS；
   10.99 FPS 依赖未显式建模的 stage overlap。
6. 20 token 在 197–198 TPS 下约需 101 ms，最多约 9.9 FPS；
   增大已经带宽饱和的阵列不能自动满足 10 FPS。
7. 增加 0.3B action expert 后，7B+0.675B ViT+action expert 的
   INT4 原始权重约 3.99 GB。5 GB 容量仅剩约 1.01 GB；如果只有
   90% 可用，则只剩约 0.51 GB 给 activation、cache、workspace 和 DMA。

复核输出见
[verify-current-s3-workload.md](../.omo/ultraresearch/20260723-134232/verify-current-s3-workload.md)
和
[verify-scenario-envelopes.md](../.omo/ultraresearch/20260723-134232/verify-scenario-envelopes.md)。

`origin/feat/scenario-driven-dse` 的 `6288a4d` 已改善 scenario loading、
hard constraints、attention/SFU、频率、内存单位和 capacity，但其
`warehouse_vla` 仍是 LLM prompt/output 模型，loader 也仍拒绝非 LLM
workload。因此它是下一步实现基础，而不是已经完成的 VLA DSE。

## 9. 架构筛选的通过条件

候选架构只有同时满足以下条件才可排名：

1. 编译后的完整 graph 无未计时算子，也没有未计入 WCET 的 CPU fallback。
2. 精度、稀疏性、量化和准确率门槛明确。
3. 完整内存 footprint 放得下，并保留至少 20% 工程余量。
4. H2D/D2H、queue、DMA、compute、completion 都进入端到端时间线。
5. 在并发背景负载和 thermal steady state 下满足 P99/WCET。
6. 快路径的 non-preemptible blocking 满足期限。
7. 超时、stale、ECC、IOMMU、watchdog 和 reset 有可观察的终态。
8. NPU 输出不会绕过 CPU/安全 MCU 的校验直接驱动 actuator。

TOPS、平均 FPS 或单模型 latency 只能作为中间指标，不能单独形成架构推荐。
