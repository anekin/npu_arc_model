# Ultraresearch Synthesis: Embodied / Physical-AI NPU Coprocessor Requirements

Date: 2026-07-23
Repository: `/home/prj/zhengs/codex/npu_arc_model`

## Research execution

- Three expansion waves were run across market platforms, VLA models, shipping
  robots, Chinese edge silicon, CPU-NPU partitioning, host-connected
  coprocessors, action-chunk timing, dual-rate scheduling, continuous physical
  AI, and repository implementation/history.
- Fourteen initial specialist lanes disconnected before findings. The axes were
  retried through default research workers and direct primary-source browsing;
  no axis was dropped.
- Thirteen retry/expansion lanes completed successfully: six replacement
  baselines, four Wave-2 lead expansions, and three Wave-3 convergence checks.
- Vendor/model lanes inspected dozens of official sources each. The platform
  ledger contains 67 official URLs and the VLA ledger 48 primary-source links;
  these counts overlap with other lanes and are not presented as unique totals.
- Two direct executable checks reproduced current S3 arithmetic, measured trace
  omissions, compared operation dispatch, and calculated capacity/transfer/rate
  envelopes.

The user-facing requirements baseline is
[docs/embodied-physical-ai-npu-requirements.md](../../../docs/embodied-physical-ai-npu-requirements.md).
The corresponding non-executable schema proposal is
[sim/config/embodied-physical-ai-requirements.example.yaml](../../../sim/config/embodied-physical-ai-requirements.example.yaml).

## Executive result

The current application requirement is not clear enough for a defensible
architecture search. “7B VLM/VLA + 10–20 action tokens + 10 FPS” mixes at least
four distinct markets:

1. compact 450M action-chunk VLA;
2. 3.3B continuous-action flow VLA;
3. 7B slow reasoner plus 80M/200-Hz fast policy;
4. concurrent 10–30-FPS perception/localization/inspection physical AI.

Public models prove that action execution frequency is often not policy
inference frequency. π0 executes actions at up to 50 Hz but its published
50-Hz-robot rollout replans every 0.5 s after 25 actions; SmolVLA uses an
event-driven low-watermark queue; Helix separates a 7–9-Hz 7B model from an
80M 200-Hz path. [S1][S2][S3]

Because the NPU is a CPU-hosted coprocessor, the correct portable input is a
timestamped tensor-job DAG with bytes, dependencies, period, deadline,
freshness, priority, and fences. Camera/MIPI/ISP, sensor synchronization,
actuator interfaces, and final safety decisions remain outside this NPU.
[S4][S5][S6]

## Market findings

### Direct coprocessors

Hailo-10H is the closest public analogue: PCIe/USB host connection,
20 INT8/40 INT4 TOPS, LPDDR4/4X, and 2.5 W typical. Its public runtime exposes
asynchronous submission, backpressure and scheduling controls, although public
material does not prove in-kernel preemption. [S7][S8]

The broader M.2 market shows why memory and runtime behavior must be first-class:
Ara240 and SAKURA-II offer up to 16 GB; DEEPX combines 25 INT8 TOPS with
documented async/multi-model/error APIs; MemryX has deterministic streaming but
only a 40M INT8-weight capacity; MLU220 M.2 is 8 INT8 TOPS/8.25 W but lacks
public real-time queue/preemption detail. [S9][S10][S11][S12]

Chinese public evidence supports AX8850 M.2 as a below-8-W expansion card and
Atlas 200I A2 as an 8/20-TOPS PCIe endpoint with 4/8/12-GB ECC memory. Atlas
multi-stream priority is fixed/reserved, so multiple streams cannot be treated
as proof of real-time priority or preemption. [S13][S14][S15]

### Integrated physical-AI platforms

TI AM69A, Qualcomm IQ-9075, Renesas R-Car V4H, Orin, and Thor are not all pure
coprocessors, but they provide workload, capacity, concurrency, isolation, and
power reference points. Their TOPS values are not directly comparable:
Thor uses sparse FP4, Orin publishes multiple aggregate/dense scopes, IQ-9075
publishes dense INT8, and some vendors use proprietary/effective metrics.
[S16][S17][S18][S19][S20]

TI's public workload paper is more useful than peak TOPS: it includes
12x2MP/12-FPS, 3x8MP/10–30-FPS, and 8x2MP/30-FPS examples with corresponding
system bandwidth measurements. For this project those must be converted into
the post-CPU tensor trace rather than copied as camera-interface requirements.
[S21]

### Shipping robots

Public shipping robot documentation repeatedly separates an AI “brain” from
motion-control/real-time computers. AgiBot A2, UBTECH Walker Tienkung, Fourier
GR-1, and Leju KUAVO all support this architectural direction, though exact AI
accelerator BOMs are often undisclosed. [S22][S23][S24][S25]

First-party public evidence does not establish a named production robot SKU
using A2000/C12xx, AX8850, or AX8910. Partnerships, reference designs, and
trade-show demonstrations must retain their deployment status rather than being
reported as production BOMs. [S26][S27]

## Required scenario split

### E0: Compact VLA

Source anchor: SmolVLA 450M, approximately 100M action expert, 512x512
multi-camera observation, H=50, ten flow steps, async latest-wins refill.
[S1][S28]

Engineering exploration:

- 10–20 dense INT8 TOPS candidate range;
- 4/8 GB local memory;
- event-driven low-watermark policy requests with 1/2/5/10-Hz effective sweep;
- 20/30/50-Hz action execution sweep;
- deadline derived from remaining action-queue time;
- model/action expert resident, async DMA, at least two service classes.

These are search points, not vendor guarantees.

### E1: Continuous-action generalist VLA

Source anchor: π0 3.3B, 2–3 RGB images plus state/language, H=50, ten flow
passes, prefix KV caching, and 1.25/2-Hz published replanning on 20/50-Hz
robots. Its RTX 4090 model stages total 73 ms; that is not an NPU guarantee.
[S2]

Engineering exploration:

- 50–100 dense INT8 TOPS candidates;
- 8/16/24 GB capacity sweep, with public reference software around a 14-GB
  inference-memory class;
- 2-Hz baseline and 5/10-Hz aggressive replan stress points;
- persistent prefix plus iterative action-suffix graph;
- preemption at flow-iteration boundaries and explicit queue low-watermark.

### E2: Dual-rate humanoid

Source anchor: Helix 7B S2 at 7–9 Hz and 80M S1 at 200 Hz, with latest-latent
asynchronous exchange. Public sources do not disclose the exact S1 chunk length,
so 5 ms is an action-output period, not automatically a full-network deadline.
[S3][S29]

Engineering exploration:

- 100–200 dense INT8 TOPS candidates;
- 8/16/24/32 GB capacity sweep; 5 GB only as an aggressive INT4 feasibility point;
- both models resident;
- dedicated partitions/engines preferred;
- if shared, model maximum non-preemptible blocking and response time;
- S2 latest-value mailbox, never an accumulating FIFO;
- if S1 really invokes every tick, P99 at most 4 ms and hard deadline 5 ms.

### P0: Continuous-perception physical AI

Source anchors: TI multi-camera workload examples, 32-TOPS AM69A, 50/100-dense-
TOPS IQ-9075, and R-Car V4H isolation/safety mechanisms. [S16][S17][S18][S21]

Engineering exploration:

- 4–8 resident models and at least 16 inflight jobs;
- 33.3-ms critical perception with proposed P99 at most 25 ms;
- 50-ms localization/fusion with proposed P99 at most 37.5 ms;
- 83–100-ms auxiliary jobs with proposed P99 at most 62.5–75 ms;
- maximum non-preemptible segment at most 1–2 ms;
- initial admission below 70–75% of measured sustainable memory bandwidth;
- ECC/parity, IOMMU/domain isolation, watchdog, hierarchical context/core/device
  reset and observable terminal error statuses.

No public vendor supplies a general mixed-workload P99 guarantee, so these are
acceptance SLO proposals requiring target-silicon validation.

## CPU-NPU contract

The contract needs:

- pre-registered pools, tensor schema, model hash and compiled target;
- opaque buffer handles/offsets plus acquire/release fences;
- capture time, submit deadline, completion deadline, stale-after time;
- FIFO versus latest-wins policy, bounded depth, maximum inflight;
- priority, criticality, admission, timeout and overrun action;
- one terminal completion per accepted job;
- structured deadline/ECC/IOMMU/watchdog/reset errors;
- queue/start/end timestamps, bytes moved and execution partition.

The host validates output shape, finite/range limits, timestamp age and
application invariants. A successful accelerator return must not directly
authorize actuation. [S4][S5][S6]

## Repository audit

Current `main` has only three executable scenarios, and `onchip_7b` is a generic
Qwen2.5-7B token workload labeled VLM/VLA. It has no action head, flow/diffusion
loop, action horizon, policy/action rates, tensor DAG, or host-boundary timing.
[L1][L2]

The documentary S3 combines Qwen2.5-7B, a ViT trace, 1024 prompt tokens,
10–20 output tokens and a pipeline FPS formula. It cannot represent current
continuous action-chunk systems. [L3]

Empirical checks found:

- Qwen-VL trace total 2.582580 TMAC is reproducible;
- omitted QK^T and attention-value GEMMs add at least 0.344269 TMAC, a minimum
  11.76% undercount;
- `gelu`, `layer_norm` and `softmax` in the trace are not dispatched by the CV
  simulator;
- 10.99 FPS requires the report's overlap assumption; a single shared serial
  engine gives 7.05 FPS;
- 20 tokens at 197–198 TPS gives approximately 9.9 FPS, not 10 FPS.

The executable evidence is in [V1].

`origin/feat/scenario-driven-dse` at `6288a4d` improves scenario loading,
constraints, attention/SFU accounting, frequency, memory units and capacity,
but its `WorkloadSpec` and loader remain LLM-only. Its `warehouse_vla` is still
a prompt/output-token workload rather than an embodied task graph. [L4]

## Capacity and transfer verification

Raw-weight arithmetic:

- SmolVLA 450M: 0.225/0.450/0.900 GB at INT4/INT8/BF16;
- π0 3.3B: 1.650/3.300/6.600 GB;
- Helix 7B+80M: 3.540/7.080/14.160 GB;
- current 7B+0.675B ViT+0.3B action expert: 3.987 GB INT4.

The last case leaves about 1.013 GB in a nominal 5-GB memory, or 0.513 GB at
90% usable capacity, before activations, cached features/KV, workspace and DMA.
[V2]

Three 512x512 FP16 RGB tensors at 10 jobs/s are only about 47.2 MB/s, while
eight 1920x1920 RGB8 tensors at 30 jobs/s are about 2.65 GB/s. Therefore VLA
host traffic can be modest after CPU reduction, while continuous high-resolution
perception needs explicit link/bytes/contention modeling. [V2]

## Contradictions resolved

- `50 Hz VLA` frequently means action consumption, not 50 full VLA executions.
  [S2][S29]
- `dual system` can mean a serial VLM-to-DiT model rather than independently
  scheduled dual-rate tasks; only runtime evidence should set periods.
- Multiple streams/queues do not prove priority or preemption. Atlas A2 exposes
  streams but the public priority parameter is fixed; HailoRT exposes scheduling
  priority but public evidence does not prove in-kernel preemption. [S8][S15]
- TOPS without precision, sparsity and scope is not comparable. [S7][S19][S20]
- Board/SoC/system power cannot be copied into an NPU-only power constraint.
- Collaboration, demo, reference design and production BOM are different
  evidence states. [S26][S27]

## Remaining gaps

Public research cannot close:

- exact target compiler operator coverage and quantization quality for
  SmolVLA/π0/GR00T;
- target-NPU activation/workspace/cache footprint;
- mixed-load P99/WCET at thermal steady state;
- DMA fixed latency, sustained bandwidth and maximum outstanding requests on
  the selected host link;
- exact preemption granularity and reset scope;
- vendor safety manuals/FMEDAs and production robot BOMs.

These are explicitly `target_measurement`, `vendor_gated`, or `unknown`, not
values to infer from marketing TOPS.

## Expansion trace and convergence

- Wave 1 established market/model/platform/repository baselines.
- Wave 2 expanded action chunks, host-connected coprocessors, dual-rate
  scheduling and continuous physical AI.
- Wave 3 challenged the four-profile envelope and checked Chinese deployments
  and coprocessor forms.
- Remaining leads require unreleased documents, NDA data, target hardware or
  application tensor traces. Public-source architecture and workload searches
  otherwise converged.

## Sources

- [S1] [SmolVLA official blog](https://huggingface.co/blog/smolvla)
- [S2] [π0 paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [S3] [Figure Helix](https://www.figure.ai/news/helix)
- [S4] [NVIDIA DriveWorks CGF](https://developer.nvidia.com/docs/drive/drive-os/6.0.8/public/driveworks-nvcgf/nvcgf_html/cgf_execution.html)
- [S5] [Qualcomm FastRPC](https://github.com/qualcomm/fastrpc)
- [S6] [TI TIOVX memory management](https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/10_01_00_04/exports/docs/tiovx/docs/user_guide/TIOVX_MEMORY_MANAGEMENT.html)
- [S7] [Hailo-10H product brief](https://hailo.ai/files/hailo-10h-product-brief-en/)
- [S8] [HailoRT pinned runtime API](https://github.com/hailo-ai/hailort/blob/d503417f2a0db186a838390fb08690c4ea0f415e/hailort/libhailort/include/hailo/infer_model.hpp)
- [S9] [NXP Ara240 fact sheet](https://www.nxp.com/docs/en/fact-sheet/ARA2DNPUFS.pdf)
- [S10] [EdgeCortix hardware](https://www.edgecortix.com/en/hardware)
- [S11] [DEEPX inference API](https://developer.deepx.ai/tech-docs/DXNN-SDK/DX-RT/Inference_API/)
- [S12] [Cambricon MLU220 M.2](https://www.cambricon.com/index.php?a=lists&c=index&catid=57&m=content)
- [S13] [Axera AX8850 M.2](https://www.axera-tech.com/en/news/2989.html)
- [S14] [Atlas 200I A2 hardware](https://www.hiascend.com/hardware/accelerator-module-A2)
- [S15] [Atlas stream-priority limitation](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/runtimeapi/aclcppdevg_03_0069.html)
- [S16] [TI AM69A](https://www.ti.com/product/AM69A)
- [S17] [Qualcomm IQ9 brief Rev. G](https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_G_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf)
- [S18] [Renesas R-Car V4H](https://www.renesas.com/en/products/r-car-v4h)
- [S19] [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
- [S20] [Jetson Thor](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
- [S21] [TI AM69A workload paper](https://www.ti.com/lit/wp/spradb4/spradb4.pdf)
- [S22] [AgiBot A2 overview](https://open.agibot.com/docs/aimdk/a2/v1_3/dev_guide/01-a2_overview)
- [S23] [UBTECH Walker Tienkung manual](https://docs.ubtrobot.com/walker-tienkung/en/docs/user-guide/6/)
- [S24] [Fourier GR-1 product sheet](https://video.fftai.com/%E4%BA%BA%E5%BD%A2%E6%9C%BA%E5%99%A8%E4%BA%BA%E4%B8%AD%E6%96%87%E5%8D%95%E9%A1%B5.pdf)
- [S25] [Leju KUAVO 5](https://www.lejurobot.cn/en/products/kuavo-5)
- [S26] [Black Sesame robot cooperation](https://www.blacksesame.com/zh/list_11/780.html)
- [S27] [Axera physical-AI reference](https://www.axera-tech.com/en/node/3238?q=zh-hans%2Fnews%2F3238.html)
- [S28] [Pinned SmolVLA implementation](https://github.com/huggingface/lerobot/tree/73dbb6f43a5088583706c91fb73c6957bca5f806/src/lerobot/policies/smolvla)
- [S29] [Figure action-chunk description](https://www.figure.ai/news/helix-logistics)
- [L1] [Current scenarios](../../../sim/config/scenarios.yaml)
- [L2] [Current scenario DSE](../../../sim/dse_scenario.py)
- [L3] [Current S3 report](../../../reports/arch-dse-three-scenarios.md)
- [L4] [Repository audit](wave-1-repository-audit.md)
- [V1] [Current S3 executable verification](verify-current-s3-workload.md)
- [V2] [Scenario envelope arithmetic](verify-scenario-envelopes.md)
