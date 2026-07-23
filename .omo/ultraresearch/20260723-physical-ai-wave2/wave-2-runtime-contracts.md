# Wave 2: runtime contracts, timing, and faults

## Host-submitted tensor job

- TI supports ONNX Runtime, TFLite, TVM, and native TIDL on the A72 host. Compiled model artifacts are loaded on target and inference runs on C7x/MMA. Native TIDL can keep all layers off the Arm execution path. TIDL exposes core selection, multiple instances, per-object priority, and preemption controls (`targetPriority`, `maxPreEmptDelay`).
- Horizon UCP exposes the most explicit contract: create an asynchronous task handle with input/output tensors, submit with priority/device/core metadata, wait with a millisecond timeout, then release. The default maximum live task count is 32 and is configurable. A service/client relay mode unifies multi-process scheduling.
- Qualcomm QAIRT/QNN provides host-side graph compilation/caching and execution across CPU/GPU/HTP, but the public IQ-9075 material does not define deadline, priority, maximum-inflight, or recovery semantics.
- Renesas publicly describes compilation to CNNIP and SRAM optimization, but not the V4H host job API, model concurrency, or tail-latency contract.

Sources:

- https://github.com/TexasInstruments/edgeai-tidl-tools
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-j721s2/latest/exports/docs/c7x-mma-tidl/ti_dl/docs/user_guide_html/itidl__rt_8h_source.html
- https://developer.horizon.auto/blog/13161
- https://docs.qualcomm.com/bundle/publicresource/topics/80-87189-1/overview.html
- https://www.renesas.com/en/software-tool/r-car-dnn-compiler

## Scheduling and concurrency

- Horizon hardware cannot interrupt an executing BPU function-call. Preemption is software-scheduled only at compiler-created function-call boundaries. Priority values 0-253 queue normally, 254 preempts normal work, and 255 can preempt normal/high work. A documented example splits work into 2 ms function-calls.
- Horizon warns that multiple submitted tasks serialize on an exclusive BPU and queue wait increases latency. `CORE_ANY` uses a host scheduler to select a BPU core by load.
- TI documents multiple AI models at distinct priorities and a higher-priority network preempting a lower-priority one.
- Therefore the transferable contract must specify maximum non-preemptible execution time, not merely "priority supported."

## Memory and bandwidth

- Horizon's model manifest separates input, output, shared temporary, intermediate, dynamic, and static memory; total footprint is static plus dynamic. Shared temporary memory cannot cross BPU cores, priority classes, or processes.
- Horizon provides average and peak DDR-byte estimates. One official example is 6.01 GB/s average for a 30-fps model and 23.21 GB/s peak. It warns that measured inference can slow due to queueing and bandwidth, priority-255 preemption flushes SRAM, and designs above 75% of theoretical bandwidth require validation.
- The same official guide describes buffer map/unmap caching, explicit tensor alignment, and no-copy ROI/crop techniques. These are evidence for persistent mappings and buffer-descriptor APIs at a standalone host/NPU boundary.
- TI's application bandwidth values include camera, ISP, and codec traffic; they must not be copied wholesale into a standalone-NPU bandwidth requirement. The model's tensor, weight, activation, and output bytes must be profiled separately.

## Deadline and tail latency

- None of TI, Qualcomm, Renesas, or Horizon publishes a P99 guarantee for these products.
- Official examples publish rates or averages: TI's 30-fps pipelines imply 33.3 ms periods; 12-fps analytics implies 83.3 ms; machine-vision ROI processing spans 10-30 fps. Horizon explicitly distinguishes single-frame latency from saturated multi-thread throughput and notes queue wait under concurrency.
- Proposed acceptance rule, not a vendor claim: for a periodic tensor job of period `T`, require P99 submit-to-complete <= 0.75T and a hard stale/timeout bound <= T, tested with the full concurrent model set and hostile memory traffic. This yields 25 ms at 30 Hz, 37.5 ms at 20 Hz, 62.5 ms at 12 Hz, and 75 ms at 10 Hz.

## Fault behavior

- Horizon documents a failure mode in which a bad model instruction can hang and the error is not reported; users inspect a task-running sysfs node and BPU logs. This is a negative requirement: a production accelerator must have a per-context watchdog, asynchronous fault completion, core quarantine/reset, and no silent hang.
- TI provides ECC-protected caches/SRAM/DDR, firewall, system monitor, clock diagnostics, and an error-signaling module. Qualcomm provides ECC memory plus a safety/monitoring subsystem. Renesas provides high-coverage fast fault detection/response and domain isolation.
- Correctable ECC, uncorrectable ECC, timeout, illegal access, malformed graph, thermal throttle, and internal watchdog events must be distinguishable at the host API; a whole-device reset should be the last recovery tier.

## EXPAND

- DEAD END: public P99 guarantees — no official source found; vendors publish mean latency/FPS or capacity only.
- DEAD END: public IQ-9075 per-context NPU scheduling/isolation specification — product and QAIRT docs expose the software stack but not a schedulability contract.
- DEAD END: public R-Car V4H DNN job API/concurrency limits — public pages describe compiler/platform capability, not execution ABI details.
- LEAD: validate proposed P99 values on target silicon with full mixed workloads — WHY: bandwidth and queue interactions dominate tail latency — ANGLE: 10k+ frame traces with synchronized hostile DDR load and injected faults.
