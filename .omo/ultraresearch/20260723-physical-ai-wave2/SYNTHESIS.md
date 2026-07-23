# Ultraresearch synthesis: non-humanoid physical-AI NPU workload contracts

Searches: 52 official-domain queries · Expansion waves: 2 · Hardware execution: not available

## Executive result

The portable contract is not "N cameras" or peak TOPS. With capture, ISP, codecs, sensor synchronization, control buses, and safety MCU left on the host, the NPU sees compiled graphs plus timestamped tensor buffers. It must accept asynchronous jobs, keep multiple model contexts resident, provide deadline-aware scheduling with a bounded non-preemptible interval, admit workloads based on resident memory and peak DDR traffic, isolate contexts spatially and temporally, protect all memory paths with ECC, and fail a job explicitly rather than hanging.

No official vendor source found in this research publishes P99 inference guarantees. The workload contract below therefore separates source facts from proposed acceptance SLOs. For periodic inference with period `T`, use P99 submit-to-complete <= `0.75T` and hard stale/timeout <= `T`, under the complete concurrent workload and hostile memory traffic. Safety analysis must use the hard bound; P99 alone is not a safety guarantee.

## Source-grounded workload envelopes

| Workload | Official reference envelope | Transferable NPU job set | Proposed acceptance contract |
| --- | --- | --- | --- |
| AMR | TI demonstrates 8 x 2 MP at 30 fps with detection at 30 fps, using 24 TOPS; Qualcomm positions IQ-9075 for AMR/robotics and two tensor processors up to 100 dense TOPS. Nav2's default controller loop is 20 Hz, but that control loop belongs on the host/RT subsystem. | 2D detection/segmentation or occupancy, depth/stereo features, localization descriptors; latest-frame semantics. | 4-8 resident models, >=16 in-flight jobs; critical 30 Hz class P99 <=25 ms and timeout <=33.3 ms; secondary 10-20 Hz class P99 <=37.5-75 ms. NPU output may update control, but the 20 Hz controller must not block on NPU completion. |
| Industrial robot / machine vision | TI's reference is 3 x 8 MP at 30 fps, ROI inference at 10-30 fps, 24 TOPS and 15.35 GB/s total SoC DDR. Qualcomm reports sub-millisecond EtherCAT cycle time with <8 us jitter while AI runs, showing that motor control and inference are separate classes. | ROI classification/detection, segmentation/defect inspection, 6D pose/keypoints, optional collision/person detector. | 4-8 resident models, >=16 in-flight jobs; safety/pose/obstacle 30 Hz class P99 <=25 ms; quality-inspection 10-12 Hz class P99 <=62.5-75 ms. Sub-millisecond servo/EtherCAT work is excluded from NPU scheduling and remains on host/RT MCU. |
| ADAS/AD reference | R-Car V4H supplies 34 TOPS/four CV engines for L2+/L3 and supports dual-chip fail-degraded operation. Journey 5 supplies 128 TOPS and up to 16 HD cameras; Journey UCP defaults to 32 live tasks. | Per-camera encoders, BEV/fusion, object/lane/free-space heads, depth/flow, DMS/OMS; graph dependencies managed by host fences. | >=16 resident model contexts and >=32 live jobs; critical 30 Hz perception P99 <=25 ms, 20 Hz fusion P99 <=37.5 ms, 10-12 Hz auxiliary jobs P99 <=62.5-75 ms. A late job is completed with `DEADLINE_MISSED` or dropped as stale; it never silently blocks the next frame. |

## Host-to-NPU ABI

### Model lifecycle

- Compile offline; load/verify once; keep graph weights resident.
- Model manifest reports operator compatibility, precision, input/output shape and alignment, weights, input/output, scratch/intermediate memory, average and peak DDR bytes per inference, and maximum non-preemptible segment.
- Model load is not charged to steady-state frame latency, but warm-up and first-use effects are separately measured.

### Job descriptor

- `context_id`, `job_id`, `frame_seq`, capture timestamp, absolute deadline, stale-after timestamp.
- Priority/criticality class, selected or any core, tensor-buffer handles plus offsets/strides, dependency and completion fences.
- Optional batch/ROI metadata and expected output buffers.

### Completion

- Status distinguishes success, deadline miss, cancellation/stale drop, correctable ECC event, uncorrectable ECC, IOMMU/firewall violation, invalid graph/tensor, thermal throttling, watchdog timeout, core reset, and device loss.
- Completion includes submit/start/end timestamps, queue and execution time, bytes read/written, core assignment, and ECC/fault counters.

## Scheduling, concurrency, and QoS

- At least three service classes: safety/critical perception, regular periodic perception, and best-effort/background.
- Earliest-deadline-first within a class, with admission control. A numeric priority without an absolute deadline is insufficient.
- Maximum non-preemptible interval <=1-2 ms. Horizon's official runtime shows why: hardware cannot interrupt a function-call; compiler-created function-call boundaries are the only preemption points, and a 2 ms split is demonstrated.
- Provide per-context compute and memory-bandwidth quotas, queue-depth bounds, and stale-frame replacement. Do not let background batching induce priority inversion.
- Reserve capacity for critical jobs; do not claim isolation merely because multiple cores exist. Renesas RegionID/QoS provides the transferable model: both spatial and temporal isolation plus traffic protection.

## Memory contract

- Pin/register host buffers once and reuse mappings. Provide IOMMU-domain isolation per process/VM/safety partition and read-only shared weights.
- Do not share scratch across safety domains. Horizon's temporary-memory sharing is restricted by core, priority, and process, illustrating why sharing must be explicit.
- Admit the full workload from the sum of resident static memory plus maximum concurrent dynamic/scratch memory.
- Admit bandwidth from peak, not average, graph traffic. Horizon gives an example with 6.01 GB/s average but 23.21 GB/s peak and warns that designs above 75% of theoretical bandwidth need validation.
- Use <=70-75% of measured sustainable bandwidth as the initial admission ceiling until mixed-workload P99 tests prove more headroom. TI's 9.49-15.35 GB/s application figures include ISP/codec/camera traffic and are evidence of system-level contention, not direct standalone-NPU requirements.

## ECC, isolation, and fault recovery

- SECDED/ECC on external-memory payloads and weights, ECC/parity on on-chip SRAM/cache/register/control paths, plus end-to-end error reporting.
- Correctable errors increment counters and optionally scrub; uncorrectable errors fail the owning job/context and quarantine the affected core or memory region.
- Per-context watchdog with bounded interrupt/escalation; cancel queued dependent jobs, reset the context, then reset/quarantine one core, and only then reset the whole device.
- No silent hangs. Horizon publicly documents a case where bad model instructions can hang without an error report and require sysfs/log inspection; this is a concrete anti-requirement.
- Support fault injection for ECC, timeout, illegal address, and watchdog mechanisms. Export diagnostics to a host safety manager.
- VM/process isolation must include NPU DMA/IOMMU assignment and scheduling quotas; CPU-only KVM isolation is not sufficient.

## SoC features to exclude from a standalone NPU requirement

- Camera count, CSI lanes, ISP/VISS/Pyramid/LDC, video encode/decode, camera synchronization, radar/LiDAR interfaces.
- CAN/FlexRay/TSN/EtherCAT master, motor-control timing, navigation/control-loop execution.
- Display/GPU composition, secure boot of the host, host hypervisor, isolated safety MCU/RT subsystem.
- Dual-SoC fail-degraded topology and PMIC power sequencing.

The corresponding transferable requirements are tensor format and cadence, buffer/fence ABI, scheduling/deadline behavior, memory/bandwidth admission, IOMMU isolation, ECC and fault telemetry, watchdog/reset granularity, and the ability for the host safety manager to degrade or fail over.

## Gaps

- TI, Qualcomm, Renesas, and Horizon do not publish P99 latency under mixed model and memory contention.
- Public IQ-9075 documentation does not expose NPU priority/preemption, maximum in-flight work, per-context isolation, or recovery bounds.
- Public R-Car V4H documentation does not expose its DNN host ABI, concurrency limits, or execution-time distribution.
- Safety certifications do not substitute for an application-specific deadline/fault-reaction contract.

## Primary official sources

1. TI AM69A datasheet: https://www.ti.com/lit/ds/sprsp92d/sprsp92d.pdf
2. TI AM69A workload white paper: https://www.ti.com/lit/wp/spradb4/spradb4.pdf
3. TI TIDL tools/runtime: https://github.com/TexasInstruments/edgeai-tidl-tools
4. Qualcomm IQ9 platform brief: https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_E_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf
5. Qualcomm IQ-9075 EVK brief: https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Qualcomm-Dragonwing-IQ-9075-EVK-Brief.pdf
6. Qualcomm IQ-9075 module brief: https://docs.qualcomm.com/doc/87-97354-1/87-97354-1_REV_C_Qualcomm_Dragonwing_IQ-9075_Module_Product_Brief.pdf
7. Renesas R-Car V4H: https://www.renesas.com/en/products/r-car-v4h
8. Renesas RegionID/QoS fusion architecture: https://www.renesas.com/en/blogs/exploring-entry-fusion-application-architecture-and-cost-effective-solution-utilizing-r-car-v4h
9. Horizon Journey portfolio: https://www.horizon.auto/en/solutions/horizon-journey
10. Horizon J6 UCP runtime guide: https://developer.horizon.auto/blog/13161
11. Horizon Journey 5 safety architecture: https://www.horizon.auto/news/technology/275
12. ROS 2 Nav2 controller timing: https://docs.nav2.org/configuration/packages/configuring-controller-server.html

## EXPAND

- LEAD: target-silicon mixed-workload characterization — WHY: documentary sources stop at averages/capacity and do not establish P99 — ANGLE: 10k+ frames per workload, simultaneous critical/background models, 70-90% DDR load, thermal steady state.
- LEAD: fault-injection recovery characterization — WHY: ECC presence does not define notification/recovery time — ANGLE: single/double-bit errors, illegal DMA, stuck graph, deadline timeout, per-core reset and failover.
- LEAD: public/private vendor clarification — WHY: IQ-9075 and R-Car runtime schedulability details are absent publicly — ANGLE: request maximum live contexts/jobs, non-preemptible time, queue policy, IOMMU partitioning, and reset granularity from vendor support.
