# Expansion log: non-humanoid physical-AI NPU workload contracts

Core question: What host-submitted tensor-job contracts should a standalone NPU satisfy for AMR, industrial-robot, and autonomous-driving workloads when sensor interfaces and preprocessing remain on the host CPU?

Research axes:

1. TI AM69A / TDA4VH / J784S4: camera and accelerator capacity, job submission/runtime, memory and safety provisions.
2. Qualcomm Dragonwing IQ-9075: concurrent CV workload capacity and platform partitioning/reliability.
3. Renesas R-Car V4H: automotive reference pipelines, QoS/isolation, safety, ECC/fault handling.
4. Horizon Robotics automotive compute: BPU concurrency, memory, safety/QoS, and deployment/runtime contracts.
5. Application timing: official AMR, industrial-robot, and ADAS reference models/pipelines; frame periods and latency obligations.
6. Transferability: separate sensor-SoC features from requirements observable at the accelerator/host boundary.

Constraints: official sources only; at least 12 distinct searches; no subagents.

## Wave 1

Status: complete. Vendor envelopes and application rates extracted.

Markers gained:

- Runtime task granularity and preemption.
- Model memory and DDR admission.
- Frame period to P99 derivation.
- Silent hang/recovery semantics.

## Wave 2

Status: complete. All four leads investigated.

Closed:

- Public vendor P99 guarantees: dead end; no official P99 figures.
- IQ-9075 NPU scheduling/isolation details: not public.
- R-Car V4H host execution ABI/concurrency: not public.

New actionable lead:

- Empirical target-silicon validation of the proposed deadline contracts. This is a downstream implementation/verification task rather than another documentary-research lane.

Convergence: documentary leads are exhausted; the remaining open lead requires hardware execution.

Search record: 52 distinct official-domain searches across TI, Qualcomm, Renesas, Horizon Robotics, ROS 2/Nav2, and EtherCAT sources.
