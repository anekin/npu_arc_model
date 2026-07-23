# Expansion log

## Phase 0

- Core question: In production robots and autonomous systems, where should the CPU-host/NPU-or-GPU boundary sit, and what explicit runtime contract should cross it?
- Axes:
  - NVIDIA DRIVE/Jetson/IGX: accelerator pipeline, NvMedia/NvSci/NITROS data movement, queues, QoS, safety.
  - Qualcomm Ride/RB5/RB6: CPU/Hexagon/GPU split, QNN/SNPE buffers, FastRPC/DMA-BUF, safety architecture.
  - TI Jacinto/TDA4: A72/R5F/C7x/MMA/DSP split, TIOVX object descriptors, shared memory, IPC, watchdogs, safety island.
  - Open stack: ROS 2/Autoware/NITROS/iceoryx ownership, zero-copy, QoS/deadlines, lifecycle and failure handling.
  - Synthesis: a vendor-neutral NPU boundary contract.
- Codebase relevant: no. External: yes. Browsing: yes. Verification likely: source cross-checking, not executable hardware validation. Report requested: Markdown synthesis in response.
- Constraint: no subagents. The main researcher will execute all source lanes and expansion waves directly.

## Wave 1 — planned source sweep

- NVIDIA official architecture, SDK, memory/interprocess, safety.
- Qualcomm official product, SDK, memory/RPC, safety.
- TI official processor, TIDL/TIOVX, IPC, safety.
- ROS 2/Autoware/Open Robotics official architecture, QoS, zero-copy, safety/lifecycle.

## Wave 1 — results

- Searches completed: 64 distinct queries by the end of the initial and first expansion
  passes.
- New leads: NVIDIA STM/FSI; Qualcomm FastRPC/QNN SSR; TI TIOVX coherency/MCU recovery;
  ROS 2 executor determinism/Autoware MRM.
- Digest: `wave-1-official-platforms.md`.

## Wave 2 — results

- Searches completed: 100 distinct queries across the full session, including the final
  technical-paper sweep.
- Every Wave 1 lead was investigated.
- No new public architectural category emerged; sources converged on preallocated shared
  buffers, asynchronous bounded queues, explicit completion, rich-OS orchestration, and
  independent safety supervision.
- Remaining closed gap: detailed FMEDA/safety-manual assumptions of use are vendor
  access-controlled and cannot be derived from public documentation.
- Digest: `wave-2-expansion.md`.
- Synthesis: `SYNTHESIS.md`.

## Convergence

Two expansion waves completed. Zero unchecked public leads remain.
