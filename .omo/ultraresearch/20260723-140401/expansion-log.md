# Expansion log

Date: 2026-07-23

## Wave 1

Axes:

1. Figure Helix and Helix 02 architecture, rates, deployment isolation, shared state.
2. NVIDIA GR00T N1 architecture, model sizes, chunking, deployment API.
3. Comparable hierarchical/dual-rate policies.
4. Accelerator scheduling, resource isolation, and MCU/NPU boundary.

Searches: 52 distinct queries across Figure, NVIDIA, arXiv/PMLR/NeurIPS,
Hugging Face, Physical Intelligence, CUDA/Jetson documentation, TI motor-control
documentation, Galaxea, and robotics accelerator papers.

New leads:

- Action sample rate versus network invocation rate.
- GPU stream priority does not provide hard preemption.
- System 0 at 1 kHz is a learned outer controller, not the final motor-current loop.
- VLA inference has phase-dependent compute/bandwidth behavior.

## Wave 2

Closed the leads with:

- pi0 deployment details: 50 Hz action stream but one inference per 25 executed
  actions on 50 Hz robots.
- CUDA 13 documentation: stream priorities are hints and do not preempt a
  running kernel; Jetson Thor offers MIG and PREEMPT_RT.
- TI humanoid drive guidance: 1--4 kHz position updates and greater than 10 kHz
  current regulation, normally on decentralized MCUs.
- XPU profiling: compute-bound VLM followed by memory-bound iterative action
  expert; explicit stale-KV pipelining trade-off.

New lead:

- Galaxea G0 publishes a clean three-rate hierarchy.

## Wave 3

Closed the G0 lead:

- G0-VLM: less than 2 Hz subtask instruction.
- G0-VLA: 15 Hz motion planning/action-chunk generation.
- Robot control: 200 Hz.
- System 1 and System 2 run asynchronously.

Convergence: no unchecked architecture or scheduling lead remains for the
requested scope. Remaining gaps are vendor-undisclosed implementation details,
listed in `SYNTHESIS.md`.
