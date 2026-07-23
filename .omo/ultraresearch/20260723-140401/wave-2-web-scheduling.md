# Wave 2 digest: accelerator scheduling and real-time boundary

## Published scheduler facts

- Current CUDA programming guide:
  https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
  - Stream priority is a scheduling hint.
  - It does not preempt already executing work and does not guarantee ordering.
  - Priority may not apply to transfers.
- CUDA best practices:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
  - Separate process contexts are time-sliced and carry context-switch and
    memory overhead.
- Jetson architecture:
  https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/AR/JetsonSoftwareArchitecture.html
  - Jetson Thor supports MIG resource partitioning and PREEMPT_RT.
- XPU characterization:
  https://arxiv.org/abs/2604.24447
  - VLM phase is compute-bound; action expert is typically memory-bound and
    iterative.
  - For pi0, VLM utilization exceeds 90%, while action-expert utilization is
    20--40%; the expert can still dominate latency.
  - Stale-KV pipelining overlaps early denoising with current VLM inference,
    but success drops sharply after too many stale steps.
- Real-time chunking:
  https://arxiv.org/abs/2506.07339
  - Smooth async execution needs a defined chunk handoff, not just a queue;
    freezing committed actions and inpainting the remainder avoids jumps.

## MCU boundary

- TI humanoid motor-control brief:
  https://www.ti.com/lit/ab/slla659a/slla659a.pdf
  - Humanoid position updates occur at 1--4 kHz.
  - Current regulation exceeds 10 kHz.
  - The architecture separates centralized motion planning from decentralized
    MCU-based joint control.
- Therefore a 1 kHz learned S0 is an outer learned whole-body/position or torque
  command layer. The final current/FOC, PWM, watchdog, communications, and
  functional-safety loops remain on deterministic joint MCUs or real-time
  safety processors.

## EXPAND

none — the hard-real-time placement and accelerator preemption questions are
closed to the level supported by public sources.
