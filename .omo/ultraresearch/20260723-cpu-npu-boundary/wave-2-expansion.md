# Wave 2 — expansion and convergence

## NVIDIA closure

- STM compiles a static, non-preemptive schedule from the DAG, each pass's WCET, resource,
  dependencies, and requested deadline. It intentionally schedules one pass per hardware
  engine at a time for predictable performance.
- CGF consumer channels have bounded FIFO or one-slot mailbox semantics. Different-rate
  epochs require an explicit queue/overwrite/reuse policy.
- Orin's FSI has independent clocks/power, lockstep R52 cores, central hardware-safety
  monitoring, and communication to an external MCU. AURIX can coordinate safe shutdown,
  reset, and power-state transitions.
- TensorRT GPU fallback is an execution-placement feature, not a system safety fallback.

Sources:

- https://developer.nvidia.com/docs/drive/drive-os/6.0.8/public/driveworks-nvcgf/nvcgf_html/cgf_execution.html
- https://developer.nvidia.com/docs/drive/drive-os/6.0.8/public/driveworks-stm/nvstm_html/stm_compiler_computegraphandconstraints.html
- https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvcgf/cgf_details_channel.html
- https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/drive-os-linux-sdk/common/topics/fsi_integration/Functional_Safety_Island.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/dla-layer-restrictions.html

## Qualcomm closure

- Qualcomm's FastRPC CPU-side kernel driver queues requests over rpmsg; the DSP side
  dequeues and dispatches them. rpcmem creates zero-copy CPU/DSP buffers and exposes an
  FD suitable for DMA-BUF; dspqueue creates shared queues bound to an FD and signals in
  both directions.
- QNN exposes asynchronous execution, bounded async queue configuration, execution
  priority, signals, and subsystem-restart result codes. Public documentation explicitly
  distinguishes successful versus fatal subsystem recovery.
- Ride's safety island, hypervisor/VM separation, AUTOSAR/RTOS support, and QoS/isolation
  make it the safety control plane; HTP/GPU inference remains a monitored workload.

Sources:

- https://github.com/qualcomm/fastrpc
- https://github.com/qualcomm/fastrpc/blob/9d409211527f5c853351a8c014c2bcb271bc6f2d/Docs/manpages/fastrpc.3
- https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/api-rst_file_include_QNN_QnnBackend_h.html
- https://www.qualcomm.com/news/releases/2023/01/qualcomm-unveils-snapdragon-ride-flex---the-automotive-industry-

## TI closure

- TIOVX allocates OpenVX data objects from a shared DDR carveout at graph verification,
  passes pointers between nodes, and performs cache invalidate/writeback at map/unmap.
  Linux-visible descriptors include dma_buf_fd, offset, shared pointer, and host pointer.
- RPMessage is for small control messages; large data crosses as a pointer/handle/offset
  into shared memory.
- OpenVX queue timeouts only bound the host wait; the application must decide what to do
  on timeout. They do not cancel a running graph.
- R5F/MCU watchdogs monitor A72, C7x, and other cores. TI's 2026 main-domain recovery
  flow shuts down IPC, enters MCU-only mode, power-cycles the main domain, reloads cores,
  and re-establishes IPC.

Sources:

- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/10_01_00_04/exports/docs/tiovx/docs/user_guide/TIOVX_MEMORY_MANAGEMENT.html
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/tiovx/docs/user_guide/structtivx__shared__mem__ptr__t.html
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/09_00_00_02/exports/docs/pdk_jacinto_09_00_00_45/docs/userguide/jacinto/modules/ipc.html
- https://www.ti.com/lit/an/sdaa352/sdaa352.pdf

## Open-stack closure

- Default ROS 2 executors have mixed/round-robin scheduling, priority-inversion risk, and
  no explicit callback ordering. For a hard boundary use dedicated callback groups and
  OS priorities, WaitSet/rclc, or a stronger static scheduler.
- ROS 2 lifecycle provides externally supervised configure/activate/deactivate/error
  transitions. Autoware aggregates diagnostics and stale timeouts into HazardStatus and
  selects a minimum-risk maneuver; missing hazard status itself triggers emergency stop.
- iceoryx uses preallocated shared-memory chunks, per-subscriber bounded queues,
  reference-counted lifetime, and safe overflow/latest-wins behavior.

Sources:

- https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Executors.html
- https://design.ros2.org/articles/node_lifecycle.html
- https://autowarefoundation.github.io/autoware.universe_planning/latest/system/emergency_handler/
- https://autowarefoundation.github.io/autoware.universe_planning/pr-5566/system/system_error_monitor/
- https://github.com/eclipse-iceoryx/iceoryx/wiki/Eclipse-iceoryx%E2%84%A2-in-1000-words
- https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ECRTS.2019.6

## Convergence

All first-wave leads were investigated. A second expansion pass returned corroborating
implementations rather than a new architectural class. Remaining gaps are vendor safety
manual details under NDA and unavailable on-hardware timing validation.

## EXPAND

none — the public-source architecture, data-plane, scheduling, safety, and recovery leads
converged; remaining documents are access-controlled safety cases rather than unchecked
public leads.
