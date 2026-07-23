# Wave 1: CPU Host / NPU Coprocessor Boundary

## Outcome

Production NVIDIA DRIVE, Qualcomm Snapdragon, TI Jacinto, and ROS 2 / Autoware
stacks converge on a host-controlled, asynchronous accelerator contract:

- CPU owns graph/session lifecycle, admission, deadlines, health, result validation,
  degradation policy, and safety coordination.
- NPU owns bounded execution of precompiled tensor graphs over registered buffers.
- Data crosses as buffer handles plus offsets and fences, not raw process pointers.
- Queues are bounded and explicitly choose FIFO or latest-wins/mailbox semantics.
- Safety-relevant actuation remains under an independent safety MCU/island; a valid
  NPU result is necessary input, not actuator authority.

## Requirements imported into the scenario schema

- `host_to_npu_bytes_per_job`, `npu_to_host_bytes_per_job`
- `buffer_pool_slots`, `alignment_bytes`, `memory_domain`, `cache_policy`
- `period_ms`, `completion_deadline_ms`, `max_input_age_ms`
- `queue_policy`, `queue_depth`, `max_inflight`, `priority`, `criticality`
- `preemption_granularity`, `timeout_ms`, `overrun_action`
- `fallback_model`, `context_reset_ms`, `heartbeat_ms`
- `output_validation`, `safety_class`

## Sources

- NVIDIA CGF/STM/NvSciBuf/NvSciSync/DLA/FSI documentation
- Qualcomm QIM/QAIRT/FastRPC/QNN/Ride Flex documentation
- TI TIDL/TIOVX/IPC/WWDT documentation
- ROS 2, iceoryx, and Autoware architecture/QoS documentation

The full returned analysis is retained in the worker artifact:
`.omo/ultraresearch/20260723-cpu-npu-boundary/SYNTHESIS.md`.

## EXPAND

- LEAD: Exact vendor safety manuals/FMEDAs are access-controlled and must be
  obtained for a selected production SoC/SDK release.
- LEAD: Queue/fence/reset recovery timings require target-board fault injection.
- LEAD: Exact runtime/firmware release matrices are needed before product sign-off.
- DEAD END: Public stacks showed no materially different boundary pattern after
  two internal expansion waves.

These leads are recorded as product-validation gaps rather than general-market
research tasks because they require a selected vendor, NDA material, and hardware.
