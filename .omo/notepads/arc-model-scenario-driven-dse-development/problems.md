
## Todo 13: Deterministic Scheduler Kernel — Known Limitations

### Current limitations
- **Legacy adapters are wrappers, not full event simulations**: `CoreTimeline` still advances time synchronously inside each method; the kernel is used for ps conversion and event recording. True concurrent multi-resource scheduling (e.g., MXU + DMA + NoC simultaneously with preemption) is available in the kernel API but not yet exercised by `npu_sim.py`.
- **Byte server QoS is work-conserving but not preemptive**: Strict priority gives the next slot to the highest-priority waiting job, but an in-flight lower-priority transfer is not preempted mid-transfer.
- **Admission controller peak-bandwidth window is fixed**: It uses a configurable sliding window but does not integrate with the byte server's actual schedule.
- **No persistence/orchestration integration yet**: The scheduler kernel is in place, but `sim/arc_model.py` and `design_space_explorer.py` do not yet use it for end-to-end simulation.

### Not a bug
- `npu_sim.py` still sets `timeline._current_cycle` directly after `add_dma_parallel`. This is intentionally supported as a backward-compatibility alias.
- CLI baselines are intentionally unchanged; the goal of Todo 13 was to introduce deterministic infrastructure, not to change performance numbers.
