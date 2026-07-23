# CPU-host to NPU/GPU boundary synthesis

## Conclusion

The production boundary is an asynchronous, bounded, zero-copy job service. The host
owns graph lifecycle, model admission, buffer pools, timestamps, scheduling policy,
output validation, and degradation. The accelerator owns a precompiled, shape-bounded
tensor graph. A safety MCU or independent safety domain owns actuator permission and
fault recovery.

The unit crossing the boundary is not a raw pointer or a blocking function call. It is a
versioned descriptor containing immutable job metadata, opaque registered-buffer handles
plus offsets, tensor contracts, completion fences, deadline/priority, and a result/status
record. Buffers stay owned until the completion fence is observed.

## Normative boundary rules

1. Preallocate and register all pools before activation; no steady-state allocation.
2. Pass opaque handles/offsets, never process virtual addresses.
3. Negotiate shape, type, layout, stride, alignment, cacheability, and access rights.
4. Use explicit release/acquire fences and platform cache maintenance.
5. Bound every queue and state whether it is FIFO or latest-wins mailbox.
6. Carry capture time, sequence, validity window, submit deadline, completion deadline,
   priority, maximum in-flight jobs, and overload policy.
7. Treat timeout as an observation, not automatic cancellation; define cancel/reset and
   late-completion buffer reclamation separately.
8. Validate status, shape, numerical sanity, provenance, and age before publishing output.
9. Never treat automatic GPU/CPU operator fallback as a safety fallback.
10. Keep minimum-risk behavior and actuator gating independent from the NPU and rich OS.

## Failure ladder

- Overload: drop/replace stale input according to queue policy.
- Job miss: mark result unusable, preserve fence ownership, use last-known-valid only
  within a declared age budget.
- Recoverable accelerator fault: recreate graph/context and re-register pools.
- Repeated or fatal fault: isolate/restart the accelerator or main domain.
- Loss of host heartbeat or safety invariant: independent MCU/safety island gates output
  and requests minimum-risk behavior.

## Gaps

- Complete NVIDIA, Qualcomm, and TI safety manuals and FMEDAs are access-controlled.
- No target hardware was available, so WCET, cache-coherency, queue-overload, SSR, and
  reset timings must be measured on the exact SoC/SDK/firmware release.
- ROS 2/Autoware primitives are integration mechanisms, not a safety certification.
