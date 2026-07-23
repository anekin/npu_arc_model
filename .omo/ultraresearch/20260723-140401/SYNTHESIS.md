# Ultraresearch Synthesis: Dual-rate robot inference and accelerator scheduling

## Executive conclusion

The reference architecture should use three scheduling classes:

1. semantic/background S2 jobs at roughly 2--9 Hz;
2. reactive learned S1 jobs or chunk refills at roughly 15--200 Hz;
3. a 1 kHz learned whole-body controller only when required, followed by
   independent 1--4 kHz position and greater-than-10-kHz motor-current/safety
   loops on MCUs.

The fast learned policy belongs on an NPU/GPU/DSP with resident weights. The
final motor and safety loops belong on deterministic MCUs. A 10M dense S0 at
1 kHz implies at least roughly 10 GMAC/s and 10 GB/s of INT8 weight reads before
activation overhead; it should not be assigned to a conventional MCU without
measured worst-case execution proof.

## DSE task graphs

### Helix

```text
sensor snapshot ─┬─> S2[7B, 7--9 Hz, background, GPU/NPU-A]
                 │          └─> latest-latent mailbox
                 └─> S1[80M, 200 Hz, D=5 ms, GPU/NPU-B]
                              └─> joint-target mailbox
```

### Helix 02

```text
Helix S2/S1 graph
  -> S1 whole-body targets[200 Hz]
  -> S0[10M, 1 kHz, D=1 ms, isolated NPU/DSP]
  -> joint MCU[position 1--4 kHz; current >10 kHz; safety/watchdog]
```

### GR00T / pi0-style chunked VLA

```text
observation
  -> VLM/S2 feature forward
  -> flow/diffusion action expert, N solver steps
  -> future-action chunk queue
  -> periodic executor at robot control rate
  -> joint MCU
```

The refill deadline is not the action-sample period:

```text
D_refill <= H_executed / f_control - communication_guard
```

### G0

```text
G0-VLM[<2 Hz] -> latest subtask mailbox
  -> G0-VLA[15 Hz] -> action-chunk queue
  -> control executor[200 Hz]
  -> joint MCU
```

## Scheduling requirements

- Priorities: motor MCU > S0 > S1/chunk executor > S2.
- S0 and S1 must never block waiting for S2.
- Shared state uses latest-value, timestamped double buffers with atomic
  generation swap; no FIFO backlog for semantic latents.
- If one accelerator is shared, include non-preemptive blocking:
  `B_fast = max(lower-priority kernel quantum)`. Stream priority alone is
  insufficient because running CUDA kernels are not preempted.
- Strong options are dedicated accelerators, MIG/resource partitions, or
  time-triggered admission of S2 only into proved slack.
- Keep all fast-path weights resident in local accelerator memory; disallow
  model swapping during control.
- Memory constraint:
  `sum(weights + activations + KV/cache + queue buffers) <= partition memory`.
- Bandwidth constraint:
  `sum(bytes_per_job / period) <= effective bandwidth`, checked at P99/P999
  under camera DMA, ISP, CPU, and concurrent model traffic.
- Tail-latency constraint:
  `P(WCRT_i <= D_i) >= required safety target`; average latency is not enough.
- For chunked policies, DSE variables include generated horizon, executed
  horizon, refresh threshold, fusion/inpainting rule, stale observation age,
  and fallback queue reserve.

## Raw weight-residency estimates

These are parameter-only capacities, excluding activations, KV caches, runtime,
and allocator overhead.

| Model component | Parameters | BF16 | INT8 | INT4 |
|---|---:|---:|---:|---:|
| Helix S2 | 7B | 14 GB | 7 GB | 3.5 GB |
| Helix S1 | 80M | 160 MB | 80 MB | 40 MB |
| Helix 02 S0 | 10M | 20 MB | 10 MB | 5 MB |
| GR00T N1 total | 2.2B | 4.4 GB | 2.2 GB | 1.1 GB |
| GR00T N1 VLM | 1.34B | 2.68 GB | 1.34 GB | 0.67 GB |
| GR00T non-VLM remainder | 0.86B | 1.72 GB | 0.86 GB | 0.43 GB |
| RoboDual specialist | 20M | 40 MB | 20 MB | 10 MB |
| SmolVLA total | 0.45B | 0.9 GB | 0.45 GB | 0.225 GB |
| pi0 VLM + expert, approximate | 3.3B | 6.6 GB | 3.3 GB | 1.65 GB |

## Gaps

- Figure does not disclose Helix 02 resource placement, S2 update rate, or
  whether S1/S2 retain the original exact parameter counts.
- GR00T N1 does not publish independent S1/S2 runtime rates; its "dual system"
  label should not be modeled as asynchronous until a deployment proves it.
- Accelerator WCET, operator support, quantization accuracy, and thermal
  throttling must be measured on the target NPU.

## EXPAND

none — search converged after three waves; remaining items are vendor-private or
target-hardware measurements rather than further public-source leads.
