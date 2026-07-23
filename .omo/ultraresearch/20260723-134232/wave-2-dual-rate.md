# Wave 2: Dual-Rate and Action-Chunk Runtime

## Outcome

"Dual system" does not automatically mean independently scheduled dual-rate
runtime. Figure Helix and Galaxea G0 disclose asynchronous hierarchies; GR00T N1
is a coupled VLM-to-DiT action-chunk pipeline without published independent
S1/S2 scheduling rates.

## Published task graphs

- Figure Helix: 7B S2 at 7–9 Hz and 80M S1 at 200 Hz, on dedicated embedded
  GPUs with latest-latent asynchronous communication.
- Helix 02: adds a 10M S0 at 1 kHz below S1; placement and exact retained S1/S2
  sizes/rates are not fully disclosed.
- Galaxea G0: semantic VLM below 2 Hz, VLA planner at 15 Hz, executor at 200 Hz.
- π0: 3.3B, H=50, 10 flow steps; for 50 Hz robots, inference runs every 0.5 s
  after 25 actions, while published onboard RTX 4090 model time is 73 ms.
- SmolVLA: 450M with an approximately 100M action expert; async refill starts
  before the queue empties and merges overlapping chunks.
- GR00T N1: 2.2B coupled VLM/DiT pipeline; public 63.9 ms is chunk-generation
  latency, not proof of the downstream action/control frequency.

## Deadline model

For a chunked policy:

`refill_deadline <= executed_actions / action_execution_hz - communication - guard`

The deadline is not the individual action period. The scenario must separately
store action execution frequency, policy invocation frequency, chunk horizon,
executed actions before refill, and queue low-watermark.

## Shared-NPU requirements

- S1/action executor must never wait synchronously for S2.
- Use timestamped, generation-counted double buffers and latest-value mailboxes.
- Model non-preemptive blocking explicitly; a 1 ms learned S0 deadline generally
  requires a dedicated engine/partition or sub-millisecond scheduling quanta.
- Fast-path weights must remain resident; live model swapping is incompatible
  with 1–5 ms deadlines.
- Weight capacity must include activations, cached features/KV, workspace,
  transfer buffers, and action queues.
- Final current/position/FOC/PWM/watchdog/actuator gating belongs on a
  deterministic MCU or safety processor, not this NPU.

## Useful lower bounds

- An 80M dense S1 at 200 Hz is approximately 16 GMAC/s and 16 GB/s of INT8
  weight reads before token/CNN/activation overhead if weights are not retained
  in a closer cache.
- A 10M dense S0 at 1 kHz is approximately 10 GMAC/s and 10 GB/s of INT8
  weight reads under the same conservative assumption.

## EXPAND

None. Three waves converged. Remaining gaps are vendor-private Helix 02 details
and target-silicon WCET/thermal measurements.

Full worker artifact:
`.omo/ultraresearch/20260723-140401/SYNTHESIS.md`.
