# Wave 3: Independent Requirement-Envelope Check

## Outcome

The independent lane converged on the same four profiles and the same primary
rule: action execution, policy refresh, tensor arrival, chunk horizon, and
denoise steps must be separate. TOPS ranges can seed DSE but cannot serve as
cross-vendor acceptance criteria.

## Corrections imported

- Current SmolVLA configuration is 512x512, H=50, 10 flow steps; public LeRobot
  material reports roughly 2 GB inference memory for its reference runtime.
- Public LeRobot material reports roughly 14 GB inference memory for π0,
  implying raw-weight arithmetic alone is insufficient for capacity sizing.
- GR00T N1.7 independently corroborates a 16 GB minimum inference-memory class
  for a current approximately 3B VLA implementation.
- For high-resolution continuous perception, host-transfer requirements must be
  computed from the CPU's actual post-preprocess tensor trace, not raw camera
  marketing counts.

## Engineering assumptions retained for DSE, not vendor claims

- Compact VLA search: 10–20 dense INT8 TOPS, 4/8 GB, 5–15 Hz policy.
- π0-class search: 50–100 dense INT8 TOPS, 16/24 GB where matching published
  runtime memory is a goal.
- Helix-class search: 100–200 dense INT8 TOPS, 16–32 GB, 5 ms fast deadline
  and 0.5–1 ms preemption response or hard partition.
- Continuous perception: 32–100 dense INT8 TOPS, 8–32 GB depending on model
  count and tensors, with explicit host-link trace and deadline queues.

These are exploration envelopes. A complete compiled graph, accuracy gate,
measured memory footprint, and mixed-load P99 remain the sign-off criteria.

## EXPAND

- TARGET PROFILE: compile and measure SmolVLA/π0 operator coverage, MACs,
  activation peaks, and quantization quality.
- SIMULATOR WORK: compare no-preemption, iteration-boundary preemption, and
  dual-partition scheduling for the 7B+80M case.
- APPLICATION TRACE: extract actual post-CPU tensor shapes/dtypes/arrival times.
- DEAD END: public marketing TOPS cannot yield a unique architecture.
