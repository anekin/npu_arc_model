# Verification: Scenario Envelope Arithmetic

All values below use decimal GB/MB. They are arithmetic checks, not measured
target-NPU performance.

## Host tensor traffic

Formula:

`bytes/job = cameras * width * height * channels * bytes/channel`

Examples:

| Host-submitted tensor assumption | MB/job | MB/s |
|---|---:|---:|
| 2 x 512x512 RGB8 at 10 jobs/s | 1.573 | 15.729 |
| 3 x 512x512 RGB8 at 10 jobs/s | 2.359 | 23.593 |
| 3 x 512x512 RGB FP16 at 10 jobs/s | 4.719 | 47.186 |
| 3 x 384x384 RGB FP16 at 2 jobs/s | 2.654 | 5.308 |
| 8 x 1920x1920 RGB8 at 30 jobs/s | 88.474 | 2654.208 |

Conclusion: VLA observation transfer is usually modest after the host reduces the
input to a few policy images. Continuous high-resolution perception can reach
multi-GB/s and therefore requires bytes/job, copies, link efficiency, and
contention in the scenario. Camera link counts still do not belong to the NPU.

## Raw weight capacity

| Workload | INT4 | INT8 | BF16 |
|---|---:|---:|---:|
| SmolVLA 0.45B | 0.225 GB | 0.450 GB | 0.900 GB |
| π0 3.3B | 1.650 GB | 3.300 GB | 6.600 GB |
| Helix S2+S1 7.08B | 3.540 GB | 7.080 GB | 14.160 GB |
| Current S3 7B + 0.675B ViT + 0.3B action expert | 3.987 GB | 7.975 GB | 15.950 GB |

For the current 5 GB design, the last row leaves only 1.013 GB at nominal
capacity, or 0.513 GB if 90% is usable, before activations, cached features/KV,
runtime workspace, DMA buffers, and action queues. Capacity is therefore a
first-order constraint; the published 930 GB/s S3 bandwidth does not compensate
for an inability to keep the full workload resident.

## Rate separation

- π0 on a 50 Hz robot executes 25 actions before requesting the next chunk:
  `25 / 50 = 0.5 s`. Its published RTX 4090 model time of 73 ms consumes only
  14.6% of that refill window. This does not mean π0 runs 50 full inferences/s.
- Helix S2 at 7–9 Hz has a 142.9–111.1 ms period.
- Helix S1 at 200 Hz has a 5 ms period.
- Helix 02 S0 at 1 kHz has a 1 ms period.

## Fast-policy lower bounds

If every parameter is used once per invocation:

- 80M S1 x 200 Hz = 16 GMAC/s and 16 GB/s of INT8 weight traffic if weights
  are streamed from the modeled memory each time.
- 10M S0 x 1 kHz = 10 GMAC/s and 10 GB/s of INT8 weight traffic under the same
  conservative memory assumption.

These are lower bounds before vision features, attention, activations, and
runtime overhead. They show why weight residency and bounded interference matter
more than a single aggregate TOPS number.
