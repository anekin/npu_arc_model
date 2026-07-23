# Wave 1: Market Platform Saturation Pass

## Outcome

An official-source-only pass with more than 60 query variants, 67 URLs, and two
internal expansion waves confirms that bare TOPS cannot be used as the comparison
axis. Precision, sparsity, included engines, memory, power scope, concurrency,
real-time isolation, and safety claim must be independent fields.

## Key reference points

- Jetson Thor T5000: 128 GB, 273 GB/s, 40–130 W; 2070 is sparse FP4.
- Jetson Thor T4000: 64 GB, 273 GB/s, 40–70 W.
- Qualcomm IQ10: 350 dense / 700 sparse INT8 TOPS, multicore NPUs, 64 GB
  reference design; public maximum power and DRAM bandwidth are unavailable.
- Qualcomm IQ-9075: 50/100 dense INT8 TOPS, up to 36 GB LPDDR5, isolated
  real-time subsystem; public maximum power remains unclear in current material.
- TI TDA4VH/AM69A: 32 INT8 TOPS, up to 68 GB/s, distinct CPU/vision/MMA
  domains; safety scope differs by exact product.
- Hailo-10H: 20 INT8 TOPS, 4/8 GB local DRAM, 2.5 W typical.
- Hailo-8/8L: 26 TOPS at 2.5 W and 13 TOPS at 1.5 W respectively.
- NXP i.MX95: 2-TOPS NPU plus application, real-time, and safety domains;
  measured rail figures are workload measurements, not board TDP.
- NXP Ara240: proprietary 40 eTOPS at 6.5 W typical; model throughput is more
  defensible than the eTOPS label.
- Renesas RZ/V2H: 8 dense INT8 TOPS, not the 80 sparse headline; 25.6 GB/s raw
  DRAM bandwidth and dual real-time cores.

## Requirement consequences

- Store `peak_value`, `precision`, `dense_or_sparse`, and `scope` together.
- Treat unknown bandwidth/power as unknown, never impute them from TOPS/W claims.
- Evaluate identical compiled workloads at a fixed accuracy/quantization point.
- Include simultaneous perception/policy jobs and report p99 under contention.
- Keep sensor-ingress capability outside the NPU die requirement when the host
  owns sensors, but retain host-transfer bytes, copies, rate, and deadlines.

## EXPAND

- PRODUCT-SPECIFIC GAP: Gated IQ10, Ambarella, Ara240, and Snapdragon Ride data.
- HARDWARE GAP: Identical-workload benchmarks and simultaneous-load p99.
- CERTIFICATION GAP: Safety manuals and reset/isolation claims.
- FUTURE TRACKING: Final production IQ10 modules and unreleased/publicly
  incomplete platform variants.

Full worker artifact:
`.omo/ultraresearch/20260723-134614/SYNTHESIS.md`.
