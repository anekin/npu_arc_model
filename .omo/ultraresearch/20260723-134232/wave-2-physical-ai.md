# Wave 2: Continuous-Perception Physical AI

## Outcome

The portable NPU requirement is a set of timestamped tensor jobs with deadlines,
not a camera-count requirement. Camera/ISP/codecs, sensor synchronization, and
motor/vehicle control remain on the host in this project's partition.

## Official workload anchors

TI AM69A public workload examples provide system-level anchors:

- AI box: 12 x 2 MP at 30 FPS, inference at 12 FPS, 12 aggregate TOPS,
  9.49 GB/s total SoC DDR.
- Machine vision: 3 x 8 MP at 30 FPS, ROI inference at 10–30 FPS,
  24 TOPS, 15.35 GB/s.
- Multi-camera AI: 8 x 2 MP at 30 FPS, detection at 30 FPS,
  24 TOPS, 15.13 GB/s.

These numbers must be converted to the post-host tensor DAG, bytes/job, job rate,
and accelerator deadline instead of copied as sensor-interface requirements.

## Scheduling and robustness requirements

- Offline graph compilation and persistent resident weights
- Async descriptors with job/frame/context ID, capture timestamp, absolute
  deadline, stale-after time, priority/criticality, handles, and fences
- At least three service classes; EDF within class; latest-frame replacement
- Measured maximum non-preemptible interval target of at most 1–2 ms
- Manifest for static/dynamic/scratch memory, alignment, average/peak bytes,
  and maximum execution segment
- Initial bandwidth admission at no more than 70–75% of measured sustainable
  bandwidth, validated against peak traffic
- IOMMU/domain isolation, ECC/parity, per-context watchdog, hierarchical reset
- Queue/start/end timestamps, DDR bytes, core assignment, and fault counters

## Proposed acceptance SLOs

These are engineering proposals, not vendor guarantees:

- Critical 30 FPS perception: P99 submit-to-complete at most 25 ms
- 20 Hz localization/fusion: P99 at most 37.5 ms
- 10–12 FPS inspection/auxiliary: P99 at most 62.5–75 ms
- Hard stale/timeout bound no greater than the job period

No vendor publishes a general mixed-workload P99 guarantee.

## EXPAND

- HARDWARE GAP: 10k+ frame mixed-workload characterization at 70–90% DDR load
  and thermal steady state.
- HARDWARE GAP: ECC, illegal DMA, stuck graph, watchdog, and reset fault injection.
- VENDOR GAP: IQ-9075 and R-Car V4H schedulability, live-context limits, reset
  granularity, and IOMMU details.

Full worker artifact:
`.omo/ultraresearch/20260723-physical-ai-wave2/SYNTHESIS.md`.
