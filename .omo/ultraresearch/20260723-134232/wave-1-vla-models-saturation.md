# Wave 1: VLA Model Saturation Pass

## Outcome

A 48-primary-source, eight-pinned-repository pass confirms that the market spans
35M direct-action policies through 7B+ VLAs, and that action execution rate is
frequently much higher than policy replanning rate.

## Selected anchors

- RT-1: 35M, six 300x300 frames, single action, 3 Hz.
- OpenVLA: 7B, one 224x224 image, one 7D discretized action, no chunk/history;
  about 6 Hz and 16.8 GB BF16 on RTX 4090.
- Octo: 27M/93M policy plus frozen 111M T5, four-step diffusion chunk,
  20 denoise steps.
- π0: 3.3B, 2–3 cameras, H=50, 73 ms model time on RTX 4090,
  but deployed replanning can be 2 Hz.
- π0.7: approximately 5B, up to four 448x448 cameras and six history frames,
  H=50, five denoise steps; no released edge-NPU benchmark.
- GR00T N1: 2.2B, H=16, four denoise steps, 63.9 ms BF16 on L40.
- GR00T N1.7: 3B BF16, H=40, four denoise steps; published one-camera
  TensorRT rates are 10.7 Hz on AGX Thor and 4.6 Hz on Orin, separate from
  roughly 30 FPS action execution; minimum inference memory is 16 GB.
- SmolVLA: 450M, 512x512 multi-camera, H=50, 32D-padded actions,
  ten flow steps; no reliable named-device latency/memory benchmark.
- ACT: approximately 80M, four 480x640 images, 100x14 action chunk,
  about 10 ms on RTX 2080 Ti and 50 Hz control.
- Diffusion Policy: approximately 89M/102M image policy examples,
  two observations, 16-step prediction/eight-step execution; 100 ms for ten
  DDIM steps on RTX 3080 in the methods setup.

## EXPAND

- FUTURE RELEASE GAP: immutable Gemini cards and π0.7 code/weights.
- MEASUREMENT GAP: SmolVLA named-device latency/memory and OpenVLA fused encoder.
- BENCHMARK GAP: multi-camera/whole-body GR00T N1.7; current public benchmark is
  one-camera policy inference.

Full worker artifact:
`.omo/ultraresearch/20260723-vla-models/SYNTHESIS.md`.
