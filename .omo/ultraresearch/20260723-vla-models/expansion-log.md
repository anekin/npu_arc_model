# Expansion log

Core question: compare disclosed deployment-relevant properties of RT-1/2, OpenVLA, Octo, Physical Intelligence π-family, Gemini Robotics, GR00T, SmolVLA, ACT, and Diffusion Policy.

## Search coverage

- More than 40 varied web searches across arXiv, official project/company pages, official model cards, Hugging Face, and first-party GitHub repositories.
- Full PDFs downloaded and converted to searchable text for RT-1, RT-2, OpenVLA, Octo, π0, π0.5, π0.5 Knowledge Insulation, π0.7, Gemini Robotics, Gemini Robotics 1.5, GR00T N1, SmolVLA, ACT, and Diffusion Policy.
- Eight repositories inspected at immutable SHAs, including source configs for image preprocessing, horizons, dimensions, context windows, precision, and runtime hardware guidance.

## Wave 1: original publications

Status: complete.

Covered architecture scale, inputs, cameras/resolution, observation context, actions/chunks, frequency/latency, hardware, precision/memory, and explicit omissions for every requested original model.

Key reconciliation findings:

- Network/model latency is not control-loop latency (RT-1: 15 ms network, fixed 280 ms observation-to-action wait).
- Controller rate is not necessarily achieved model rate (OpenVLA; π0).
- Policy families have task-specific configs rather than a single universal model (Diffusion Policy; Octo downstream frequencies).
- Published parameter totals can omit frozen encoders (Octo) or be rounded differently by model metadata (OpenVLA).

## Wave 2: current successors and deployment docs

Status: complete.

Expanded:

- π0/π0.5 to current π0.7 (April 2026).
- Gemini Robotics to On-Device and current Gemini Robotics 1.5; identified Robotics-ER 1.6 as a VLM rather than a VLA.
- GR00T N1 to current GR00T N1.7 (July 2026), including device-specific eager/TensorRT policy replanning rates.

Key reconciliation findings:

- GR00T N1.7 explicitly separates ~30 FPS action/camera execution from policy replanning rate.
- Gemini Robotics explicitly combines ~250 ms end-to-end chunk latency with 50 Hz effective action execution.
- π0.7 separates 38–127 ms low-level VLA inference from a 1.25 s asynchronous 14B subgoal world model.
- The public openpi π0.5 implementation omits the paper's high-level autoregressive subtask prediction stage.

## Wave 3: missing-disclosure audit

Status: complete.

Each model was checked for absent values instead of filling gaps with backbone defaults or secondary-source guesses. Unresolved leads are recorded in the synthesis's `## EXPAND` section.
