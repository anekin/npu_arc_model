# Wave 1 Direct Web Research: VLA Workloads

## Key findings

- OpenVLA is a 7B VLA with fused SigLIP+DINOv2 vision encoders and a Llama-2-7B action-token backbone. The official repository recommends 5–10 Hz data/control because vanilla OpenVLA has no action chunking; later OFT/FAST paths exist specifically to avoid autoregressive action latency.
  - https://openvla.github.io/
  - https://github.com/openvla/openvla
- π0 is 3.3B total: a 3B PaliGemma VLM plus a 300M action expert. It ingests 2–3 RGB images, language, and proprioception; predicts an H=50 continuous action chunk; runs 10 flow-matching integration steps; and caches the observation prefix. It supports action execution up to 50 Hz, which is not the same as 50 full VLA inferences per second.
  - https://www.physicalintelligence.company/download/pi0.pdf
- SmolVLA is 450M, accepts multiple RGB views/state/language, reduces each frame to 64 visual tokens, skips half the VLM layers, and uses a roughly 100M flow-matching action expert. Its official asynchronous runtime overlaps chunk prediction with chunk execution.
  - https://huggingface.co/blog/smolvla
  - https://huggingface.co/docs/lerobot/en/smolvla
- Figure Helix is the clearest production dual-rate disclosure: 7B System 2 at 7–9 Hz and 80M System 1 at 200 Hz, deployed on two dedicated low-power embedded GPUs with asynchronous shared state. Helix 02 adds a 1 kHz System 0 below the AI policy.
  - https://www.figure.ai/news/helix
  - https://www.figure.ai/news/helix-02
- NVIDIA GR00T N1 likewise uses a slow VLM reasoner plus a diffusion-transformer action model; the original public backbone was SmolLM-1.7B-based, while current GR00T 1.7 uses a roughly 2B Qwen3-VL-derived backbone.
  - https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/
  - https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/
- Gemini Robotics On-Device confirms the commercial direction toward local low-latency VLA, but its parameter count and hardware latency remain undisclosed; its model card gives 1088 input tokens.
  - https://deepmind.google/models/gemini-robotics/gemini-robotics-on-device/

## EXPAND

- LEAD: action execution Hz versus policy refresh Hz — WHY: current S3 conflates frame FPS, token TPS, and action rate — ANGLE: action queue/chunk scheduling sources
- LEAD: flow/diffusion operator workload — WHY: current simulator only models CV and autoregressive LLM traces — ANGLE: repeated action-expert forward passes and prefix caching
- LEAD: dual-rate resource isolation — WHY: a fast 5 ms policy deadline cannot wait behind a 7B reasoning job — ANGLE: partitions, preemption, or separate engine instances
