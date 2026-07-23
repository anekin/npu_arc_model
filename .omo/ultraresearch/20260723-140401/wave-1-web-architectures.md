# Wave 1 digest: dual-rate robot architectures

## Figure Helix

- Official release: https://www.figure.ai/news/helix
- S2 is a 7B VLM at 7--9 Hz; S1 is an 80M cross-attention
  encoder-decoder policy at 200 Hz.
- S2 and S1 run as separate processes on dedicated embedded GPUs.
- S2 asynchronously overwrites a shared-memory latent; S1 consumes the latest
  observation and most recent latent without blocking.
- Training injects an observation offset matching deployed S2 latency.

## Figure Helix 02

- Official release: https://www.figure.ai/news/helix-02
- S2 is semantic planning, S1 outputs whole-body joint targets at 200 Hz, and
  S0 is a 10M learned whole-body controller at 1 kHz.
- S0 inputs full-body joint state and base motion and outputs joint-level
  actuator commands.
- The vendor does not disclose whether S0 has a dedicated accelerator, whether
  Helix 02 retains Helix's exact 7B/80M sizes, or the S2 update rate.

## NVIDIA GR00T N1

- Paper: https://arxiv.org/abs/2503.14734
- Public checkpoint totals 2.2B parameters, including 1.34B in the VLM.
- A 16-action chunk takes 63.9 ms on an L40 GPU in BF16.
- The VLM output conditions a DiT flow-matching action head; the paper presents
  serially coupled inference, not independently scheduled S2/S1 rates.
- Public configuration uses action horizon 16 and 16 inference timesteps:
  https://huggingface.co/nvidia/GR00T-N1-2B/blob/main/config.json

## Comparable systems

- HiRT paper: https://arxiv.org/abs/2410.05273
  - InstructBLIP 7B S2 plus compact latent-conditioned S1.
  - S2 writes a latent cache; S1 uses the latest latent asynchronously.
  - 9.8 Hz measured inference versus 4.1 Hz for monolithic VLA.
- RoboDual paper: https://arxiv.org/abs/2410.08001
  - OpenVLA 7B generalist plus 20M trainable DiT specialist.
  - One generalist result conditions eight specialist steps.
  - 15 Hz measured control versus 3.9 Hz for OpenVLA on A5000 Ada.
  - Training explicitly offsets/delays generalist outputs.
- Fast-in-Slow paper: https://arxiv.org/abs/2506.01953
  - 7B VLM; System 1 repurposes two final VLM blocks.
  - 1:4 S2:S1 frequency ratio; 21.9 Hz with one action, 117.7 Hz with
    action chunk eight on RTX 4090.
- SmolVLA paper: https://arxiv.org/abs/2506.01844
  - 0.45B total and about 100M action expert.
  - Async queue refresh at 70% remaining, overlapping action execution and
    inference, with chunk fusion at the boundary.
- pi0 paper: https://www.physicalintelligence.company/download/pi0.pdf
  - Up to 50 Hz action output, but on 50 Hz robots inference runs every 0.5 s
    after 25 actions; 73 ms onboard latency on RTX 4090.
- Galaxea G0 paper: https://arxiv.org/abs/2509.00576
  - Less than 2 Hz S2 subtask instruction, 15 Hz S1 motion planning/action
    chunks, and 200 Hz robot control.
  - G0-VLA checkpoint is approximately 3B:
    https://github.com/OpenGalaxea/G0

## EXPAND

none — all architecture leads needed for the requested DSE translation were
closed; unpublished vendor implementation details remain explicit gaps.
