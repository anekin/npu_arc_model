# Expansion log — action-chunk / asynchronous VLA inference

Date: 2026-07-23

## Wave 1 leads inherited

- π0: distinguish 50-step prediction horizon from the 16/25-step execution horizon and 10-step flow integration.
- SmolVLA: resolve the async queue trigger, actual policy invocation cadence, and observation staleness handling.
- OpenVLA: verify whether 5–10 Hz refers to action control or chunk generation.
- Helix: separate its 200 Hz fast loop from its 7–9 Hz slow loop.

## Wave 2 opened and closed

- Opened π0 paper appendix and pinned openpi source. Closed: H=50 and 10 flow steps are model defaults; published rollout invokes every 16 steps at 20 Hz or every 25 steps at 50 Hz.
- Opened RTC paper as a follow-on async contract. Closed: real-world π0.5 uses H=50, 50 Hz, 5 denoising steps, s_min=25; no-idle feasibility requires inference delay d <= H-s.
- Opened SmolVLA paper, Hub config, current LeRobot docs, and pinned runtime source. Closed: model horizon is 50, default flow steps are 10, while actions_per_chunk and queue fraction g are runtime controls. The paper illustrates g=0.7; current code defaults to 0.5; current docs are internally inconsistent and recommend tuning around 0.5–0.6.
- Opened OpenVLA paper/repository and pinned inference source. Closed: vanilla OpenVLA predicts one action vector, autoregressively emitting one token per action dimension; it has no action chunk or denoising loop. Repository guidance is 5–10 Hz.
- Opened Figure's Helix and logistics posts. Closed: 200 Hz action/control and 7–9 Hz S2 cadence are public, but the S1 action-chunk length and full-network invocation cadence are not.

## Convergence

No unchecked source lead remains for the requested fields. The remaining unknowns are nondisclosures, not discoverable values: Helix S1 chunk length, Helix numeric temporal offset, and any vendor-certified wall-clock stale-observation cutoff.
