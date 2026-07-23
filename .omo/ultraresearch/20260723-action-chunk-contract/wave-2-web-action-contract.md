# Wave 2 digest — temporal contract of edge VLA policies

Primary sources inspected:

1. π0 paper: https://www.physicalintelligence.company/download/pi0.pdf
2. openpi config at `15a9616a00943ada6c20a0f158e3adb39df2ccac`: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0_config.py#L18-L27
3. openpi sampler at the same revision: https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/models/pi0.py#L217-L278
4. RTC paper: https://www.physicalintelligence.company/download/real_time_chunking.pdf
5. SmolVLA paper: https://arxiv.org/pdf/2506.01844
6. SmolVLA model config: https://huggingface.co/lerobot/smolvla_base/blob/main/config.json
7. LeRobot SmolVLA config at `73dbb6f43a5088583706c91fb73c6957bca5f806`: https://github.com/huggingface/lerobot/blob/73dbb6f43a5088583706c91fb73c6957bca5f806/src/lerobot/policies/smolvla/configuration_smolvla.py#L24-L64
8. LeRobot async client at the same revision: https://github.com/huggingface/lerobot/blob/73dbb6f43a5088583706c91fb73c6957bca5f806/src/lerobot/async_inference/robot_client.py#L403-L479
9. LeRobot async server at the same revision: https://github.com/huggingface/lerobot/blob/73dbb6f43a5088583706c91fb73c6957bca5f806/src/lerobot/async_inference/policy_server.py#L173-L328
10. LeRobot similarity filter at the same revision: https://github.com/huggingface/lerobot/blob/73dbb6f43a5088583706c91fb73c6957bca5f806/src/lerobot/async_inference/helpers.py#L276-L298
11. LeRobot async guide at the same revision: https://github.com/huggingface/lerobot/blob/73dbb6f43a5088583706c91fb73c6957bca5f806/docs/source/async.mdx
12. OpenVLA repository at `c8f03f48af692657d3060c19588038c7220e9af9`: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/README.md#L618-L627
13. OpenVLA single-action decoder at the same revision: https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/models/vlas/openvla.py#L35-L101
14. OpenVLA paper: https://arxiv.org/pdf/2406.09246
15. Figure Helix primary description: https://www.figure.ai/news/helix
16. Figure Helix logistics/action-chunk description: https://www.figure.ai/news/helix-logistics

All sources accessed 2026-07-23.

## Key findings

- π0 predicts H=50 actions with 10 Euler/flow passes. Its published controller does not invoke the policy at 50 Hz: it executes at 20 or 50 Hz and replans after 16 or 25 actions, respectively (1.25 or 2 policy calls/s).
- RTC is a distinct deployment mode, not the original π0 scheduler. Its real-world configuration uses π0.5 at 50 Hz, H=50, five flow passes, and s_min=25. It reports 97 ms model latency and 109–139 ms total latency; +200 ms injection corresponds to about d=16 controller steps.
- SmolVLA's checkpoint fixes the predicted shape at 50 actions and defaults to 10 flow passes. `actions_per_chunk` may transmit a shorter prefix. The client trigger is `queue_size / received_chunk_size <= g`. The paper's representative async setting is g=0.7; current code defaults to 0.5.
- SmolVLA does not have a fixed policy frequency. Below the threshold the client may send observations every control tick; the server serializes inference, retains only one pending observation, replaces the pending one with the newest, and filters near-duplicate joint states. Returned actions at already-executed timesteps are dropped.
- Vanilla OpenVLA has H=1, no denoising, and no queue. Its repository recommends direct 5–10 Hz control because it was not trained with action chunking. Policy and action frequency therefore coincide.
- Helix publicly fixes a 200 Hz S1 action/control stream and a 7–9 Hz asynchronous S2 stream. It trains with a temporal offset matched to deployment latency and consumes the latest S2 latent via shared memory. Figure later says S1 emits action chunks sampled at 200 Hz, but does not publish chunk length or prove that the complete S1 network is invoked 200 times/s.

## EXPAND

none — all requested public fields are resolved; the remaining Helix values are proprietary/nondisclosed.
