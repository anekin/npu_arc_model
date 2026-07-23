# Wave 2: Action-Chunk Contract

## Outcome

`action_hz`, `policy_invocation_hz`, `prediction_horizon_steps`, emitted prefix,
denoise/flow passes, and queue low-watermark are independent fields. A single
`VLA FPS` field is incorrect.

## Public reference points

- π0: H=50, 10 flow passes; policy invocation is 1.25 Hz after 16 actions on
  20 Hz robots and 2 Hz after 25 actions on 50 Hz robots. Derived no-pause
  budgets are 800 ms and 500 ms minus non-NPU overhead.
- π0.5 + real-time chunking: nominal 50 Hz actions, H=50, five passes in the
  evaluated setup; measured 97 ms model and 109–139 ms total on RTX 4090.
- SmolVLA async: 30 Hz runtime profile, event-driven latest-wins refill rather
  than a fixed policy rate, H=50, 10 flow passes.
- Vanilla OpenVLA: H=1, no denoising, 5–10 Hz recommended direct control;
  end-to-end budget is therefore 100–200 ms.
- Helix: S1 output at 200 Hz and S2 at 7–9 Hz, but public S1 chunk length and
  exact full-network invocation rate are undisclosed.

## Required fields

- `action_hz`
- `policy_invocation_hz` or event-driven trigger
- `prediction_horizon_steps`
- `emitted_actions_per_chunk`
- `denoise_or_flow_steps`
- `queue_low_watermark`
- `max_observation_age_ms`
- `pending_observation_depth`
- `merge_or_fusion_policy`
- `expired_action_policy`

Always derive:

`deadline_npu = deadline_e2e - capture - pre/postprocess - transport - runtime`

## EXPAND

None. Remaining Helix values are proprietary/nondisclosed.

Full worker artifact:
`.omo/ultraresearch/20260723-action-chunk-contract/SYNTHESIS.md`.
