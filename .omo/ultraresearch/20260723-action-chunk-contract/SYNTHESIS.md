# Ultraresearch synthesis: edge VLA action-chunk and async-inference contract

Sources: 16 primary paper/model/repository artifacts, plus more than 24 varied web queries. Repository revisions are pinned in `wave-2-web-action-contract.md`.

## Executive result

A single `VLA_FPS` field is invalid. The DSE needs at least: action emission rate, policy invocation/request rate, prediction horizon, execution horizon or emitted prefix, iterative passes, refill trigger, observation-age policy, and end-to-end deadline. Helix additionally requires separate fast- and slow-loop records.

The model-shape facts are π0 H=50 / 10 flow passes, SmolVLA H=50 / 10 flow passes, vanilla OpenVLA H=1 / no denoising, and Helix regression rather than an iterative diffusion/flow head. Only π0's and SmolVLA's horizons are directly amortizable. OpenVLA must complete one inference per action. Helix's public 200 Hz number is an action/control stream and process-level S1 rate; its proprietary action-chunk length is not public.

## Source-grounded temporal contracts

| System | Action execution | Policy invocation | Prediction horizon | Iterative passes | Queue/trigger | Public staleness behavior | Derived edge deadline |
|---|---:|---:|---:|---:|---|---|---|
| π0 published rollout | 20 Hz on UR5e/Franka; 50 Hz on other robots | 1.25 Hz after 16 actions at 20 Hz; 2 Hz after 25 actions at 50 Hz | 50 | 10 | No async queue; open-loop execution, no temporal ensemble | No age cutoff. Re-observation gap is 800 or 500 ms, not a certified tolerance | For a no-pause overlapped implementation, <=800 or <=500 ms end-to-end; subtract non-NPU overhead |
| π0.5 + RTC follow-on | 50 Hz | Dynamic, minimum 25 executed steps between starts, nominal about 2 Hz | 50 | 5 in RTC evaluation | `s_min=25`, delay buffer 10; not SmolVLA's fractional low-watermark | Tested at about d=16, or 320 ms observation-to-chunk age, with no degradation; mathematical no-idle cap is d<=25, or 500 ms | `<=(H-s)*dt = 500 ms` end-to-end for s=25; measured RTC model 97 ms and total 109–139 ms |
| SmolVLA async | 30 Hz in paper/harness | Event-driven, not fixed. At g=.7 and 50 emitted actions, first refill opportunity is after 15 ticks (2 Hz); below threshold the client can send every tick while the server serializes/filter requests | 50 fixed in checkpoint; emitted prefix typically 10–50, default/example 50 | 10 | Paper example g=.7; current code default g=.5; docs recommend tuning ~.5–.6 | Server retains one pending observation and replaces it with the newest; no wall-clock age rejection. Client drops returned actions whose timestep already executed and fuses overlaps | `<=g*A/f_action`: 1167 ms for g=.7,A=50,f=30; 833 ms for g=.5. `A` is emitted actions, not necessarily model H |
| Vanilla OpenVLA | 5–10 Hz repository recommendation; paper also has embodiment-specific 5 Hz and 15 Hz experiments | Equal to action rate because H=1 | 1 | None; autoregressive decode emits one token per action dimension | None | No explicit age cutoff or future-action buffer | 100–200 ms end-to-end for 10–5 Hz direct control |
| Helix | S1 action/control at 200 Hz | S2 asynchronous 7–9 Hz. S1 process is described at 200 Hz, but later Figure material says S1 outputs chunks; full-network calls/s are not disclosed | Undisclosed | None described; trained with standard regression | Latest S2 latent in shared memory, no public low-watermark | A training temporal offset is matched to deployment latency, but its numeric value and allowed latent age are undisclosed | 5 ms is a hard action-emission period, not safely a full-S1-model deadline without chunk length. S2 steady-state update budget is 111–143 ms but is soft because S1 continues on the latest latent |

## Hard DSE fields

These can be fixed from a selected checkpoint/deployment profile:

- `action_hz`: π0 profile 20 or 50; SmolVLA paper profile 30; OpenVLA scenario 5 or 10; Helix S1 200.
- `prediction_horizon_steps`: π0 50; SmolVLA 50; OpenVLA 1. Helix unknown must not be invented.
- `decoder_kind`: π0/SmolVLA flow; OpenVLA autoregressive; Helix regression.
- `default_iterative_passes`: π0 10; SmolVLA 10; OpenVLA/Helix 0. Keep it separately sweepable because RTC validly uses π0.5 with five.
- `output_action_tokens`: OpenVLA equals action dimension, normally seven for a 7-DoF action.
- `async_pending_obs_depth=1`, `pending_obs_policy=latest_wins`, and `drop_expired_action_steps=true` for the current LeRobot async runtime.
- `fast_loop_hz=200` and `slow_loop_hz=7..9` for Helix. Do not collapse them.

## Swept assumptions / scheduler fields

- `execution_horizon_steps`: π0 16/25 are paper profiles; SmolVLA can execute fewer than 50; RTC uses `s_min=25`.
- `emitted_chunk_steps`: SmolVLA transport/runtime prefix, typically 10–50, distinct from H=50.
- `queue_low_watermark_fraction`: SmolVLA g. Sweep at least 0.5, 0.6, 0.7; the sources do not establish one universal value.
- `policy_invocation_hz`: hard for direct OpenVLA, derived for fixed π0 profiles, and event-driven for SmolVLA. For SmolVLA record measured invocation demand rather than forcing the nominal trigger opportunity rate into this field.
- `max_observation_age_ms`: no examined implementation exposes a certified bound. Sweep it and report miss/drop behavior. RTC supplies evidence points around 120 ms baseline and 320 ms delayed, with a 500 ms feasibility cap for H=50,s=25,50 Hz.
- `deadline_npu_ms`: always derive from an end-to-end scheduling constraint after subtracting capture, preprocessing, postprocessing, transport, and runtime overhead.
- Helix `s1_chunk_steps`, `s1_full_model_hz`, `s1_npu_deadline_ms`, and numeric `s2_latent_max_age_ms`: all nondisclosed and therefore assumptions, not product facts.

## Required feasibility checks

1. Direct, unchunked policy: `latency_e2e <= 1/action_hz`.
2. Fractional queue refill: `latency_e2e <= g*emitted_chunk_steps/action_hz`.
3. RTC-style overlap: `floor(latency_e2e/dt) <= H-execution_horizon`.
4. Any returned chunk must contain at least one unexpired step after late-action dropping.
5. Track action deadline misses and stale-observation drops separately; a queue that avoids underflow can still act on semantically stale perception.

## Contradictions resolved

- SmolVLA's paper illustrates g=.7, current code defaults to .5, and current docs contain both a `.7` table entry and `.5` prose/example. This is evidence that g is a tuning variable, not a hard model parameter.
- Helix's initial description presents a 200 Hz S1 process; a later logistics post explicitly describes S1 action chunks sampled at 200 Hz. Therefore 200 Hz is safe as the action stream, but unsafe as a guaranteed full-network invocation rate.
- π0's “up to 50 Hz” describes action execution, not 50 complete VLA calls/s. The paper appendix gives the actual call cadence as 2 Hz for the 50 Hz robots.

## EXPAND

none — remaining missing values are explicitly nondisclosed rather than unsearched.
