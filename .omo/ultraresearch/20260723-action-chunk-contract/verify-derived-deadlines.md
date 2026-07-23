# Verification — derived timing constraints

The source facts were converted to deadlines with:

- periodic direct policy: `deadline_e2e <= 1 / action_hz`;
- chunk refill at low watermark `g`: `deadline_e2e <= g * emitted_chunk_steps / action_hz`;
- nominal first refill trigger interval: `(1-g) * emitted_chunk_steps / action_hz`;
- RTC no-idle feasibility: `d <= H-s`, where `d=floor(latency/dt)`;
- NPU-only deadline: `deadline_npu = deadline_e2e - capture - preprocess - postprocess - transport - runtime`.

Executed calculation output:

```text
pi0-50Hz: policy_hz=2.000, reobserve_interval=500.0ms
pi0-20Hz: policy_hz=1.250, reobserve_interval=800.0ms
smolvla-paper-g0.7: remaining=35 actions, trigger_after=15 actions, trigger_hz=2.000, e2e_no_underflow_deadline=1166.7ms
smolvla-code-g0.5: remaining=25 actions, trigger_after=25 actions, trigger_hz=1.200, e2e_no_underflow_deadline=833.3ms
openvla-5Hz: policy_hz=5.000, reobserve_interval=200.0ms
openvla-10Hz: policy_hz=10.000, reobserve_interval=100.0ms
helix-S1 action period: 5.0 ms
helix-S2 periods: 111.1 to 142.9 ms
rtc tested approximate age at d=16, dt=20ms: 320 ms; feasibility cap d=25: 500 ms
```

Verdict: CONFIRMED as arithmetic consequences of the cited source configurations. The SmolVLA “trigger Hz” is an opportunity rate, not a guarantee of actual inference frequency, because the implementation can issue a request burst, filter observations, and serialize inference.
