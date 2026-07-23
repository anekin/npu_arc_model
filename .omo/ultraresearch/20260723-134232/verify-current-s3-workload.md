# Verification: Current S3 Workload Model

Environment: repository Python 3, `PYTHONPATH=sim`

Executed checks:

1. Regenerated the four-crop Qwen-VL trace and summed GEMM MACs.
2. Analytically computed the omitted QK^T and attention-value matrix multiplications.
3. Compared Qwen-VL trace SFU operation names against the CV simulator dispatch set.
4. Recomputed current S3 pipeline and single-engine serial FPS.
5. Computed raw weight capacity for representative market model sizes.

Key stdout:

```text
trace_macs=2582579773440 (2.582580 TMAC)
analytical_qk_av_missing=344268800000 (0.344269 TMAC)
corrected_min_macs=2.926849 TMAC
undercount_pct=11.76%
trace_sfu_types=['gelu', 'layer_norm', 'softmax']
cv_sim_sfu_types=['global_avg_pool', 'hard_sigmoid', 'hard_swish', 'relu']
unhandled_trace_sfu_types=['gelu', 'layer_norm', 'softmax']
tokens=10 decode_ms=50.761 report_pipeline_fps=10.989 serial_fps=7.054
tokens=20 decode_ms=101.523 report_pipeline_fps=9.850 serial_fps=5.194
params=0.45B weights_int4=0.225GB weights_int8=0.450GB weights_bf16=0.900GB
params=3.3B weights_int4=1.650GB weights_int8=3.300GB weights_bf16=6.600GB
params=7.0B weights_int4=3.500GB weights_int8=7.000GB weights_bf16=14.000GB
params=13.0B weights_int4=6.500GB weights_int8=13.000GB weights_bf16=26.000GB
```

Verdict:

- CONFIRMED: the report's 2.583 TMAC trace total is reproducible.
- REFUTED as a complete ViT workload: the trace omits at least 0.344 TMAC of attention matrix multiplications, an 11.76% minimum undercount.
- REFUTED as full SFU timing: `gelu`, `layer_norm`, and `softmax` are not dispatched as SFU work by `cv_sim.py`.
- PARTIAL: 10.99 pipeline FPS is arithmetically correct only under the report's overlap assumption; a single shared engine with serial stages gives 7.05 FPS.
- REFUTED: the 20-token case does not meet 10 FPS at 197 TPS.
