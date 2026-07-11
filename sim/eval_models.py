#!/usr/bin/env python3
"""Simulator eval: 1.5B / 3B / 7B decode tok/s"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from npu_sim import NPUSimulator

models = {
    "1.5B": {"hidden": 1536, "intermediate": 8960, "layers": 28, "heads": 12, "kv_heads": 2, "head_dim": 128},
    "3B":   {"hidden": 2560, "intermediate": 9728, "layers": 28, "heads": 32, "kv_heads": 2, "head_dim": 128},
    "7B":   {"hidden": 3584, "intermediate": 18944, "layers": 28, "heads": 28, "kv_heads": 4, "head_dim": 128},
}

def generate_trace(spec, prompt_len=1):
    H = spec["hidden"]
    I = spec["intermediate"]
    L = spec["layers"]
    QKV = spec["heads"] * spec["head_dim"]
    KV  = spec["kv_heads"] * spec["head_dim"]
    trace = []
    for layer in range(L):
        trace.append((prompt_len, H, QKV, layer, "Q_proj"))
        trace.append((prompt_len, H, KV,  layer, "K_proj"))
        trace.append((prompt_len, H, KV,  layer, "V_proj"))
        trace.append((prompt_len, QKV, H,  layer, "O_proj"))
        trace.append((prompt_len, H, I,   layer, "FFN_gate"))
        trace.append((prompt_len, H, I,   layer, "FFN_up"))
        trace.append((prompt_len, I, H,   layer, "FFN_down"))
    return trace

sim = NPUSimulator("config/npu_config.yaml")
cfg = sim.config["mxu"]
print(f"Config: {cfg['array_height']}x{cfg['array_width']} @ {cfg['frequency_mhz']}MHz")
print()

header = f"{'Model':<6} {'Hidden':>7} {'Inter':>8} {'GEMMs':>6} {'tok/s':>7} {'us/tok':>9} {'GFLOPS':>7} {'Total MAC':>12}"
print(header)
print("-" * len(header))

for name, spec in models.items():
    trace = generate_trace(spec, prompt_len=1)
    report = sim.simulate_decode(trace)
    n_gemm = len(trace)
    total_mac = sum(m * k * n for m, k, n, _, _ in trace) * 2
    gflops = total_mac / (report.decode_per_token_us * 1e-6) / 1e9
    print(f"{name:<6} {spec['hidden']:>7} {spec['intermediate']:>8} {n_gemm:>6} {report.decode_tok_per_s:>7.1f} {report.decode_per_token_us:>9.0f} {gflops:>7.1f} {total_mac/1e9:>9.2f}G")
