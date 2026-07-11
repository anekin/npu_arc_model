#!/usr/bin/env python3
"""NPU 参数扫描 v2 — 探索设计空间达到 24 tok/s (DRAM BW bounded)"""

import sys, math, json, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from npu_sim import NPUSimulator, generate_qwen3b_trace

def load_config():
    import yaml
    with open("config/npu_config.yaml") as f:
        return yaml.safe_load(f)

def run_sim(config):
    """Run decode sim with given config, return tok/s"""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        tmp = f.name
    try:
        sim = NPUSimulator(tmp)
        trace = generate_qwen3b_trace(prompt_len=1)
        report = sim.simulate_decode(trace)
        return report.decode_tok_per_s, report.decode_per_token_us
    finally:
        os.unlink(tmp)

def main():
    base = load_config()
    results = []

    H = base["mxu"]["array_height"]
    W = base["mxu"]["array_width"]

    # --- Baseline ---
    tok, us = run_sim(base)
    print(f"Baseline ({H}×{W}): {tok:.0f} tok/s ({us:.0f} μs)")
    results.append({"config": f"{H}×{W}, M=1", "tok_s": tok, "us": us, "area_mm2": 27})

    # --- Option 1: Wider array ---
    for h, w, area in [(128, 256, 42), (256, 256, 108), (64, 256, 32), (256, 128, 54)]:
        c = copy.deepcopy(base)
        c["mxu"]["array_height"] = h
        c["mxu"]["array_width"] = w
        tok, us = run_sim(c)
        label = f"{h}×{w}, M=1"
        print(f"  {label}: {tok:.0f} tok/s, {area}mm²")
        results.append({"config": label, "tok_s": tok, "us": us, "area_mm2": area})

    # --- Option 2: Batch M > 1 (模拟连续批处理) ---
    for M in [2, 4, 8]:
        c = copy.deepcopy(base)
        # 模拟 M 个 token 批处理: 每个 GEMM 的 M 乘 M
        sim = NPUSimulator.__new__(NPUSimulator)
        # 直接调 estimate
        from models.mxu import MXUModel
        mxu = MXUModel(c)

        # 计算 28 层 × 7 matmuls 的总 cycles
        trace = generate_qwen3b_trace(prompt_len=1)
        total_cycles = 0
        for (_, K, N, _, _) in trace:
            r = mxu.estimate(M, K, N)
            total_cycles += r.total_cycles
        us = total_cycles / 1000  # 1GHz
        tok_s = M * 1e6 / us if us > 0 else 0
        label = f"{base['mxu']['array_height']}×{base['mxu']['array_width']}, M={M} (batch)"
        print(f"  {label}: {tok_s:.0f} tok/s ({us:.0f} μs for {M} tokens)")
        results.append({"config": label, "tok_s": tok_s, "us": us, "area_mm2": 27})

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'Config':<25} {'tok/s':>8} {'Area':>8} {'Cost/tok':>10}")
    print(f"{'-'*60}")
    target = 21  # M=1 decode DRAM BW bounded target (matches overnight_loop.py TARGET_TOK_S=21; .0f rounds to 22)
    for r in sorted(results, key=lambda x: x["tok_s"], reverse=True):
        cost_per_tok = r["area_mm2"] / r["tok_s"] if r["tok_s"] > 0 else 999
        flag = "✅" if r["tok_s"] >= target else "  "
        print(f"{flag} {r['config']:<23} {r['tok_s']:>7.0f} {r['area_mm2']:>6d}mm² {cost_per_tok:>8.1f}")

    with open("results/param_sweep_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to results/param_sweep_v2.json")


if __name__ == "__main__":
    main()
