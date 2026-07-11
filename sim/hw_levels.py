#!/usr/bin/env python3
"""硬件优化三层对比 — 一键跑 L0/L1/L2 配置，输出性能对比表"""

import sys, subprocess
from pathlib import Path

SIM_DIR = Path(__file__).parent
CONFIGS = [
    ("L0: Baseline", "config/npu_config.yaml",
     "128×128, 64-bit LPDDR5, INT4"),
    ("L1: +Weight Cache", "config/npu_config_l1_cache.yaml",
     "PE双weight寄存器, gate+up合并"),
    ("L2: +DMA×2", "config/npu_config_l2_dma.yaml",
     "128-bit DRAM 或 4ch DMA"),
]


def run_config(name, config_path, desc):
    """Run npu_sim.py with given config and parse tok/s."""
    import json, re
    result = subprocess.run(
        [sys.executable, str(SIM_DIR / "npu_sim.py"),
         "-c", config_path, "--json"],
        capture_output=True, text=True,
        timeout=120,
        cwd=str(SIM_DIR),
    )

    tok_s = 0
    per_token_us = 0

    # Parse JSON block from output
    json_match = re.search(r'\{.*"decode".*\}', result.stdout, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(0))
        tok_s = data["decode"]["tok_per_s"]
        per_token_us = data["decode"]["per_token_us"]

    return {
        "name": name,
        "desc": desc,
        "tok_s": round(tok_s, 1),
        "per_token_us": round(per_token_us, 1),
    }


def main():
    print("=" * 70)
    print("  CaduceusCore 硬件优化三层对比")
    print("  Qwen2.5-3B Decode (M=1), 128×128, 1GHz")
    print("=" * 70)

    results = []
    for name, cfg, desc in CONFIGS:
        print(f"\n  Running: {name}...", end=" ", flush=True)
        try:
            r = run_config(name, cfg, desc)
            results.append(r)
            print(f"{r['tok_s']:.0f} tok/s")
        except Exception as e:
            print(f"ERROR: {e}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  性能对比")
    print(f"  {'Level':<25} {'tok/s':>8} {'μs/token':>12} {'改善':>8}")
    print(f"  {'-'*60}")

    base_tok = results[0]["tok_s"] if results else 15
    for r in results:
        delta = r["tok_s"] - base_tok
        print(f"  {r['name']:<25} {r['tok_s']:>7.1f} {r['per_token_us']:>10.0f}  {delta:>+6.1f}")

    # Hardware cost
    print(f"\n  硬件代价")
    cost_table = [
        ("L0: Baseline",           "27mm²", "—"),
        ("L1: +Weight Cache",      "28mm²", "PE +15%, +1mm²"),
        ("L2: +DMA×2",            "28mm²", "128-bit PHY 或 DMA引擎×2"),
    ]
    for name, area, note in cost_table:
        print(f"  {name:<25} {area:>8}  {note}")

    # Combined with batching
    print(f"\n  叠加 M=2 Batching")
    for r in results:
        # Rough estimate: 2× with ~5% contention penalty
        batch_tok = r["tok_s"] * 2 * 0.95
        print(f"  {r['name']:<25} ~{batch_tok:.0f} tok/s")

    print(f"\n  结论: L0→L2 纯硬件路径: {base_tok:.0f} → {results[-1]['tok_s']:.0f} tok/s")


if __name__ == "__main__":
    main()
