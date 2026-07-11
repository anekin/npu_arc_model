#!/usr/bin/env python3
"""参数扫描 — 遍历 NPU 配置空间，找到最优参数组合

扫描维度:
- MXU 尺寸: 64×64, 128×128, 256×128
- 频率: 800, 1000, 1200 MHz
- L2 SRAM: 1, 2, 4, 8 MB
- 核心数: 1, 2, 4

输出: 延迟→面积→功耗 Pareto 前沿
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from models.mxu import MXUModel
from models.sfu import SFUModel
from models.dma import DMAModel
from models.kv_cache import KVCacheModel
from models.dram import DRAMModel
from engine.timeline import CoreTimeline, SimulationReport, breakdown_events


class ParamSweeper:
    """Sweep NPU configuration parameters and collect results."""

    def __init__(self, base_config_path: str):
        with open(base_config_path) as f:
            self.base_config = yaml.safe_load(f)

    def sweep(self) -> List[Dict]:
        """Run full parameter sweep."""
        results = []

        # Define search space
        mxu_sizes = [(64, 64), (128, 128), (256, 128)]  # H×W
        frequencies = [800, 1000, 1200]  # MHz
        l2_sizes = [1024, 2048, 4096, 8192]  # KB
        num_cores = [1, 2, 4]

        # Representative GEMM sizes from Qwen2.5-3B decode
        gemms = [
            (1, 2560, 4096, "Q_proj"),
            (1, 2560, 9728, "FFN_up"),
            (1, 9728, 2560, "FFN_down"),
        ]

        total_configs = len(mxu_sizes) * len(frequencies) * len(l2_sizes)
        count = 0

        for H, W in mxu_sizes:
            for freq in frequencies:
                for l2_kb in l2_sizes:
                    count += 1
                    config = self._build_config(H, W, freq, l2_kb)
                    result = self._evaluate(config, gemms)
                    result["config_id"] = count
                    result["total_configs"] = total_configs
                    results.append(result)
                    print(f"[{count}/{total_configs}] {H}×{W} @{freq}MHz "
                          f"L2={l2_kb//1024}MB → {result['decode_tok_per_s']:.0f} tok/s")

        # Add multi-core projections
        self._add_multicore(results, num_cores)

        return results

    def _build_config(self, H: int, W: int, freq: int, l2_kb: int) -> Dict:
        config = dict(self.base_config)  # shallow copy
        config["mxu"] = dict(config["mxu"])
        config["mxu"]["array_height"] = H
        config["mxu"]["array_width"] = W
        config["mxu"]["frequency_mhz"] = freq
        config["sram"] = dict(config["sram"])
        config["sram"]["l2_shared_kb"] = l2_kb
        return config

    def _evaluate(self, config: Dict, gemms: List[Tuple]) -> Dict:
        """Run decode simulation for this config."""
        mxu = MXUModel(config)
        sfu = SFUModel(config)
        dma = DMAModel(config)
        kv = KVCacheModel(config)
        dram = DRAMModel(config)
        kv.configure_for_model(num_kv_heads=2, head_dim=128, num_layers=28)

        timeline = CoreTimeline()
        total_mxu = 0

        # Simulate 28 layers × 7 matmuls
        for layer in range(28):
            for (M, K, N, op_name) in gemms:
                r = mxu.estimate(M, K, N)  # v2: weight_preloaded removed, default False
                total_mxu += r.total_cycles
            # SFU per layer
            total_mxu += sfu.estimate("softmax", 2560)
            total_mxu += sfu.estimate("layernorm", 2560)
            total_mxu += sfu.estimate("gelu", 9728)
            total_mxu += kv.estimate_per_decode(128, 128)

        # Extend to full 7-ops per layer
        total_mxu = int(total_mxu * (7 / 3))  # 3 gemms sampled, 7 actual
        total_mxu += dram.add_refresh_overhead(total_mxu)

        freq = config["mxu"]["frequency_mhz"]
        decode_us = total_mxu / freq
        decode_tok_per_s = 1e6 / decode_us if decode_us > 0 else 0

        # Area estimation (very rough, from design doc)
        H, W = config["mxu"]["array_height"], config["mxu"]["array_width"]
        area_mxu = (H * W) / (128 * 128) * 8  # 8mm² baseline at 128×128
        area_sram = config["sram"]["l2_shared_kb"] / 1024 * 2  # ~2mm² per MB
        area_total = area_mxu + area_sram + 6  # 6mm² fixed (RISC-V, PCIe, etc.)

        # Power estimation (proportional to freq × area × activity)
        power_w = freq / 1000 * area_total * 0.15  # 0.15 W/mm² at 1GHz, very rough

        return {
            "mxu_size": f"{H}×{W}",
            "freq_mhz": freq,
            "l2_mb": config["sram"]["l2_shared_kb"] // 1024,
            "area_mm2": round(area_total, 1),
            "power_w": round(power_w, 1),
            "decode_tok_per_s": round(decode_tok_per_s, 0),
            "decode_us": round(decode_us, 1),
            "meets_target": decode_tok_per_s >= 21,  # matches overnight_loop.py TARGET_TOK_S=21
        }

    def _add_multicore(self, results: List[Dict], num_cores: List[int]):
        """Add multi-core projections to each result."""
        for r in results:
            r["multicore"] = {}
            base_tok = r["decode_tok_per_s"]
            for nc in num_cores:
                contention = max(0.5, 1.0 - (nc - 1) * 0.05)
                r["multicore"][f"{nc}c"] = round(base_tok * nc * contention, 0)


def main():
    sweeper = ParamSweeper(str(Path(__file__).parent / "config" / "npu_config.yaml"))
    results = sweeper.sweep()

    # Save results
    out_path = Path(__file__).parent / "results" / "param_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Pareto summary
    print(f"\n{'='*70}")
    print(f"  PARETO FRONTIER: meets 21 tok/s target, sorted by area")
    print(f"{'='*70}")
    print(f"{'MXU':>8s} {'MHz':>5s} {'L2':>4s} {'Area':>6s} {'Power':>6s} {'Decode':>8s} {'2c':>8s} {'4c':>8s}")
    print(f"{'─'*60}")

    pareto = [r for r in results if r["meets_target"]]
    pareto.sort(key=lambda r: r["area_mm2"])
    for r in pareto:
        print(f"{r['mxu_size']:>8s} {r['freq_mhz']:>5d} {r['l2_mb']:>3d}MB "
              f"{r['area_mm2']:>5.1f}mm² {r['power_w']:>5.1f}W "
              f"{r['decode_tok_per_s']:>6.0f}t/s "
              f"{r['multicore']['2c']:>6.0f} "
              f"{r['multicore']['4c']:>6.0f}")

    if pareto:
        best = pareto[0]
        print(f"\n  RECOMMENDED: {best['mxu_size']} @{best['freq_mhz']}MHz "
              f"L2={best['l2_mb']}MB → {best['decode_tok_per_s']:.0f} tok/s, "
              f"{best['area_mm2']}mm²")
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
