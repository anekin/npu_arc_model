#!/usr/bin/env python3
"""
L3 signoff helpers for Qwen2.5-3B 36-layer Func Model.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from qwen25_forward import cosine_similarity


# Mapping from Func Model intermediate name → llama.cpp dump base pattern.
INTERMEDIATE_MAP = {
    "attn_norm": "attn_norm_{layer}",
    "Q_proj": "Qcur_{layer}",
    "K_proj": "Kcur_{layer}",
    "V_proj": "Vcur_{layer}",
    "resid1": "ffn_inp_{layer}",
    "ffn_norm": "ffn_norm_{layer}",
    "gate": "ffn_gate_{layer}",
    "up": "ffn_up_{layer}",
    "ffn_out": "ffn_out_{layer}",
    "final": "l_out_{layer}",
}


def parse_layers_arg(value: Any, num_hidden_layers: int) -> List[int]:
    """Parse --layers argument: ints, '0..35', or --all-layers sentinel."""
    if value is None or value == "all":
        return list(range(num_hidden_layers))

    tokens: List[str] = []
    if isinstance(value, str):
        value = value.strip()
        if ".." in value:
            tokens = [value]
        else:
            tokens = value.split()
    elif isinstance(value, (list, tuple)):
        tokens = [str(x) for x in value]
    else:
        tokens = [str(value)]

    layers: List[int] = []
    for tok in tokens:
        tok = tok.strip()
        if ".." in tok:
            parts = tok.split("..")
            start = int(parts[0].strip()) if parts[0].strip() else 0
            end = int(parts[1].strip()) if parts[1].strip() else num_hidden_layers - 1
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(tok))
    return sorted(set(layers))


def drift_analysis(cos_sim_per_layer: Dict[int, float]) -> Dict[str, Any]:
    """Analyze cos_sim trend across layers for monotonic degradation."""
    layers = sorted(cos_sim_per_layer.keys())
    cos_vals = np.array([cos_sim_per_layer[L] for L in layers], dtype=np.float64)

    # Linear regression slope
    if len(layers) >= 2:
        slope, intercept = np.polyfit(layers, cos_vals, 1)
    else:
        slope, intercept = 0.0, float(cos_vals[0]) if cos_vals.size else 0.0

    # Sequential diffs
    diffs = np.diff(cos_vals)
    max_drop = float(np.min(diffs)) if diffs.size else 0.0
    avg_drop = float(np.mean(diffs[diffs < 0])) if np.any(diffs < 0) else 0.0

    # Pass criteria: slope not negative beyond noise, no large single drop
    slope_pass = slope >= -1e-5
    drop_pass = max_drop >= -1e-4
    overall_pass = slope_pass and drop_pass

    return {
        "layers": layers,
        "cos_sims": cos_vals.tolist(),
        "slope": float(slope),
        "intercept": float(intercept),
        "max_drop": max_drop,
        "avg_drop": avg_drop,
        "slope_pass": slope_pass,
        "drop_pass": drop_pass,
        "drift_pass": overall_pass,
    }


def worst_layer_decomposition(worst_layer: int,
                              fm_intermediates: Dict[str, np.ndarray],
                              llama_outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute per-op cos_sim for the worst layer against llama.cpp dumps."""
    per_op: Dict[str, Dict[str, Any]] = {}
    for op_name, pattern in INTERMEDIATE_MAP.items():
        key = pattern.format(layer=worst_layer)
        fm_arr = fm_intermediates.get(op_name)
        ll_arr = llama_outputs.get(key)
        if fm_arr is None or ll_arr is None:
            continue
        # llama.cpp flattened; ensure same length
        fm_flat = fm_arr.astype(np.float64).flatten()
        ll_flat = ll_arr.astype(np.float64).flatten()
        if fm_flat.size != ll_flat.size:
            continue
        cos_sim = cosine_similarity(fm_flat, ll_flat)
        rel_err = float(np.max(np.abs(fm_flat - ll_flat) / (np.abs(ll_flat) + 1e-8)))
        max_abs = float(np.max(np.abs(fm_flat - ll_flat)))
        per_op[op_name] = {
            "cos_sim": cos_sim,
            "max_rel_err": rel_err,
            "max_abs_err": max_abs,
        }
    return per_op


def write_l3_evidence(evidence_path: Path,
                      layer_results: Dict[str, Any],
                      drift: Dict[str, Any],
                      worst_layer: Optional[int],
                      worst_per_op: Dict[str, Any],
                      checkpoint_layers: List[int]) -> None:
    """Write the L3 signoff evidence file."""
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    per_layer = layer_results["per_layer"]
    layers = sorted(per_layer.keys())

    with open(evidence_path, "w") as f:
        f.write("# W1.6: Qwen2.5-3B 36-Layer Func Model L3 Signoff\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Host: {__import__('socket').gethostname()}\n\n")

        total = layer_results["tests"]
        passed = layer_results["passed"]
        failed = layer_results["failed"]
        f.write(f"SUMMARY: TESTS={total} PASS={passed} FAIL={failed}\n")
        f.write(f"Drift analysis: {'PASS' if drift['drift_pass'] else 'FAIL'}\n")
        f.write(f"Worst layer: {worst_layer if worst_layer is not None else 'N/A'}\n\n")

        f.write("## Checkpoint results (cos_sim >= 0.999 required)\n")
        for L in checkpoint_layers:
            r = per_layer.get(L, {})
            status = "PASS" if r.get("passed") else "FAIL"
            f.write(f"  [{status}] Layer {L}: cos_sim={r.get('cos_sim', 0.0):.6f}\n")
        f.write("\n")

        f.write("## Per-layer cos_sim\n")
        for L in layers:
            r = per_layer.get(L, {})
            f.write(f"  Layer {L}: cos_sim={r.get('cos_sim', 0.0):.6f}, "
                    f"max_rel_err={r.get('max_rel_err', 0.0):.2e}, "
                    f"max_abs_err={r.get('max_abs_err', 0.0):.2e}\n")
        f.write("\n")

        f.write("## Drift analysis\n")
        f.write(f"  linear_slope={drift['slope']:.6e}\n")
        f.write(f"  max_sequential_drop={drift['max_drop']:.6e}\n")
        f.write(f"  avg_negative_drop={drift['avg_drop']:.6e}\n")
        f.write(f"  slope_pass={drift['slope_pass']}\n")
        f.write(f"  drop_pass={drift['drop_pass']}\n")
        f.write(f"  drift_verdict={'PASS' if drift['drift_pass'] else 'FAIL'}\n\n")

        if worst_per_op:
            f.write(f"## Worst-layer per-op decomposition (layer {worst_layer})\n")
            for op_name in sorted(worst_per_op.keys()):
                r = worst_per_op[op_name]
                f.write(f"  {op_name:12s}: cos_sim={r['cos_sim']:.6f}, "
                        f"max_rel_err={r['max_rel_err']:.2e}, "
                        f"max_abs_err={r['max_abs_err']:.2e}\n")
        else:
            f.write("## Worst-layer per-op decomposition: N/A\n")

    print(f"\nL3 evidence saved: {evidence_path}")
