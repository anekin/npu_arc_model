#!/usr/bin/env python3
"""Arc Model — INT4 quantization validation (simplified, fast)."""
import sys, time, numpy as np
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                                    # sim/
sys.path.insert(0, str(_HERE.parent / "ggml-npu"))                # ggml-npu/

from q4_dequant import load_weights_from_gguf
from golden_executor import GoldenMXU
from quantize import quantize_int4_per_channel, quantize_int4_per_block

def validate_model(gguf_path):
    print(f"Model: {gguf_path}")
    t0 = time.time()
    weights = load_weights_from_gguf(gguf_path)
    print(f"Loaded {len(weights)} tensors in {time.time()-t0:.1f}s\n")

    mxu = GoldenMXU()
    rng = np.random.RandomState(42)
    pc_cos, pb_cos = [], []

    for name, W_f32 in sorted(weights.items()):
        if W_f32.ndim != 2 or "weight" not in name.lower():
            continue
        K, N = W_f32.shape
        if K < 64 or N < 64:
            continue
        act = rng.randint(-128, 128, size=K, dtype=np.int8).reshape(1, K)
        golden = act.astype(np.float32) @ W_f32.astype(np.float32)
        g_vec = golden[0, :].astype(np.float64)
        ng = np.linalg.norm(g_vec)

        # Per-channel
        p, sc, _ = quantize_int4_per_channel(W_f32)
        r = mxu.matmul_int4_per_channel(act, p, sc, 1, K, N)
        t = r[0, :].astype(np.float64)
        nt = np.linalg.norm(t)
        pc_cos.append(float(np.dot(g_vec, t) / max(ng * nt, 1e-16)))

        # Per-block
        p, sc, _ = quantize_int4_per_block(W_f32, 128)
        r = mxu.matmul_int4_per_block(act, p, sc, 1, K, N, group_size=128)
        t = r[0, :].astype(np.float64)
        nt = np.linalg.norm(t)
        pb_cos.append(float(np.dot(g_vec, t) / max(ng * nt, 1e-16)))

    n = len(pc_cos)
    print(f"Tested {n} weight matrices\n")
    print(f"Per-channel:  mean={np.mean(pc_cos):.6f}  min={np.min(pc_cos):.6f}")
    print(f"Per-block:    mean={np.mean(pb_cos):.6f}  min={np.min(pb_cos):.6f}\n")

    if np.mean(pb_cos) > np.mean(pc_cos) + 0.01:
        print("→ Recommend: per-block (g=128)")
        print(f"  +{np.mean(pb_cos)-np.mean(pc_cos):.4f} cos_sim improvement over per-channel")
    else:
        print("→ Recommend: per-channel")
        print("  Comparable accuracy, simpler hardware (+3% area)")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="")
    a = p.parse_args()
    path = a.model or str(__import__("pathlib").Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
    validate_model(path)
