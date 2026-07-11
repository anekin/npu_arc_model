#!/usr/bin/env python3
"""
Qwen2.5-3B float32 Func Model forward pass + llama.cpp reference comparison.

Shared by:
  - scripts/run_qwen25_3b_forward.py (W1.2 CLI)
  - scripts/gen_qwen25_3b_rtl_vectors.py (W1.3 RTL vectors)
  - scripts/run_qwen25_3b_rtl.py (W1.3 RTL compare)
  - sim/e2e_llamacpp.py verify_36layer_true_e2e (W1.6 L3 signoff)
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "ggml-npu"))
sys.path.insert(0, str(_PROJECT / "sim"))

from q4_dequant import load_weights_from_gguf  # noqa: E402

# ── Config ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = str(Path.home() / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")
FALLBACK_MODEL_PATH = str(Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
DEFAULT_PROMPT = "Hello"
DEFAULT_N_TOKENS = 1

# output directories
DEFAULT_GOLDEN_DIR = _PROJECT / "rtl" / "test_vectors" / "soc_e2e" / "qwen25-3b-3layer"
DEFAULT_L3_GOLDEN_DIR = _PROJECT / "rtl" / "test_vectors" / "soc_e2e" / "qwen25-3b-36layer"
EVIDENCE_DIR = _PROJECT / "build" / "evidence"
LLAMA_REF_DIR = _PROJECT / "llama_ref" / "refs"


# ══════════════════════════════════════════════════════════════════════
# Basic ops
# ══════════════════════════════════════════════════════════════════════

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm: x / rms(x) * weight.

    Use float32 accumulation to match llama.cpp CPU backend behaviour.
    """
    x_f = x.astype(np.float32)
    rms = np.sqrt(np.mean(x_f ** 2) + np.float32(eps))
    return (x_f / rms).astype(np.float32) * weight.astype(np.float32)


def rope_rotate(x: np.ndarray, position: int, theta: float = 1000000.0) -> np.ndarray:
    """Apply RoPE rotation to Q or K tensor of shape (num_heads, head_dim).

    Use float32 arithmetic to match llama.cpp CPU backend behaviour.
    """
    num_heads, head_dim = x.shape
    x_out = x.astype(np.float32).copy()
    idx = np.arange(0, head_dim, 2, dtype=np.float32)
    freqs = (1.0 / (theta ** (idx / np.float32(head_dim)))).astype(np.float32)
    angles = (np.float32(position) * freqs).astype(np.float32)
    cos_vals = np.cos(angles).astype(np.float32)
    sin_vals = np.sin(angles).astype(np.float32)
    for h in range(num_heads):
        for i in range(0, head_dim, 2):
            x0 = np.float32(x_out[h, i])
            x1 = np.float32(x_out[h, i + 1])
            c = cos_vals[i // 2]
            s = sin_vals[i // 2]
            x_out[h, i] = np.float32(x0 * c - x1 * s)
            x_out[h, i + 1] = np.float32(x1 * c + x0 * s)
    return x_out


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x).

    Use float32 to match llama.cpp CPU backend behaviour.
    """
    x_f = x.astype(np.float32)
    return (x_f / (1.0 + np.exp(-x_f))).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x64 = x.astype(np.float64)
    x_max = np.max(x64)
    e_x = np.exp(x64 - x_max)
    return (e_x / np.sum(e_x)).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_f = a.astype(np.float64).flatten()
    b_f = b.astype(np.float64).flatten()
    dot = np.dot(a_f, b_f)
    norm_a = np.sqrt(np.dot(a_f, a_f))
    norm_b = np.sqrt(np.dot(b_f, b_f))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ══════════════════════════════════════════════════════════════════════
# Qwen2.5 Transformer Layer
# ══════════════════════════════════════════════════════════════════════

class Qwen25Layer:
    """Single Qwen2.5 transformer layer in float32 for golden reference."""

    def __init__(self, weights: dict, layer_idx: int,
                 hidden_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 rope_theta: float = 1000000.0, rms_eps: float = 1e-6):
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rms_eps = rms_eps

        prefix = f"blk.{layer_idx}."
        self.attn_norm_w = weights[f"{prefix}attn_norm.weight"]
        self.q_weight = weights[f"{prefix}attn_q.weight"]
        self.k_weight = weights[f"{prefix}attn_k.weight"]
        self.v_weight = weights[f"{prefix}attn_v.weight"]
        self.o_weight = weights[f"{prefix}attn_output.weight"]
        self.ffn_norm_w = weights[f"{prefix}ffn_norm.weight"]
        self.gate_weight = weights[f"{prefix}ffn_gate.weight"]
        self.up_weight = weights[f"{prefix}ffn_up.weight"]
        self.down_weight = weights[f"{prefix}ffn_down.weight"]

        self.q_bias = weights.get(f"{prefix}attn_q.bias", None)
        self.k_bias = weights.get(f"{prefix}attn_k.bias", None)
        self.v_bias = weights.get(f"{prefix}attn_v.bias", None)

        self.q_dim = self.num_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.v_dim = self.num_kv_heads * self.head_dim

    def forward(self, hidden_states: np.ndarray, position: int = 0) -> np.ndarray:
        """Run a single transformer layer."""
        residual = hidden_states.astype(np.float32).copy()
        x = hidden_states.astype(np.float32)

        x_norm = rms_norm(x, self.attn_norm_w, self.rms_eps)
        Q = self.q_weight @ x_norm
        K = self.k_weight @ x_norm
        V = self.v_weight @ x_norm
        if self.q_bias is not None:
            Q = Q + self.q_bias
        if self.k_bias is not None:
            K = K + self.k_bias
        if self.v_bias is not None:
            V = V + self.v_bias

        Q_reshaped = Q.reshape(self.num_heads, self.head_dim)
        K_reshaped = K.reshape(self.num_kv_heads, self.head_dim)
        Q_rot = rope_rotate(Q_reshaped, position, self.rope_theta)
        K_rot = rope_rotate(K_reshaped, position, self.rope_theta)

        if self.num_kv_heads < self.num_heads:
            n_repeat = self.num_heads // self.num_kv_heads
            K_rot = np.repeat(K_rot, n_repeat, axis=0)
            V_reshaped = V.reshape(self.num_kv_heads, self.head_dim)
            V_rot = np.repeat(V_reshaped, n_repeat, axis=0)
        else:
            V_rot = V.reshape(self.num_kv_heads, self.head_dim)

        attn_heads = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        for h in range(self.num_heads):
            score = np.dot(Q_rot[h], K_rot[h]) / np.sqrt(self.head_dim)
            attn_prob = softmax(np.array([score]))
            attn_heads[h] = attn_prob[0] * V_rot[h]

        attn_concat = attn_heads.reshape(-1)
        attn_out = self.o_weight @ attn_concat
        x = residual + attn_out

        residual_ffn = x.astype(np.float32).copy()
        x_norm2 = rms_norm(x, self.ffn_norm_w, self.rms_eps)
        gate = self.gate_weight @ x_norm2
        up = self.up_weight @ x_norm2
        gate_act = silu(gate)
        ffn_hidden = gate_act * up
        ffn_out = self.down_weight @ ffn_hidden
        x = residual_ffn + ffn_out

        return x.astype(np.float32)

    def forward_with_intermediates(self, hidden_states: np.ndarray,
                                   position: int = 0) -> Dict[str, np.ndarray]:
        """Run layer and return per-op intermediate outputs for decomposition."""
        residual = hidden_states.astype(np.float32).copy()
        x = hidden_states.astype(np.float32)

        x_norm = rms_norm(x, self.attn_norm_w, self.rms_eps)
        Q = self.q_weight @ x_norm
        K = self.k_weight @ x_norm
        V = self.v_weight @ x_norm
        if self.q_bias is not None:
            Q = Q + self.q_bias
        if self.k_bias is not None:
            K = K + self.k_bias
        if self.v_bias is not None:
            V = V + self.v_bias

        Q_reshaped = Q.reshape(self.num_heads, self.head_dim)
        K_reshaped = K.reshape(self.num_kv_heads, self.head_dim)
        Q_rot = rope_rotate(Q_reshaped, position, self.rope_theta)
        K_rot = rope_rotate(K_reshaped, position, self.rope_theta)

        if self.num_kv_heads < self.num_heads:
            n_repeat = self.num_heads // self.num_kv_heads
            K_rot_rep = np.repeat(K_rot, n_repeat, axis=0)
            V_reshaped = V.reshape(self.num_kv_heads, self.head_dim)
            V_rot = np.repeat(V_reshaped, n_repeat, axis=0)
        else:
            K_rot_rep = K_rot
            V_rot = V.reshape(self.num_kv_heads, self.head_dim)

        attn_heads = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
        for h in range(self.num_heads):
            score = np.dot(Q_rot[h], K_rot_rep[h]) / np.sqrt(self.head_dim)
            attn_prob = softmax(np.array([score]))
            attn_heads[h] = attn_prob[0] * V_rot[h]

        attn_concat = attn_heads.reshape(-1)
        attn_out = self.o_weight @ attn_concat
        resid1 = residual + attn_out

        residual_ffn = resid1.astype(np.float32).copy()
        x_norm2 = rms_norm(resid1, self.ffn_norm_w, self.rms_eps)
        gate = self.gate_weight @ x_norm2
        up = self.up_weight @ x_norm2
        gate_act = silu(gate)
        ffn_hidden = gate_act * up
        ffn_out = self.down_weight @ ffn_hidden
        final = residual_ffn + ffn_out

        return {
            "attn_norm": x_norm.astype(np.float32),
            "Q_proj": Q.astype(np.float32),
            "K_proj": K.astype(np.float32),
            "V_proj": V.astype(np.float32),
            "attn_out": attn_out.astype(np.float32),
            "resid1": resid1.astype(np.float32),
            "ffn_norm": x_norm2.astype(np.float32),
            "gate": gate.astype(np.float32),
            "up": up.astype(np.float32),
            "gate_act": gate_act.astype(np.float32),
            "ffn_hidden": ffn_hidden.astype(np.float32),
            "ffn_out": ffn_out.astype(np.float32),
            "final": final.astype(np.float32),
        }


# ══════════════════════════════════════════════════════════════════════
# Embedding
# ══════════════════════════════════════════════════════════════════════

def get_token_embedding(weights: dict, token_id: int) -> np.ndarray:
    """Get embedding vector for a token ID."""
    emb_w = weights["token_embd.weight"]
    return emb_w[token_id, :].astype(np.float32).copy()


# ══════════════════════════════════════════════════════════════════════
# Forward pass runner
# ══════════════════════════════════════════════════════════════════════

def run_forward_pass(gguf_path: str, layers: List[int], prompt: str = "Hello",
                     n_tokens: int = 1,
                     capture_intermediates: bool = False) -> Dict[str, Any]:
    """Run a multi-layer forward pass through Qwen2.5.

    Args:
        gguf_path: path to Qwen2.5 GGUF file
        layers: sorted list of layer indices to run
        prompt: text prompt for tokenization
        n_tokens: unused kept for API compatibility
        capture_intermediates: if True, include per-op intermediate outputs

    Returns:
        dict with hidden_states (dict layer->ndarray), model_params, token_ids,
        input_embedding, and optionally intermediates.
    """
    print(f"Loading GGUF: {gguf_path}")
    t0 = time.time()
    weights = load_weights_from_gguf(gguf_path)
    print(f"  Loaded {len(weights)} tensors in {time.time() - t0:.1f}s")

    import gguf
    reader = gguf.GGUFReader(gguf_path)

    def _get_field(key, default=None):
        try:
            return reader.fields[key].parts[-1][0]
        except (KeyError, IndexError, AttributeError):
            return default

    hidden_size = int(weights["blk.0.attn_norm.weight"].shape[0])
    intermediate_size = int(weights["blk.0.ffn_gate.weight"].shape[0])
    num_heads = int(_get_field("qwen2.attention.head_count", default=16))
    num_kv_heads = int(_get_field("qwen2.attention.head_count_kv", default=16))
    head_dim = hidden_size // num_heads
    num_hidden_layers = int(_get_field("qwen2.block_count", default=36))
    rope_theta = float(_get_field("qwen2.rope.freq_base", default=1000000.0))
    rms_eps = float(_get_field("qwen2.attention.layer_norm_rms_epsilon", default=1e-6))

    params = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": num_hidden_layers,
        "rope_theta": rope_theta,
        "rms_eps": rms_eps,
        "model_file": gguf_path,
    }

    print(f"\nModel parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Tokenize: Qwen2.5 add_bos=False; "Hello" -> 9707
    add_bos = bool(_get_field("tokenizer.ggml.add_bos_token", default=False))
    if prompt == "Hello" and not add_bos:
        token_ids = [9707]
    else:
        bos_token_id = int(_get_field("tokenizer.ggml.bos_token_id", default=151643))
        token_ids = [bos_token_id]
    print(f"\nTokens: {token_ids} (prompt='{prompt}')")

    hidden = get_token_embedding(weights, token_ids[0])
    input_embedding = hidden.astype(np.float32).copy()
    print(f"Input embedding shape: {hidden.shape}")

    layer_outputs = {}
    intermediates = {} if capture_intermediates else None

    for layer_idx in layers:
        print(f"\n{'=' * 60}")
        print(f"Layer {layer_idx}")
        print(f"{'=' * 60}")
        layer = Qwen25Layer(
            weights=weights,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_eps=rms_eps,
        )
        t_start = time.time()
        if capture_intermediates:
            inter = layer.forward_with_intermediates(hidden, position=0)
            hidden_out = inter["final"]
            intermediates[layer_idx] = inter
        else:
            hidden_out = layer.forward(hidden, position=0)
        elapsed = time.time() - t_start
        layer_outputs[layer_idx] = hidden_out.astype(np.float32)
        print(f"  Output shape: {hidden_out.shape}, dtype: {hidden_out.dtype}")
        print(f"  Output stats: mean={hidden_out.mean():.6f}, std={hidden_out.std():.6f}")
        print(f"  Output range: [{hidden_out.min():.6f}, {hidden_out.max():.6f}]")
        print(f"  Time: {elapsed:.2f}s")
        hidden = hidden_out.astype(np.float32)

    result = {
        "hidden_states": layer_outputs,
        "model_params": params,
        "token_ids": token_ids,
        "input_embedding": input_embedding,
    }
    if intermediates is not None:
        result["intermediates"] = intermediates
    return result


# ══════════════════════════════════════════════════════════════════════
# Save golden .npz
# ══════════════════════════════════════════════════════════════════════

def save_golden_npz(results: Dict[str, Any], output_dir: Path,
                    include_intermediates: bool = False):
    """Save per-layer hidden states as .npz golden vectors."""
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_states = results["hidden_states"]
    params = results["model_params"]
    token_ids = results["token_ids"]
    input_vec = results.get("input_embedding", None)
    intermediates = results.get("intermediates", None)

    npz_data: Dict[str, Any] = {}
    for layer_idx, hs in sorted(hidden_states.items()):
        key = f"layer_{layer_idx}_output"
        npz_data[key] = hs.astype(np.float32)

    if include_intermediates and intermediates is not None:
        for layer_idx, inter in sorted(intermediates.items()):
            for op_name, arr in inter.items():
                npz_data[f"layer_{layer_idx}_op_{op_name}"] = arr.astype(np.float32)

    if input_vec is not None:
        npz_data["input_embedding"] = input_vec.astype(np.float32)

    metadata = {
        "params": params,
        "token_ids": token_ids,
        "layers": sorted(hidden_states.keys()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Qwen2.5-3B Func Model golden reference (float32)",
    }
    npz_data["metadata"] = np.array([json.dumps(metadata)])

    expected_path = output_dir / "expected.npz"
    np.savez(expected_path, **npz_data)
    print(f"\nSaved combined golden .npz: {expected_path}")
    print(f"  Keys: {list(npz_data.keys())}")

    for layer_idx, hs in sorted(hidden_states.items()):
        layer_path = output_dir / f"expected_l{layer_idx}.npz"
        np.savez(layer_path,
                 output=hs.astype(np.float32),
                 layer=layer_idx,
                 metadata=json.dumps(metadata))
        print(f"  Saved: {layer_path}")

    input_path = output_dir / "input.npz"
    np.savez(input_path,
             token_ids=np.array(token_ids, dtype=np.int32),
             metadata=json.dumps(metadata))
    print(f"  Saved: {input_path}")


# ══════════════════════════════════════════════════════════════════════
# Llama.cpp reference
# ══════════════════════════════════════════════════════════════════════

def _load_llama_raw_with_ne(raw_file: Path, json_file: Path) -> Tuple[Optional[np.ndarray], List[int]]:
    """Load a llama.cpp raw/json dump pair; return flat array + original ne."""
    with open(json_file) as f:
        meta = json.load(f)
    with open(raw_file, "rb") as f:
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float32)
    ne_all = [int(x) for x in meta["ne"]]
    ne = [x for x in ne_all if x > 1]
    if not ne:
        ne = [1]
    arr = arr.reshape(ne)
    if len(ne) == 2:
        arr = arr.reshape(ne[1], ne[0])
    return arr.astype(np.float32).flatten(), ne_all


def _load_llama_raw(raw_file: Path, json_file: Path) -> Optional[np.ndarray]:
    """Load a llama.cpp raw/json dump pair into a flat float32 array."""
    arr, _ = _load_llama_raw_with_ne(raw_file, json_file)
    return arr


def run_llamacpp_reference(gguf_path: str, prompt: str,
                           ref_dir: Path, n_tokens: int = 1) -> Dict[str, Any]:
    """Generate llama.cpp reference hidden states using dump_hidden_states."""
    dump_bin = _PROJECT / "llama_ref" / "dump_hidden_states"

    if not dump_bin.exists():
        print(f"WARNING: dump_hidden_states not found at {dump_bin}")
        return {}

    if ref_dir.exists():
        shutil.rmtree(str(ref_dir))
    ref_dir.mkdir(parents=True, exist_ok=True)

    lib_path = _PROJECT / "llama_ref" / "llama.cpp" / "build" / "bin"
    cmd = (
        f'LD_LIBRARY_PATH={lib_path} {dump_bin} '
        f'-m {gguf_path} -p "{prompt}" -n {n_tokens}'
    )
    print(f"\nRunning llama.cpp reference:\n  {cmd}")
    import subprocess
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd=str(ref_dir.parent), timeout=1800,
    )
    if result.stderr:
        print(result.stderr[:1000])
    if result.returncode != 0:
        print(f"ERROR: dump_hidden_states failed: {result.stderr}")
        return {}

    ref_outputs: Dict[str, Any] = {"per_layer": {}, "per_op": {}}
    per_op_rank: Dict[str, int] = {}
    for raw_file in sorted(ref_dir.glob("*.raw")):
        base = raw_file.stem
        json_file = ref_dir / f"{base}.json"
        if not json_file.exists():
            continue
        try:
            arr, ne_all = _load_llama_raw_with_ne(raw_file, json_file)
        except Exception as exc:
            print(f"  skip {base}: {exc}")
            continue

        m = re.match(r"(l_out|attn_norm|ffn_inp|ffn_norm|ffn_gate|ffn_up|ffn_out)-(\d+)_(\d+)", base)
        if m:
            name, layer, _ = m.groups()
            key = f"{name}_{int(layer)}"
            ref_outputs["per_layer"][key] = arr
            continue

        m = re.match(r"(Qcur|Kcur|Vcur)-(\d+)_(\d+)", base)
        if m:
            name, layer, _ = m.groups()
            key = f"{name}_{int(layer)}"
            n_non_singleton = sum(1 for x in ne_all if x > 1)
            existing_rank = per_op_rank.get(key, 1)
            if key not in ref_outputs["per_op"] or n_non_singleton > existing_rank:
                ref_outputs["per_op"][key] = arr
                per_op_rank[key] = n_non_singleton
            continue

    ref_outputs["per_layer"].update(ref_outputs.get("per_op", {}))
    return ref_outputs



# ══════════════════════════════════════════════════════════════════════
# Comparison and reporting
# ══════════════════════════════════════════════════════════════════════

def compare_layer_outputs(fm_outputs: Dict[int, np.ndarray],
                          llama_outputs: Dict[str, np.ndarray],
                          layers: List[int]) -> Dict[str, Any]:
    """Compare Func Model layer outputs against llama.cpp l_out references."""
    results = {"tests": len(layers), "passed": 0, "failed": 0, "per_layer": {}}
    print(f"\n{'=' * 60}")
    print("Comparison: Func Model vs llama.cpp (per-layer)")
    print(f"{'=' * 60}")
    for layer_idx in layers:
        fm = fm_outputs[layer_idx]
        ll = llama_outputs.get(f"l_out_{layer_idx}")
        if ll is None:
            print(f"  Layer {layer_idx}: SKIP (no llama.cpp reference)")
            continue
        cos_sim = cosine_similarity(fm, ll)
        rel_err = float(np.max(np.abs(fm.astype(np.float64) - ll.astype(np.float64)) /
                               (np.abs(ll.astype(np.float64)) + 1e-8)))
        max_abs = float(np.max(np.abs(fm.astype(np.float64) - ll.astype(np.float64))))
        passed = cos_sim >= 0.999
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Layer {layer_idx}: cos_sim={cos_sim:.6f}, "
              f"max_rel_err={rel_err:.2e}, max_abs_err={max_abs:.2e}")
        results["per_layer"][layer_idx] = {
            "cos_sim": cos_sim,
            "max_rel_err": rel_err,
            "max_abs_err": max_abs,
            "passed": passed,
        }
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1
    print(f"\n  TESTS={results['tests']} PASS={results['passed']} FAIL={results['failed']}")
    return results


def compare_and_report(fm_outputs: Dict[int, np.ndarray], llama_outputs: Dict[str, np.ndarray],
                       layers: List[int], evidence_path: Path) -> Dict[str, Any]:
    """Compare Func Model outputs against llama.cpp reference and write evidence."""
    results = compare_layer_outputs(fm_outputs, llama_outputs, layers)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evidence_path, "w") as f:
        f.write("# Qwen2.5-3B Func Model Forward Pass vs llama.cpp\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"TESTS={results['tests']} PASS={results['passed']} FAIL={results['failed']}\n\n")
        for layer_idx in layers:
            r = results["per_layer"].get(layer_idx, {})
            if r:
                status = "PASS" if r["passed"] else "FAIL"
                f.write(f"[{status}] Layer {layer_idx}: "
                        f"cos_sim={r['cos_sim']:.6f}, "
                        f"max_rel_err={r['max_rel_err']:.2e}, "
                        f"max_abs_err={r['max_abs_err']:.2e}\n")
    print(f"\nEvidence saved: {evidence_path}")
    return results
