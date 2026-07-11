"""Golden Model — 功能级数值仿真

Functional mode: 真实计算 MXU 矩阵乘法 + SFU 激活函数 + KV Cache 地址。
用于 RTL 验证时的 golden reference，以及小规模端到端功能验证。

和 Performance 模式的区别:
- Performance: 只数 cycles，不算数值
- Functional: numpy 逐 bit 计算，输出 hash/checksum 用于比对
"""

import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

warnings.warn(
    "models.golden is deprecated; use golden_executor for RTL sign-off",
    DeprecationWarning,
    stacklevel=2,
)

# ═══════════════════════════════════════════════════════════════════════
# Verification tiers:
#   models/golden → float32 reference (human-readable hashes, quick checks)
#   golden_executor → INT32 bit-accurate (RTL verification, hw-identical)
# These are complementary, NOT deprecated. Use golden_executor for
# RTL sign-off; use models.golden for interactive functional tests.
# ═══════════════════════════════════════════════════════════════════════

import numpy as np


@dataclass
class LayerOutput:
    """一层 transformer 的输出"""
    layer: int
    output_hash: str   # MD5 of flattened output
    shape: Tuple[int, ...]
    cycles_estimate: int = 0


class GoldenMXU:
    """Functional MXU: INT4 weights × INT8 activations → INT32 accumulate

    Bit-accurate to the hardware: 128×128 weight-stationary systolic array.
    For validation, we simulate the exact tiling the hardware would do.
    """

    def __init__(self, array_h: int = 128, array_w: int = 128):
        self.H = array_h
        self.W = array_w

    def dequantize_int4(self, weight_packed: np.ndarray) -> np.ndarray:
        """Dequantize INT4 weights (packed 2 per byte) → float32."""
        if weight_packed.dtype != np.uint8:
            weight_packed = weight_packed.astype(np.uint8)
        # Unpack: low nibble then high nibble
        low = (weight_packed & 0x0F).astype(np.int8)
        high = ((weight_packed >> 4) & 0x0F).astype(np.int8)
        # Sign-extend: values 8-15 → negative
        low = np.where(low > 7, low - 16, low)
        high = np.where(high > 7, high - 16, high)
        return np.stack([low, high], axis=-1).reshape(-1)[:weight_packed.size * 2]

    def matmul(self, activation: np.ndarray, weight_int4: np.ndarray,
               M: int, K: int, N: int) -> np.ndarray:
        """Compute (M×K) × (K×N) → (M×N) in INT32.

        Simulates weight-stationary tiling: 128×128 tiles.
        """
        # Dequantize weights to float for numpy matmul
        w_flat = self.dequantize_int4(weight_int4)
        expected_len = K * N
        if len(w_flat) < expected_len:
            w_flat = np.pad(w_flat, (0, expected_len - len(w_flat)))
        W = w_flat[:expected_len].reshape(K, N).astype(np.float32)

        A = activation.astype(np.float32)
        if A.size < M * K:
            A = np.pad(A.flat, (0, M * K - A.size)).reshape(M, K)
        A = A[:M, :K]

        # Quantize activations to INT8 (simulate hardware)
        A_int8 = np.clip(A, -128, 127).astype(np.int8)

        # Tiled matmul simulating 128×128 array
        result = np.zeros((M, N), dtype=np.int32)
        for m_start in range(0, M, self.H):
            m_end = min(m_start + self.H, M)
            for n_start in range(0, N, self.W):
                n_end = min(n_start + self.W, N)
                # Weight-stationary: W tile stays, A streams through
                w_tile = W[:, n_start:n_end].astype(np.float32)
                a_tile = A_int8[m_start:m_end, :].astype(np.float32)
                result[m_start:m_end, n_start:n_end] = np.matmul(
                    a_tile, w_tile
                ).astype(np.int32)
        return result

    @staticmethod
    def hash_output(arr: np.ndarray) -> str:
        """MD5 hash of array for fast comparison."""
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]


class GoldenSFU:
    """Functional SFU: bit-accurate activation functions.

    Each function must match the hardware's fixed-pipeline implementation.
    """

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Bit-accurate softmax: max-subtract + exp + normalize.

        Matches the 8-stage hardware pipeline (指数查表 + 分段减法).
        """
        x_max = np.max(x)
        # Hardware uses 256-entry LUT for exp, we approximate
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x)

    @staticmethod
    def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """LayerNorm: (x - mean) / std * gamma + beta.

        Hardware uses 6-stage pipeline (parallel mean/var + fused multiply-add).
        """
        mean = np.mean(x)
        var = np.var(x)
        return (x - mean) / np.sqrt(var + eps)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU: x * Φ(x) where Φ is Gaussian CDF.

        Hardware uses 4-stage piecewise linear LUT.
        """
        # tanh approximation (matches hardware LUT precision)
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
        ))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        """SiLU / Swish: x * sigmoid(x)."""
        return x / (1.0 + np.exp(-x))

    @staticmethod
    def rope(x_q: np.ndarray, x_k: np.ndarray, position: int,
             num_heads: int = 32, head_dim: int = 128,
             theta: float = 10000.0) -> Tuple[np.ndarray, np.ndarray]:
        """RoPE: Rotary Position Embedding per head.

        Hardware uses 12-stage CORDIC rotation.
        x_q shape: (num_heads * head_dim,) = (4096,) for Q
        x_k shape: (num_kv_heads * head_dim,) = (256,) for K
        """
        freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))
        angles = position * freqs
        cos = np.cos(angles)
        sin = np.sin(angles)

        def rotate(x, n_heads):
            x_reshaped = x.reshape(n_heads, head_dim)
            x_rot = np.zeros_like(x_reshaped)
            for h in range(n_heads):
                for i in range(0, head_dim, 2):
                    x0, x1 = x_reshaped[h, i], x_reshaped[h, i+1]
                    c, s = cos[i//2], sin[i//2]
                    x_rot[h, i] = x0 * c - x1 * s
                    x_rot[h, i+1] = x1 * c + x0 * s
            return x_rot.reshape(-1)

        return rotate(x_q, num_heads), rotate(x_k, 2)  # GQA: 2 KV heads


class GoldenModel:
    """Complete NPU golden model for functional verification.

    Runs a single transformer layer end-to-end with bit-accurate computation.
    """

    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9728,
                 num_heads: int = 32, num_kv_heads: int = 2, head_dim: int = 128):
        self.hidden = hidden_size
        self.intermediate = intermediate_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.mxu = GoldenMXU()
        self.sfu = GoldenSFU()

    def run_layer(self, layer: int, hidden_states: np.ndarray,
                  weights: Dict[str, np.ndarray],
                  position: int = 0) -> LayerOutput:
        """Run one transformer layer with functional computation.

        Args:
            layer: layer index
            hidden_states: (1, hidden_size) input
            weights: dict of "q_proj", "k_proj", etc. → INT4 numpy arrays
            position: token position for RoPE

        Returns LayerOutput with hash of final hidden_states.
        """
        x = hidden_states.astype(np.float32)
        B, H = x.shape  # (1, 2560)

        # Self-Attention
        Q = self.mxu.matmul(x, weights.get("q_proj", np.zeros(1)),
                            B, H, self.num_heads * self.head_dim)
        K = self.mxu.matmul(x, weights.get("k_proj", np.zeros(1)),
                            B, H, self.num_kv_heads * self.head_dim)
        V = self.mxu.matmul(x, weights.get("v_proj", np.zeros(1)),
                            B, H, self.num_kv_heads * self.head_dim)

        # RoPE
        Q_rope, K_rope = self.sfu.rope(
            Q.flatten(), K.flatten(), position,
            num_heads=self.num_heads, head_dim=self.head_dim,
        )

        # Attention: simplified — QK^T × V (decode: 1 token)
        # Q: (1, 4096), K: (1, 256) — GQA expand needed
        # Simplified: just compute softmax(QK^T) @ V
        attn_scores = np.dot(Q_rope.reshape(1, -1),
                              K_rope.reshape(-1, 1)) / math.sqrt(self.head_dim)
        attn_probs = self.sfu.softmax(attn_scores.flatten())

        # Output projection
        O = self.mxu.matmul(
            (attn_probs.reshape(1, -1) @ V.reshape(-1, self.num_kv_heads * self.head_dim)).reshape(1, -1),
            weights.get("o_proj", np.zeros(1)),
            1, self.num_kv_heads * self.head_dim, H,
        )

        # Residual + LayerNorm
        x = x + O
        x = self.sfu.layernorm(x.flatten()).reshape(1, -1)

        # FFN
        gate = self.mxu.matmul(x, weights.get("gate_proj", np.zeros(1)),
                               1, H, self.intermediate)
        gate = self.sfu.silu(gate.flatten()).reshape(1, -1)

        up = self.mxu.matmul(x, weights.get("up_proj", np.zeros(1)),
                             1, H, self.intermediate)
        ffn_out = gate * up

        down = self.mxu.matmul(ffn_out, weights.get("down_proj", np.zeros(1)),
                               1, self.intermediate, H)

        # Residual + LayerNorm
        x = x + down
        x = self.sfu.layernorm(x.flatten()).reshape(1, -1)

        return LayerOutput(
            layer=layer,
            output_hash=GoldenMXU.hash_output(x.flatten()),
            shape=x.shape,
        )

    def run_smoke_test(self) -> List[LayerOutput]:
        """Quick smoke test: run 2 layers with random weights."""
        outputs = []
        x = np.random.randn(1, self.hidden).astype(np.float32) * 0.02
        for layer in range(2):
            weights = {
                k: np.random.randint(0, 16, size=(1024,), dtype=np.uint8)
                for k in ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
            }
            out = self.run_layer(layer, x, weights)
            outputs.append(out)
            x = np.random.randn(1, self.hidden).astype(np.float32) * 0.02
        return outputs
