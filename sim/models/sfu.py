"""SFU 延迟模型 v2 — 分解式 Special Function Unit

v2 改进：
- softmax 不再作为单操作黑盒，分解为 exp + div 两步
- 增加 scalar 操作（标量广播）
- 区分 SFU（复杂数学）和 Vector（逐元素简单操作）
- 匹配 WeChat 文章分析的四大瓶颈
"""

from typing import Any, Dict, Tuple


class SFUModel:
    """分解式 SFU: exp, log, sqrt, div 等复杂数学函数。

    Pipeline:
    - softmax: max_reduce(Vector) → sub(Vector) → exp(SFU) → sum_reduce(Vector) → div(SFU)
    - layernorm: mean(Vector) → var(Vector) → sub(Vector) → div(SFU) → scale(Vector) → bias(Vector)
    - gelu: tanh_approx(SFU) + mul(Vector)
    """

    def __init__(self, config: Dict[str, Any]):
        sfu = config["sfu"]
        self.width = int(sfu["width"])
        self.freq_mhz = int(config.get("mxu", {}).get("frequency_mhz", 1000))

        # Pipeline depth per operation (cycles to fill + drain pipeline)
        self.latency_map = {
            "exp": int(sfu.get("pipeline_cycles", {}).get("exp", 12)),
            "div": int(sfu.get("pipeline_cycles", {}).get("div", 16)),
            "sqrt": int(sfu.get("pipeline_cycles", {}).get("sqrt", 20)),
            "log": int(sfu.get("pipeline_cycles", {}).get("log", 18)),
            "tanh": int(sfu.get("pipeline_cycles", {}).get("tanh", 14)),
            # Legacy compat (these are now Vector ops)
            "relu": 1,
            "gelu": int(sfu.get("pipeline_cycles", {}).get("gelu", 4)),
            "silu": int(sfu.get("pipeline_cycles", {}).get("silu", 4)),
            "layernorm": int(sfu.get("pipeline_cycles", {}).get("layernorm", 6)),
            "rmsnorm": int(sfu.get("pipeline_cycles", {}).get("rmsnorm", 6)),
            "rope": int(sfu.get("pipeline_cycles", {}).get("rope", 12)),
            # unified softmax (kept for backward compat)
            "softmax": int(sfu.get("pipeline_cycles", {}).get("softmax", 8)),
            # CV-specific ops
            "h_swish": 4,                                          # clip+add+mul+div pipeline
            "hard_sigmoid": 3,                                     # clip+add+mul pipeline
            "global_avg_pool": 8,                                  # reduction tree
            "maxpool": int(sfu.get("pipeline_cycles", {}).get("maxpool", 3)),
            "avgpool": int(sfu.get("pipeline_cycles", {}).get("avgpool", 3)),
        }

    def estimate(self, op_type: str, num_elements: int) -> int:
        """Return cycles for SFU operation on num_elements."""
        latency = self.latency_map.get(op_type.lower(), 4)
        batches = (num_elements + self.width - 1) // self.width
        return batches * latency

    def estimate_softmax_decomposed(self, num_elements: int) -> Dict[str, int]:
        """Decomposed softmax: returns SFU-only portions.

        Softmax(x) = exp(x - max) / sum(exp(x - max))

        SFU handles: exp, div
        Vector handles: max_reduce, sub, sum_reduce

        Returns dict with 'exp' and 'div' cycle counts.
        The caller is responsible for adding Vector portion via VectorModel.
        """
        batches = (num_elements + self.width - 1) // self.width
        return {
            "exp": batches * self.latency_map["exp"],
            "div": batches * self.latency_map["div"],
        }

    def estimate_attention_sfu(self, hidden_size: int,
                                num_heads: int = 32) -> Dict[str, int]:
        """Estimate SFU portion of attention for one decode token.

        Per head: softmax over head_dim × seq_len
        For decode (M=1): attention scores are 1 × seq_len per head,
        but Q*K^T gives [1, seq_len] scores → softmax over seq_len elements.

        Actually: softmax is over the last dimension. For attention:
        Q[1, head_dim] × K^T[head_dim, seq_len] → scores[1, seq_len]
        Softmax over seq_len elements per head, total num_heads × seq_len.

        But seq_len varies. In our simulator, we model per-token decode,
        so seq_len = current context length (modeled via KV cache, not here).
        We only model the element count.

        Returns SFU cycle counts for one attention block.
        """
        # Per-head softmax: head_dim elements per head
        head_dim = hidden_size // num_heads
        # Actually, softmax is over the attention scores, which is seq_len per head
        # For simplicity, use hidden_size as a proxy for total attention compute
        decomposed = self.estimate_softmax_decomposed(hidden_size)
        return {
            "attn_exp": decomposed["exp"],
            "attn_div": decomposed["div"],
        }

    def estimate_all_layer(self, hidden_size: int, intermediate_size: int,
                           has_attention: bool = True, has_rope: bool = True) -> Tuple[int, Dict[str, int]]:
        """Estimate ALL SFU operations for one transformer layer.

        Returns (total_cycles, breakdown_dict)
        """
        breakdown = {}
        total = 0

        if has_attention:
            dec = self.estimate_softmax_decomposed(hidden_size)
            breakdown["softmax_exp"] = dec["exp"]
            breakdown["softmax_div"] = dec["div"]
            total += dec["exp"] + dec["div"]

        # Post-attention layernorm: uses div(SFU) + vector ops
        ln = self.estimate("layernorm", hidden_size)
        breakdown["ln1"] = ln
        total += ln

        # FFN gelu: uses tanh_approx(SFU) + vector mul
        gelu = self.estimate("gelu", intermediate_size)
        breakdown["gelu"] = gelu
        total += gelu

        # Post-FFN layernorm
        ln2 = self.estimate("layernorm", hidden_size)
        breakdown["ln2"] = ln2
        total += ln2

        # RoPE
        if has_rope:
            rope = self.estimate("rope", hidden_size * 2)
            breakdown["rope"] = rope
            total += rope

        return total, breakdown
