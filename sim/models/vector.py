"""Vector 单元延迟模型 — 逐元素运算流水线

Vector Unit 处理简单逐元素操作（add, mul, scale, bias, mask, relu）。
与 SFU（复杂数学）和 MXU（矩阵乘）并行，共享 SRAM 带宽。

Pipeline: MXU output → Vector reduce/reshape → SFU (exp/div) → Vector finalize
"""

from typing import Any, Dict


class VectorModel:
    """宽度可配的向量流水线单元。

    ops/cycle: 每个 cycle 完成的操作数（取决于数据宽度和并行度）。
    默认 128 位宽，FP16 = 8 ops/cycle，INT8 = 16 ops/cycle。
    """

    def __init__(self, config: Dict[str, Any]):
        vec = config.get("vector", {})
        self.width = int(vec.get("width", 128))        # SIMD width (elements)
        self.freq_mhz = int(config.get("mxu", {}).get("frequency_mhz", 1000))

        # Latency per batch: how many cycles to flush a full-width batch
        ops = vec.get("ops", {})
        self.op_latency = {
            "add": int(ops.get("add", 1)),      # element-wise add
            "mul": int(ops.get("mul", 1)),      # element-wise multiply
            "scale": int(ops.get("scale", 1)),  # scalar broadcast multiply
            "bias": int(ops.get("bias", 1)),    # element-wise add bias
            "relu": int(ops.get("relu", 1)),    # max(0, x)
            "mask": int(ops.get("mask", 1)),    # boolean mask select
            "max": int(ops.get("max", 12)),     # reduce_tree: element-wise max
            "sum": int(ops.get("sum", 12)),     # reduce_tree: sum reduce
            "reduce": int(ops.get("reduce", 12)),  # generic reduce_tree path
            "conv_f16_i32": int(ops.get("conv_f16_i32", 260)),  # type_convert: INT32→FP16
            "resid": int(ops.get("resid", 5)),  # residual add
        }

        # DMA bandwidth shared with MXU (SRAM ↔ Vector)
        mem = config.get("memory", {})
        self.dram_bw = float(mem.get("bandwidth_bytes_per_cycle", 51.2))
        self.dram_efficiency = float(mem.get("dram_efficiency", 0.85))
        self.effective_bw = self.dram_bw * self.dram_efficiency

    def estimate(self, op_type: str, num_elements: int) -> int:
        """Return cycles for processing num_elements through op_type.

        op_type: 'add', 'mul', 'scale', 'bias', 'relu', 'mask'
        """
        latency_per_batch = self.op_latency.get(op_type.lower(), 1)
        batches = (num_elements + self.width - 1) // self.width
        return batches * latency_per_batch

    def estimate_softmax_vector_parts(self, num_elements: int) -> Dict[str, int]:
        """Estimate Vector-only portions of softmax (not exp/div).

        Softmax = max_reduce → sub → exp(SFU) → sum_reduce → div(SFU)
        Vector handles: max_reduce, sub, sum_reduce
        SFU handles: exp, div

        Returns dict of {step: cycles}
        """
        batches = (num_elements + self.width - 1) // self.width
        return {
            "max_reduce": batches * self.op_latency["max"],   # tree reduction
            "scale_sub": batches * self.op_latency["scale"],  # subtract max
            "sum_reduce": batches * self.op_latency["sum"],   # tree reduction
        }

    def estimate_residual_add(self, num_elements: int) -> int:
        """Residual connection: x = x + attn_out (or x = x + ffn_out)."""
        return self.estimate("add", num_elements)

    def estimate_data_movement(self, num_bytes: int) -> int:
        """Estimate cycles to move data between MXU output and Vector unit.

        Both share SRAM; worst case goes through DRAM if SRAM is full.
        For simplicity, assume L1 SRAM hit (1 cycle per 256-bit word).
        """
        words = (num_bytes + 31) // 32  # 256-bit = 32 bytes
        return words  # 1 cycle per word on SRAM
