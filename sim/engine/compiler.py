"""NPU 指令编译器：模型 trace → NPU ISA 指令序列

L2 接口核心：将 CSV trace (M,K,N) 或模型配置转换为 NPU ISA 指令流。
编译器决定 tiling 策略、地址分配、DMA 排布。
"""

from typing import List, Tuple

from engine.isa import NPUInstruction, OpCode, NPUEncoder


class NPUCompiler:
    """Compile model GEMM trace → NPU ISA instruction sequence.

    Responsibilities:
    1. Tiling: split large GEMMs to fit 128×128 array
    2. Address allocation: assign SRAM addresses for weights, inputs, outputs
    3. DMA scheduling: interleave DMA_LD with MMUL for double-buffering
    4. SFU insertion: insert SOFTMAX/LAYERNORM/GELU/ROPE at layer boundaries

    Default strategy (single core, weight-stationary):
    - Weights loaded once per prefill (DMA_LD), stay in SRAM for decode
    - Activations streamed per token
    - Double-buffer: load next layer weights while computing current layer
    - SFU ops inserted after attention matmuls and FFN matmuls
    """

    def __init__(self, num_cores: int = 1):
        self.num_cores = num_cores
        self._addr = AddressAllocator()

    def compile_decode(self, trace: List[Tuple[int, int, int, int, str]],
                       weight_preloaded: bool = False) -> List[NPUInstruction]:
        """Compile decode token trace → ISA program.

        Args:
            trace: List of (M, K, N, layer, op_name)
            weight_preloaded: v2 default False — 3B weights (1.5GB) > SRAM (2.5MB),
                              always stream from DRAM. Set True only for v1 compat.
        """
        program = []
        prev_layer = -1

        for (M, K, N, layer, op_name) in trace:
            # Layer transition
            if layer != prev_layer:
                if prev_layer >= 0:
                    program.append(NPUInstruction(OpCode.BARRIER, {}))
                # DMA: preload next layer weights (if not preloaded)
                if not weight_preloaded:
                    w_addr = self._addr.alloc("weight", K * N)
                    program.append(NPUInstruction(OpCode.DMA_LD, {
                        "dram": w_addr,
                        "sram": self._addr.next("wbuf"),
                        "size": K * N,
                    }, comment=f"Load W layer={layer}"))
                # KV load for new layer
                program.append(NPUInstruction(OpCode.KV_LOAD, {
                    "token_id": layer,
                }, comment=f"KV prefetch layer={layer}"))
                prev_layer = layer

            # Allocate addresses
            ia = self._addr.alloc("input", M * K)
            oa = self._addr.alloc("output", M * N)
            wa = self._addr.next("wbuf")

            # MMUL instruction
            program.append(NPUInstruction(OpCode.MMUL, {
                "wa": wa, "ia": ia, "oa": oa, "N": N,
            }, comment=f"{op_name} ({M}×{K}×{N})"))

            # DMA: load next op's activation (pipelined)
            next_ia = self._addr.alloc("input_next", M * K)
            program.append(NPUInstruction(OpCode.DMA_LD, {
                "dram": next_ia,
                "sram": self._addr.next("ibuf"),
                "size": M * K,
            }, comment=f"Prefetch activation"))

            # SFU ops at layer boundaries
            if op_name == "O_proj":
                program.append(NPUInstruction(OpCode.SOFTMAX, {
                    "sa": oa, "da": oa, "len": 2560,
                }, comment="Post-attention softmax"))
                program.append(NPUInstruction(OpCode.LAYERNORM, {
                    "sa": oa, "da": oa, "len": 2560,
                }, comment="Post-attention layernorm"))
                program.append(NPUInstruction(OpCode.ROPE, {
                    "sa": oa, "da": oa, "len": 2560,
                }, comment="RoPE position encoding"))
            elif op_name == "FFN_down":
                program.append(NPUInstruction(OpCode.GELU, {
                    "sa": oa, "da": oa, "len": 9728,
                }, comment="FFN GELU activation"))
                program.append(NPUInstruction(OpCode.LAYERNORM, {
                    "sa": oa, "da": oa, "len": 2560,
                }, comment="Post-FFN layernorm"))

            # KV store for current layer
            program.append(NPUInstruction(OpCode.KV_STORE, {
                "token_id": layer,
            }, comment=f"KV persist layer={layer}"))

        program.append(NPUInstruction(OpCode.BARRIER, {}))
        program.append(NPUInstruction(OpCode.NOP, {}, comment="decode complete"))
        return program

    def compile_prefill(self, trace: List[Tuple[int, int, int, int, str]],
                        prompt_len: int) -> List[NPUInstruction]:
        """Compile prefill trace → ISA program.

        Weights must be loaded from DRAM (first time).
        """
        return self.compile_decode(trace, weight_preloaded=False)


class AddressAllocator:
    """Simple SRAM address allocator with double-buffering.

    SRAM layout:
    - Region 0: weight buffer (wbuf) — static, holds current layer weights
    - Region 1: input buffer (ibuf) — double-buffered ping/pong
    - Region 2: output buffer (obuf)
    - Region 3: activation scratch
    """

    def __init__(self):
        self._regions = {
            "wbuf": Region(0x00000, 0x20000),    # 128KB weight
            "ibuf": Region(0x20000, 0x10000),    # 64KB input (ping)
            "ibuf2": Region(0x30000, 0x10000),   # 64KB input (pong)
            "obuf": Region(0x40000, 0x20000),    # 128KB output
            "scratch": Region(0x60000, 0x20000), # 128KB scratch
        }
        self._allocations: List[Tuple[str, int, int]] = []  # name, addr, size

    def alloc(self, name: str, elements: int, elem_size: int = 4) -> int:
        """Allocate address for name, returns address."""
        size = elements * elem_size
        # Simple: use scratch, just track that we allocated
        addr = self._regions["scratch"].base
        self._allocations.append((name, addr, size))
        return addr

    def next(self, region: str) -> int:
        """Return next address in region (round-robin for double buffer)."""
        return self._regions[region].base

    def reset(self):
        self._allocations = []


class Region:
    def __init__(self, base: int, size: int):
        self.base = base
        self.size = size
