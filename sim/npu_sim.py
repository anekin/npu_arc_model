#!/usr/bin/env python3
"""NPU System Simulator — Phase 3: MXU + SFU + DMA + KV Cache + DRAM + NoC

Usage:
    python3 npu_sim.py                          # default settings
    python3 npu_sim.py -c config/npu_2core.yaml # custom config
    python3 npu_sim.py --prefill 2048           # longer prompt
    python3 npu_sim.py --json                   # JSON output
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Add parent to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from engine.mac_engine import create_engine
from models.sfu import SFUModel
from models.vector import VectorModel
from models.dma import DMAModel
from models.kv_cache import KVCacheModel
from models.dram import DRAMModel
from models.noc import NoCModel
from engine.timeline import (
    CoreTimeline, LayerBreakdown, SimulationReport, breakdown_events,
)
from engine.multicore import MultiCoreTimeline, FIFOConfig, CrossbarConfig


# ── Default 3B model trace ──────────────────────────────────────────
# Each GEMM: (M, K, N, layer, op_name)
# Derived from Qwen2.5-3B config (36 layers, hidden=2048, intermediate=11008)
# Decode: M=1 (single token), Prefill: M=128

def generate_qwen3b_trace(prompt_len: int = 128) -> List[Tuple[int, int, int, int, str]]:
    """Generate GEMM trace from Qwen2.5-3B architecture.

    Each transformer layer has 7 matmuls:
    - Q projection: (M, 2048, 2048) — Q full heads
    - K projection: (M, 2048, 2048) — KV heads (GQA=16)
    - V projection: (M, 2048, 2048)
    - O projection: (M, 2048, 2048)
    - FFN gate: (M, 2048, 11008) — SiLU gate
    - FFN up: (M, 2048, 11008)
    - FFN down: (M, 11008, 2048)
    """
    HIDDEN = 2048
    INTERMEDIATE = 11008
    NUM_LAYERS = 36
    NUM_HEADS = 16
    NUM_KV_HEADS = 16
    HEAD_DIM = 128
    QKV_DIM = NUM_HEADS * HEAD_DIM   # 16 * 128 = 2048
    KV_DIM = NUM_KV_HEADS * HEAD_DIM # 16 * 128 = 2048

    trace = []
    for layer in range(NUM_LAYERS):
        trace.append((prompt_len, HIDDEN, QKV_DIM, layer, "Q_proj"))
        trace.append((prompt_len, HIDDEN, KV_DIM, layer, "K_proj"))
        trace.append((prompt_len, HIDDEN, KV_DIM, layer, "V_proj"))
        trace.append((prompt_len, QKV_DIM, HIDDEN, layer, "O_proj"))
        trace.append((prompt_len, HIDDEN, INTERMEDIATE, layer, "FFN_gate"))
        trace.append((prompt_len, HIDDEN, INTERMEDIATE, layer, "FFN_up"))
        trace.append((prompt_len, INTERMEDIATE, HIDDEN, layer, "FFN_down"))
    return trace


# ── Simulator ────────────────────────────────────────────────────────

class NPUSimulator:
    """Phase 1: Single-core NPU performance simulator."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.num_cores = int(self.config.get("cores", 1))
        self.f_mhz = int(self.config["mxu"]["frequency_mhz"])

        # Initialize models
        self.mxu = create_engine(self.config)
        self.sfu = SFUModel(self.config)
        self.vector = VectorModel(self.config)
        self.dma = DMAModel(self.config)
        self.kv = KVCacheModel(self.config)
        self.dram = DRAMModel(self.config)
        self.noc = NoCModel(self.config)

        # Configure KV cache for Qwen2.5-3B
        self.kv.configure_for_model(
            num_kv_heads=16, head_dim=128, num_layers=36, max_context=2048
        )

    def simulate_decode(self, trace: List[Tuple[int, int, int, int, str]]) -> SimulationReport:
        """Simulate decode: M=1 per GEMM — v2 bandwidth-aware model.

        v2: Weights stream from DRAM every token (too large for SRAM).
        MXU v2 model accounts for weight tiling + DMA overlap internally.

        v3: weight_cache optimization — merges FFN_gate+FFN_up pairs
        that share (M,K) into a single dual-weight-register estimation.
        """
        timeline = CoreTimeline(core_id=0)
        layer_data: Dict[int, LayerBreakdown] = {}
        total_tokens = 128  # KV cache tracked context depth (baseline for Qwen2.5-3B)
        total_weight_bytes = 0

        # Check optimizations
        opts = self.config.get("optimizations", {})
        weight_cache_enabled = opts.get("weight_cache", False)

        debug_enabled = False  # Set True for per-op wall-clock tracing

        # Pre-process: merge gate+up pairs if weight_cache enabled
        i = 0
        while i < len(trace):
            M, K, N, layer, op_name = trace[i]

            if layer not in layer_data:
                layer_data[layer] = LayerBreakdown(layer=layer)
                _prev_cycle = timeline._current_cycle
                kv_switch = self.kv.layer_switch_cost()
                timeline.add_kv("layer_switch", kv_switch, layer)
                layer_data[layer].kv_cache += kv_switch
                if debug_enabled:
                    print(f"[DEBUG L{layer}/layer_switch] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

            # ── Weight Cache: merge FFN_gate + FFN_up ──
            if (weight_cache_enabled and op_name == "FFN_gate"
                    and i + 1 < len(trace)):
                next_M, next_K, next_N, next_layer, next_op = trace[i + 1]
                if (next_op == "FFN_up" and next_M == M
                        and next_K == K and next_layer == layer):
                    # Merged estimation
                    _prev_cycle = timeline._current_cycle
                    mxu_result = self.mxu.estimate_weight_cache_pair(M, K, N)
                    mxu_cycles = mxu_result.total_cycles
                    timeline.add_mxu(
                        f"Gate+Up (cache, {M}×{K}×{N}, "
                        f"{mxu_result.num_tiles} dual-tiles)",
                        mxu_cycles, layer)
                    layer_data[layer].mxu += mxu_cycles
                    total_weight_bytes += mxu_result.weight_bytes
                    if debug_enabled:
                        print(f"[DEBUG L{layer}/Gate+Up_cache] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

                    # DMA: merged weight pair event for breakdown tracking
                    mxu_end = timeline._current_cycle
                    timeline.add_dma_parallel(
                        f"dma_weights Gate+Up {M}x{K}x{N}",
                        mxu_result.dma_cycles, layer)
                    timeline._current_cycle = mxu_end

                    effective, hidden = self.dma.estimate_effective(
                        mxu_result.dma_cycles, mxu_result.compute_cycles)
                    layer_data[layer].dma_effective += effective
                    layer_data[layer].dma_weight += hidden

                    # NoC: merged weight pair transfer over interconnect (breakdown only)
                    noc_cycles = self.noc.estimate_transfer(
                        mxu_result.weight_bytes, src_id=0, dst_id=0)
                    timeline.add_noc(
                        f"noc_weights Gate+Up {M}x{K}x{N}",
                        noc_cycles, layer)
                    timeline._current_cycle = mxu_end
                    layer_data[layer].noc_latency += noc_cycles

                    # KV access (once for the pair)
                    _prev_cycle = timeline._current_cycle
                    kv_cycles = self.kv.estimate_per_decode(total_tokens, total_tokens)
                    timeline.add_kv("kv_access", kv_cycles, layer)
                    layer_data[layer].kv_cache += kv_cycles
                    if debug_enabled:
                        print(f"[DEBUG L{layer}/kv_cache_pair] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

                    i += 2  # skip both gate and up
                    continue

            # ── Standard single GEMM ──
            _prev_cycle = timeline._current_cycle
            mxu_result = self.mxu.estimate(M, K, N)
            mxu_cycles = mxu_result.total_cycles
            timeline.add_mxu(
                f"{op_name} ({M}×{K}×{N}, {mxu_result.num_tiles}tiles)",
                mxu_cycles, layer)
            layer_data[layer].mxu += mxu_cycles
            total_weight_bytes += mxu_result.weight_bytes
            if debug_enabled:
                print(f"[DEBUG L{layer}/{op_name}] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

            # DMA: weight streaming event for breakdown tracking.
            # Engine total_cycles already accounts for DMA stall, so we
            # add the DMA event and then restore the MXU-end position.
            mxu_end = timeline._current_cycle
            timeline.add_dma_parallel(
                f"dma_weights {op_name} {M}x{K}x{N}",
                mxu_result.dma_cycles, layer)
            timeline._current_cycle = mxu_end

            effective, hidden = self.dma.estimate_effective(
                mxu_result.dma_cycles, mxu_result.compute_cycles)
            layer_data[layer].dma_effective += effective
            layer_data[layer].dma_weight += hidden

            # NoC: weight transfer over interconnect (breakdown only)
            noc_cycles = self.noc.estimate_transfer(
                mxu_result.weight_bytes, src_id=0, dst_id=0)
            timeline.add_noc(
                f"noc_weights {op_name} {M}x{K}x{N}",
                noc_cycles, layer)
            timeline._current_cycle = mxu_end
            layer_data[layer].noc_latency += noc_cycles

            # ── SFU + Vector: decomposed softmax ──
            if op_name in ("O_proj",):
                # Softmax decomposition: Vector(max_reduce, sub) → SFU(exp) → Vector(sum_reduce) → SFU(div)
                HIDDEN = 2048
                vec_softmax = self.vector.estimate_softmax_vector_parts(HIDDEN)
                sfu_softmax = self.sfu.estimate_softmax_decomposed(HIDDEN)

                sfu_cycles = vec_softmax["max_reduce"] + sfu_softmax["exp"] + vec_softmax["sum_reduce"] + sfu_softmax["div"]
                sfu_cycles += self.sfu.estimate("layernorm", HIDDEN)
                sfu_cycles += self.sfu.estimate("rope", HIDDEN * 2)
                _prev_cycle = timeline._current_cycle
                timeline.add_sfu("attn_sfu (exp+div+ln+rope)", sfu_cycles, layer)
                layer_data[layer].sfu += sfu_cycles
                if debug_enabled:
                    print(f"[DEBUG L{layer}/attn_sfu] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

                vec_cycles = vec_softmax["scale_sub"] + self.vector.estimate_residual_add(HIDDEN)
                _prev_cycle = timeline._current_cycle
                timeline.add_vector("attn_vec (sub+residual)", vec_cycles, layer)
                layer_data[layer].vector += vec_cycles
                if debug_enabled:
                    print(f"[DEBUG L{layer}/attn_vec] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

            elif op_name in ("FFN_down",):
                INTERMEDIATE = 11008
                HIDDEN = 2048
                sfu_cycles = self.sfu.estimate("gelu", INTERMEDIATE)
                sfu_cycles += self.sfu.estimate("layernorm", HIDDEN)
                _prev_cycle = timeline._current_cycle
                timeline.add_sfu("ffn_sfu (gelu+ln)", sfu_cycles, layer)
                layer_data[layer].sfu += sfu_cycles
                if debug_enabled:
                    print(f"[DEBUG L{layer}/ffn_sfu] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

                vec_cycles = self.vector.estimate_residual_add(HIDDEN)
                _prev_cycle = timeline._current_cycle
                timeline.add_vector("ffn_vec (residual)", vec_cycles, layer)
                layer_data[layer].vector += vec_cycles
                if debug_enabled:
                    print(f"[DEBUG L{layer}/ffn_vec] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

            # KV Cache: per-GEMM access
            _prev_cycle = timeline._current_cycle
            kv_cycles = self.kv.estimate_per_decode(total_tokens, total_tokens)
            timeline.add_kv("kv_access", kv_cycles, layer)
            layer_data[layer].kv_cache += kv_cycles
            if debug_enabled:
                print(f"[DEBUG L{layer}/kv_access] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

            # Update layer total
            layer_data[layer].total = (layer_data[layer].mxu + layer_data[layer].sfu
                                        + layer_data[layer].vector + layer_data[layer].kv_cache
                                        + layer_data[layer].dma_effective)

            i += 1

        # Add DRAM refresh overhead (proportional to total)
        total_cycles_before = timeline.total_cycles
        _prev_cycle = timeline._current_cycle
        refresh_cycles = self.dram.add_refresh_overhead(total_cycles_before)
        timeline.add_kv("dram_refresh", refresh_cycles, -1)
        if debug_enabled:
            print(f"[DEBUG SYSTEM/dram_refresh] wall_clock += {timeline._current_cycle - _prev_cycle:>10,}  (accumulated: {timeline._current_cycle:>12,})")

        total_cycles = timeline.total_cycles
        decode_us = total_cycles / self.f_mhz
        decode_tok_per_s = 1e6 / decode_us if decode_us > 0 else 0

        breakdown = breakdown_events(timeline.events)

        # Build report
        report = SimulationReport(
            model_name="Qwen2.5-3B",
            num_layers=36,
            array_height=int(self.config.get("mxu", {}).get("array_height", 128)),
            array_width=int(self.config.get("mxu", {}).get("array_width", 128)),
            weight_bits=int(self.config.get("mxu", {}).get("weight_precision_bits", 4)),
            freq_mhz=int(self.config.get("mxu", {}).get("frequency_mhz", 1000)),
            engine_type=self.mxu.engine_type,
            prefill_prompt_len=128,
            prefill_total_ms=0.0,  # Not simulated in Phase 1
            decode_per_token_us=decode_us,
            decode_tok_per_s=decode_tok_per_s,
            decode_breakdown={k: v / self.f_mhz for k, v in breakdown.items()},
            layer_breakdowns=sorted(layer_data.values(), key=lambda lb: lb.layer),
            events=timeline.events,
        )
        return report

    # ── L2: ISA instruction interface ───────────────────────────────

    def run_instructions(self, program: "List[NPUInstruction]") -> SimulationReport:
        """L2 interface: execute NPU ISA program on the simulator.

        Maps each ISA instruction to the appropriate model call,
        advancing the timeline with proper overlap semantics.
        """
        from engine.isa import NPUInstruction, OpCode

        timeline = CoreTimeline(core_id=0)
        layer_data: Dict[int, LayerBreakdown] = {}
        current_layer = 0

        for instr in program:
            op = instr.opcode
            ops = instr.operands

            if op in (OpCode.NOP, OpCode.BARRIER):
                continue

            if op == OpCode.MMUL:
                N = ops.get("N", 2048)
                M = 1  # decode mode
                K = 2048  # hidden_size
                mxu_result = self.mxu.estimate(M, K, N)
                timeline.add_mxu(f"MMUL N={N}", mxu_result.total_cycles, current_layer)
                if current_layer not in layer_data:
                    layer_data[current_layer] = LayerBreakdown(layer=current_layer)
                layer_data[current_layer].mxu += mxu_result.total_cycles

            elif op == OpCode.DMA_LD:
                size = ops.get("size", 0)
                dma_cycles = self.dma.estimate_transfer(size, "load")
                timeline.add_dma_parallel(f"DMA_LD {size}B", dma_cycles, current_layer)

            elif op == OpCode.DMA_ST:
                size = ops.get("size", 0)
                dma_cycles = self.dma.estimate_transfer(size, "store")
                timeline.add_dma_parallel(f"DMA_ST {size}B", dma_cycles, current_layer)

            elif op in (OpCode.SOFTMAX, OpCode.LAYERNORM, OpCode.GELU, OpCode.RELU,
                        OpCode.SILU, OpCode.MAXPOOL, OpCode.AVGPOOL, OpCode.ROPE):
                length = ops.get("len", 2048)
                op_name = op.name.lower()
                sfu_cycles = self.sfu.estimate(op_name, length)
                timeline.add_sfu(op_name, sfu_cycles, current_layer)
                if current_layer not in layer_data:
                    layer_data[current_layer] = LayerBreakdown(layer=current_layer)
                layer_data[current_layer].sfu += sfu_cycles

            elif op in (OpCode.KV_LOAD, OpCode.KV_STORE):
                kv_cycles = self.kv.estimate_per_decode(128, 128)
                timeline.add_kv(op.name.lower(), kv_cycles, current_layer)

        # DRAM refresh
        refresh_cycles = self.dram.add_refresh_overhead(timeline.total_cycles)
        timeline.add_kv("dram_refresh", refresh_cycles, -1)

        total_cycles = timeline.total_cycles
        decode_us = total_cycles / self.f_mhz
        decode_tok_per_s = 1e6 / decode_us if decode_us > 0 else 0

        breakdown = breakdown_events(timeline.events)

        return SimulationReport(
            model_name="Qwen2.5-3B",
            num_layers=36,
            array_height=int(self.config.get("mxu", {}).get("array_height", 128)),
            array_width=int(self.config.get("mxu", {}).get("array_width", 128)),
            weight_bits=int(self.config.get("mxu", {}).get("weight_precision_bits", 4)),
            freq_mhz=int(self.config.get("mxu", {}).get("frequency_mhz", 1000)),
            engine_type=self.mxu.engine_type,
            decode_per_token_us=decode_us,
            decode_tok_per_s=decode_tok_per_s,
            decode_breakdown={k: v / self.f_mhz for k, v in breakdown.items()},
            layer_breakdowns=sorted(layer_data.values(), key=lambda lb: lb.layer),
            events=timeline.events,
        )

    def simulate_prefill(self, prompt_len: int = 128) -> SimulationReport:
        """Simulate prefill: M=prompt_len per GEMM.

        Prefill is compute-heavy but bandwidth-friendly: large M means
        the systolic array stays full, utilization is high.
        """
        trace = generate_qwen3b_trace(prompt_len=prompt_len)
        timeline = CoreTimeline(core_id=0)
        layer_data: Dict[int, LayerBreakdown] = {}

        for (M, K, N, layer, op_name) in trace:
            if layer not in layer_data:
                layer_data[layer] = LayerBreakdown(layer=layer)

            mxu_result = self.mxu.estimate(M, K, N)
            mxu_cycles = mxu_result.total_cycles
            timeline.add_mxu(f"{op_name} ({M}×{K}×{N})", mxu_cycles, layer)
            layer_data[layer].mxu += mxu_cycles

            # DMA: weight streaming event for breakdown tracking
            mxu_end = timeline._current_cycle
            timeline.add_dma_parallel(
                f"dma_weights {op_name} {M}x{K}x{N}",
                mxu_result.dma_cycles, layer)
            timeline._current_cycle = mxu_end

            effective, hidden = self.dma.estimate_effective(
                mxu_result.dma_cycles, mxu_result.compute_cycles)
            layer_data[layer].dma_effective += effective
            layer_data[layer].dma_weight += hidden

            # NoC: weight transfer over interconnect (breakdown only)
            noc_cycles = self.noc.estimate_transfer(
                mxu_result.weight_bytes, src_id=0, dst_id=0)
            timeline.add_noc(
                f"noc_weights {op_name} {M}x{K}x{N}",
                noc_cycles, layer)
            timeline._current_cycle = mxu_end
            layer_data[layer].noc_latency += noc_cycles

            if op_name in ("O_proj",):
                sfu_cycles = self.sfu.estimate("softmax", 2048 * prompt_len)
                sfu_cycles += self.sfu.estimate("layernorm", 2048 * prompt_len)
                sfu_cycles += self.sfu.estimate("rope", 2048 * 2 * prompt_len)
                timeline.add_sfu("attn_sfu", sfu_cycles, layer)
                layer_data[layer].sfu += sfu_cycles
            elif op_name in ("FFN_down",):
                sfu_cycles = self.sfu.estimate("gelu", 11008 * prompt_len)
                sfu_cycles += self.sfu.estimate("layernorm", 2048 * prompt_len)
                timeline.add_sfu("ffn_sfu", sfu_cycles, layer)
                layer_data[layer].sfu += sfu_cycles

            kv_cycles = self.kv.estimate_per_decode(prompt_len, prompt_len)
            timeline.add_kv("kv_access", kv_cycles, layer)
            layer_data[layer].kv_cache += kv_cycles

            # Update layer total
            layer_data[layer].total = (layer_data[layer].mxu + layer_data[layer].sfu
                                        + layer_data[layer].kv_cache
                                        + layer_data[layer].dma_effective)

        # DRAM refresh overhead
        total_before = timeline.total_cycles
        refresh_cycles = self.dram.add_refresh_overhead(total_before)
        timeline.add_kv("dram_refresh", refresh_cycles, -1)

        total_cycles = timeline.total_cycles
        prefill_ms = total_cycles / self.f_mhz / 1000

        breakdown = breakdown_events(timeline.events)

        report = SimulationReport(
            model_name="Qwen2.5-3B",
            num_layers=36,
            prefill_prompt_len=prompt_len,
            prefill_total_ms=prefill_ms,
            prefill_breakdown={k: v / self.f_mhz / 1000 for k, v in breakdown.items()},
            decode_per_token_us=0,
            decode_tok_per_s=0,
            layer_breakdowns=sorted(layer_data.values(), key=lambda lb: lb.layer),
            events=timeline.events,
        )
        return report


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NPU System Simulator — Phase 3")
    parser.add_argument("-c", "--config", default="config/npu_config.yaml")
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--isa", action="store_true",
                        help="Use ISA instruction mode (L2 interface)")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--json", action="store_true")

    # ── Live override args ──
    parser.add_argument("--engine", default=None,
                        choices=["systolic", "os_systolic", "block", "tensor_core",
                                 "wmma", "gmma", "input_stationary"],
                        help="Override engine type (ignores config)")
    parser.add_argument("--dram", default=None,
                        choices=["25", "50", "100", "200", "460", "819"],
                        help="Override DRAM bandwidth in GB/s")
    parser.add_argument("--array", default=None,
                        help="Override array dimensions, e.g. 128x256")
    parser.add_argument("--freq", type=int, default=None,
                        help="Override frequency in MHz")
    parser.add_argument("--precision", type=int, default=None, choices=[2, 4, 8],
                        help="Override weight precision bits")
    parser.add_argument("--weight-cache", action="store_true", default=None,
                        help="Enable weight cache (PE dual register)")
    parser.add_argument("--list-engines", action="store_true",
                        help="List available engines and exit")
    parser.add_argument("--list-dram", action="store_true",
                        help="List DRAM bandwidth presets and exit")
    args = parser.parse_args()

    # List modes
    if args.list_engines:
        print("Available MAC engines:")
        print("  systolic          Weight-Stationary Systolic Array (TPUv1)")
        print("  os_systolic       Output-Stationary Systolic (Gemmini)")
        print("  block             Block Engine — full parallel MAC (TPUv4 VMU)")
        print("  tensor_core       Multi 16×16 Tensor Cores (A100 style)")
        print("  wmma              16×16 Warp MMA (Volta/Ampere style)")
        print("  gmma              Group MMA + TMA async DMA (Hopper H100 style)")
        print("  input_stationary  Input-Stationary (Eyeriss)")
        return
    if args.list_dram:
        print("DRAM bandwidth presets:")
        print("  25   LPDDR5-32b  (25.6 GB/s) — low-end mobile")
        print("  50   LPDDR5-64b  (51.2 GB/s) — baseline")
        print("  100  LPDDR5-128b (102.4 GB/s) — dual channel")
        print("  200  LPDDR5-256b (204.8 GB/s) — quad channel")
        print("  460  HBM2e-1024b (460 GB/s)")
        print("  819  HBM3-1024b  (819 GB/s)")
        return

    sim_dir = Path(__file__).parent
    config_path = sim_dir / args.config
    sim = NPUSimulator(str(config_path))

    # Override config from CLI args
    overrides = []
    cfg = sim.config
    if args.engine:
        mac = cfg.get("mxu", cfg.get("mac_engine", {}))
        mac["type"] = args.engine
        overrides.append(f"engine={args.engine}")
    if args.dram:
        bw = {"25": 25.6, "50": 51.2, "100": 102.4, "200": 204.8,
              "460": 460.0, "819": 819.2}[args.dram]
        mem = cfg.get("memory", {})
        mem["bandwidth_gbps"] = bw
        mem["bandwidth_bytes_per_cycle"] = bw
        mem["dram_width_bits"] = {"25": 32, "50": 64, "100": 128,
                                  "200": 256, "460": 1024, "819": 1024}[args.dram]
        overrides.append(f"DRAM={bw}GB/s")
    if args.array:
        h, w = args.array.split("x")
        mac = cfg.get("mxu", cfg.get("mac_engine", {}))
        mac["array_height"] = int(h)
        mac["array_width"] = int(w)
        overrides.append(f"array={h}×{w}")
    if args.freq:
        mac = cfg.get("mxu", cfg.get("mac_engine", {}))
        mac["frequency_mhz"] = args.freq
        overrides.append(f"freq={args.freq}MHz")
    if args.precision:
        mac = cfg.get("mxu", cfg.get("mac_engine", {}))
        mac["weight_precision_bits"] = args.precision
        overrides.append(f"INT{args.precision}")
    if args.weight_cache is not None:
        opts = cfg.get("optimizations", {})
        opts["weight_cache"] = args.weight_cache
        overrides.append(f"wc={args.weight_cache}")

    if overrides:
        if not args.json:
            print(f"[override] {', '.join(overrides)}")
        # Re-init models with overridden config
        sim.mxu = create_engine(cfg)
        sim.dma = DMAModel(cfg)
        sim.dram = DRAMModel(cfg)
        sim.config = cfg

    if args.isa:
        # ── L2: ISA instruction mode ─────────────────────────────
        from engine.compiler import NPUCompiler
        compiler = NPUCompiler(num_cores=1)
        decode_trace = generate_qwen3b_trace(prompt_len=1)
        program = compiler.compile_decode(decode_trace)  # v2: weights stream from DRAM (compiler default=False)
        print(f"[ISA] Compiled {len(program)} instructions for decode")
        report = sim.run_instructions(program)
    else:
        # ── L1: CSV trace mode ───────────────────────────────────
        decode_trace = generate_qwen3b_trace(prompt_len=1)
        report = sim.simulate_decode(decode_trace)

    # Inject prefill results
    prefill_report = sim.simulate_prefill(prompt_len=args.prefill)
    report.prefill_prompt_len = args.prefill
    report.prefill_total_ms = prefill_report.prefill_total_ms
    report.prefill_breakdown = prefill_report.prefill_breakdown

    # JSON output — when --json, suppress text report and multi-core projection
    if args.json or args.output:
        output = {
            "decode": {
                "engine_type": report.engine_type,
                "array_height": report.array_height,
                "array_width": report.array_width,
                "per_token_us": round(report.decode_per_token_us, 1),
                "tok_per_s": round(report.decode_tok_per_s, 4),
                "breakdown": {k: round(v, 1) for k, v in report.decode_breakdown.items()},
            },
            "prefill": {
                "prompt_len": args.prefill,
                "total_ms": round(prefill_report.prefill_total_ms, 1),
                "breakdown": {k: round(v, 1) for k, v in prefill_report.prefill_breakdown.items()},
            },
        }
        if args.json:
            print(json.dumps(output, indent=2))
        if args.output:
            with open(sim_dir / args.output, "w") as f:
                json.dump(output, f, indent=2)
                print(f"\nResults saved to {args.output}")
    else:
        print(report.to_text())

        # ── Multi-core projection ────────────────────────────────────
        if report.decode_tok_per_s > 0:
            print(f"\n--- Multi-core Projection ---")
            mct = MultiCoreTimeline(num_cores=1)
            base_us = report.decode_per_token_us
            print(f"  {'Config':12s} {'Decode':>10s} {'Area':>8s} {'Notes'}")
            print(f"  {'─'*50}")
            for nc in [1, 2, 4, 8]:
                mct.num_cores = nc
                dp = mct.simulate_data_parallel(int(base_us), 1)
                area = {1: 27, 2: 42, 4: 69, 8: 122}.get(nc, 0)
                note = "Baseline" if nc == 1 else f"DP, -{int((1-dp['contention_penalty'])*100)}% contention"
                print(f"  {f'{nc} core':12s} {dp['effective_tok_per_s']:7,.0f} tok/s  {area:4d}mm²  {note}")


if __name__ == "__main__":
    main()
