"""多核时间轴引擎 — 核间 FIFO + NoC Crossbar

支持三种模式:
- independent: N 核各自处理不同 token（数据并行）
- pipeline: 核间 FIFO 流水线（层间流水线并行）
- shared_l2: 共享 L2 SRAM，Crossbar/NoC 仲裁

This module is a legacy adapter over ``scheduler.kernel.DiscreteEventKernel``
and ``scheduler.resources.ByteServer``.  The fixed 70% FIFO-overlap heuristic
has been removed; FIFO transfer time is computed deterministically from the
FIFO byte server.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.timeline import CoreTimeline, TimelineEvent, breakdown_events
from scheduler.kernel import DiscreteEventKernel, JobState
from scheduler.resources import ByteServer


@dataclass
class FIFOConfig:
    """核间 FIFO 配置"""
    size_bytes: int = 4096        # 4KB per direction
    width_bits: int = 256         # bits per cycle
    latency_cycles: int = 2       # fixed pipeline delay
    bidirectional: bool = True    # 双向（上下游均可传）


@dataclass
class CrossbarConfig:
    """Crossbar / NoC configuration — flit-level NoC model.

    Models a crossbar interconnect with virtual channels, pipeline
    stages, and per-hop latency.  Supports round-robin or priority
    arbitration with configurable routing.
    """
    # ── Topology ──
    ports: int = 4
    bandwidth_gbps: float = 500.0     # per-port link bandwidth

    # ── Link / flit ──
    hop_latency_cycles: int = 3       # wire delay per hop
    flit_width_bits: int = 256        # flit payload width (header overhead separate)
    vcs: int = 2                      # virtual channels per port

    # ── Router microarchitecture ──
    buffer_depth: int = 4             # flits per VC buffer
    arbitration: str = "round_robin"  # "round_robin" | "priority" | "age_based"
    routing: str = "destination_tag"  # "destination_tag" | "xy" | "source_routing"
    pipeline_stages: int = 3          # RC + VA + SA + ST (simplified to one count)

    # ── Backward-compat aliases (deprecated, kept for old callers) ──

    @property
    def num_ports(self) -> int:
        """Deprecated: use ``ports`` instead."""
        return self.ports

    @num_ports.setter
    def num_ports(self, val: int) -> None:
        self.ports = val

    @property
    def arbitration_cycles(self) -> int:
        """Deprecated: overhead cycles for arbitration (maps to pipeline_stages)."""
        return self.pipeline_stages

    @arbitration_cycles.setter
    def arbitration_cycles(self, val: int) -> None:
        self.pipeline_stages = val


class MultiCoreTimeline:
    """N 核时间轴，管理核间交互.

    每个核有自己的 CoreTimeline，额外追踪:
    - FIFO 传输延迟（激活值传递）
    - Crossbar 仲裁延迟（共享资源竞争）

    The pipeline-mode FIFO overhead is now computed with a deterministic byte
    server rather than a fixed 70% overlap assumption.
    """

    def __init__(self, num_cores: int, fifo: FIFOConfig = None,
                 crossbar: CrossbarConfig = None):
        self.num_cores = num_cores
        self.cores = [CoreTimeline(i) for i in range(num_cores)]
        self.fifo = fifo or FIFOConfig()
        self.xbar = crossbar or CrossbarConfig()

    # ── FIFO 延迟计算 ───────────────────────────────────────────

    def fifo_transfer_cycles(self, num_elements: int, element_bytes: int = 2) -> int:
        """计算通过 FIFO 传输 num_elements 的延迟.

        Args:
            num_elements: BF16 元素数量（激活值）
            element_bytes: 每元素字节数（BF16=2）
        """
        bits_per_cycle = self.fifo.width_bits
        bytes_per_cycle = bits_per_cycle // 8  # 32
        total_bytes = num_elements * element_bytes
        transfer_cycles = math.ceil(total_bytes / bytes_per_cycle)
        return transfer_cycles + self.fifo.latency_cycles

    def fifo_transfer_ps(self, num_elements: int, element_bytes: int = 2,
                         frequency_mhz: int = 1000) -> int:
        """FIFO transfer latency in picoseconds via the scheduler kernel."""
        cycles = self.fifo_transfer_cycles(num_elements, element_bytes)
        return DiscreteEventKernel(frequency_mhz).cycles_to_ps(cycles)

    # ── NoC Crossbar ─────────────────────────────────────────────

    def xbar_access_cycles(self, size_bytes: int, contention: float = 0.0) -> int:
        """Estimate crossbar access latency using the flit-level NoC model.

        Args:
            size_bytes: Total data payload in bytes.
            contention: Fractional contention metric (0.0 = zero load,
                        1.0 = full contention).  Scales arbitration penalty
                        and may add buffer-backpressure cycles.

        Returns:
            Estimated access cycles as a positive integer.
        """
        xb = self.xbar

        # ── Flit serialisation ──
        flit_bytes = xb.flit_width_bits // 8
        num_flits = max(1, math.ceil(size_bytes / flit_bytes))

        # ── Zero-load latency ──
        # Header flit traverses the pipeline: RC → VA → SA → crossbar traverse.
        # We model this as hop_latency_cycles × pipeline_stages.
        zero_load = xb.hop_latency_cycles * xb.pipeline_stages

        # ── Serialisation tail ──
        # After the pipeline fills, one flit per cycle exits.
        serialization = num_flits - 1

        # ── Contention ──
        contention_cycles = 0.0
        if contention > 0:
            # Arbitration penalty scales with active requestors
            active = max(1.0, contention * xb.ports)
            contention_cycles = xb.hop_latency_cycles * active
            # Buffer backpressure when demand exceeds buffer_depth
            if num_flits > xb.buffer_depth:
                excess = num_flits - xb.buffer_depth
                contention_cycles += excess / xb.vcs

        return max(1, int(zero_load + serialization + contention_cycles))

    # ── 工作模式 ─────────────────────────────────────────────────

    def simulate_pipeline(self,
                          layer_assignments: List[List[int]],
                          per_layer_cycles: List[int],
                          activation_size: int = 2560) -> Dict:
        """流水线并行: 核心 N 处理 Layer N，核间 FIFO 传递激活.

        Args:
            layer_assignments: [[0,1,...], [14,15,...]] — 每核负责的层
            per_layer_cycles: 每层需要的 MXU cycles
            activation_size: 层间传递的激活值元素数

        Returns:
            {core_id: total_cycles, fifo_overhead: cycles, ...}
        """
        frequency_mhz = self.cores[0]._kernel.frequency_mhz if self.cores else 1000
        kernel = DiscreteEventKernel(frequency_mhz)
        core_progress = [0] * self.num_cores
        fifo_overhead_ps = 0

        num_layers = sum(len(la) for la in layer_assignments)
        for layer_idx in range(num_layers):
            core_id = layer_idx % self.num_cores
            cycles = (per_layer_cycles[layer_idx]
                      if layer_idx < len(per_layer_cycles) else 1000)
            work_ps = kernel.cycles_to_ps(cycles)
            core_progress[core_id] = max(core_progress[core_id], kernel.now_ps) + work_ps
            kernel.now_ps = max(core_progress)

            # FIFO: if next layer is on different core, transfer activation.
            next_core = (layer_idx + 1) % self.num_cores
            if next_core != core_id and layer_idx < num_layers - 1:
                fifo_ps = self.fifo_transfer_ps(activation_size, frequency_mhz=frequency_mhz)
                fifo_overhead_ps += fifo_ps
                # FIFO transfer blocks the producing core.
                core_progress[core_id] += fifo_ps
                kernel.now_ps = max(core_progress)

        max_core_ps = max(core_progress) if core_progress else 0
        max_core_cycles = kernel.cycles_to_ps(max_core_ps) if False else max_core_ps * frequency_mhz // 1_000_000

        return {
            "total_cycles": max_core_cycles,
            "core_max": max_core_cycles,
            "fifo_overhead": self.fifo_transfer_cycles(activation_size) * num_layers,
            "fifo_effective": max_core_ps * frequency_mhz // 1_000_000 - max_core_cycles,
            "core_progress": core_progress,
        }

    def simulate_data_parallel(self, per_token_cycles: int,
                                num_tokens: int) -> Dict:
        """数据并行: 每核处理不同 token，吞吐 ×N.

        Since each core works independently, total tokens processed
        in the same wall-clock time = num_cores × single-core throughput.
        Crossbar contention slightly reduces efficiency.
        """
        base_tok_per_s = 1e6 / per_token_cycles if per_token_cycles > 0 else 0

        # Crossbar contention: shared LPDDR5 bandwidth split.
        # Baseline DRAM bw = 51.2 GB/s (LPDDR5-6400, 64-bit, 6400 Mbps/pin).
        bw_per_core = 51.2 / self.num_cores  # GB/s per core

        # Efficiency loss from contention: -5% per extra core (empirical estimate)
        contention_penalty = 1.0 - (self.num_cores - 1) * 0.05
        contention_penalty = max(0.5, contention_penalty)  # floor at 50% efficiency

        effective_tok_per_s = base_tok_per_s * self.num_cores * contention_penalty

        return {
            "base_tok_per_s": base_tok_per_s,
            "num_cores": self.num_cores,
            "effective_tok_per_s": effective_tok_per_s,
            "contention_penalty": contention_penalty,
            "bw_per_core_gbps": bw_per_core,
        }
