"""事件驱动时间轴引擎 — 核心调度器，合并 MXU/SFU/DMA 事件"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TimelineEvent:
    """A single event on the core timeline."""
    module: str          # 'mxu', 'sfu', 'dma', 'kv', 'riscv'
    op: str              # specific operation description
    start_cycle: int
    end_cycle: int
    layer: int = -1      # -1 = system-level event
    overlapped: bool = False  # was this hidden behind another event?


@dataclass
class LayerBreakdown:
    """Per-layer cycle breakdown."""
    layer: int
    mxu: int = 0
    sfu: int = 0
    vector: int = 0
    dma_weight: int = 0
    dma_effective: int = 0
    kv_cache: int = 0
    riscv: int = 0
    noc_latency: float = 0.0
    noc_contention: float = 0.0
    total: int = 0


@dataclass
class SimulationReport:
    """Complete simulation output."""
    model_name: str
    num_layers: int
    # NPU config for display
    array_height: int = 128
    array_width: int = 128
    weight_bits: int = 4
    freq_mhz: int = 1000
    engine_type: str = "systolic"
    # Prefill
    prefill_prompt_len: int = 0
    prefill_total_ms: float = 0.0
    prefill_breakdown: Dict[str, float] = field(default_factory=dict)
    # Decode (per token)
    decode_per_token_us: float = 0.0
    decode_tok_per_s: float = 0.0
    decode_breakdown: Dict[str, float] = field(default_factory=dict)
    # Detailed
    layer_breakdowns: List[LayerBreakdown] = field(default_factory=list)
    events: List[TimelineEvent] = field(default_factory=list)
    ttft_ms: float = 0.0
    decode_total_us: float = 0.0
    module_breakdowns: dict = field(default_factory=dict)
    tps: float = 0.0
    tpot_us: float = 0.0
    itl_us_list: list = field(default_factory=list)

    def to_text(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append(f"  NPU System Simulation Report")
        lines.append(f"  Model: {self.model_name} | Layers: {self.num_layers}")
        lines.append(f"  NPU: 1 core, {self.engine_type}, "
                     f"{self.array_height}×{self.array_width}, "
                     f"INT{self.weight_bits}, {self.freq_mhz}MHz")
        lines.append("=" * 60)

        # Prefill section
        if self.prefill_total_ms > 0:
            lines.append(f"\n--- Prefill (prompt={self.prefill_prompt_len} tokens) ---")
            for mod, ms in self.prefill_breakdown.items():
                pct = ms / self.prefill_total_ms * 100 if self.prefill_total_ms > 0 else 0
                lines.append(f"  {mod:20s} {ms:8.1f} ms  ({pct:5.1f}%)")
            lines.append(f"  {'─' * 36}")
            lines.append(f"  {'TOTAL':20s} {self.prefill_total_ms:8.1f} ms")

        # Decode section
        if self.decode_per_token_us > 0:
            lines.append(f"\n--- Decode (per token) ---")
            for mod, us in self.decode_breakdown.items():
                pct = us / self.decode_per_token_us * 100 if self.decode_per_token_us > 0 else 0
                lines.append(f"  {mod:20s} {us:8.1f} μs  ({pct:5.1f}%)")
            lines.append(f"  {'─' * 36}")
            lines.append(f"  {'TOTAL':20s} {self.decode_per_token_us:8.1f} μs")
            lines.append(f"  → {self.decode_tok_per_s:,.0f} tok/s")

            if self.decode_tok_per_s >= 25:
                lines.append(f"  ✅ Target 25 tok/s met!")
            else:
                lines.append(f"  ❌ Target 25 tok/s NOT met (gap: {25 - self.decode_tok_per_s:.0f} tok/s)")

            lines.append(f"\n--- Bottleneck Analysis ---")
            mxu_pct = self.decode_breakdown.get("MXU", 0) / self.decode_per_token_us * 100
            dma_pct = self.decode_breakdown.get("DMA (stall)", 0) / self.decode_per_token_us * 100
            if mxu_pct > 60:
                lines.append(f"  🔴 MXU dominates at {mxu_pct:.1f}% — compute-bound, consider wider array")
            else:
                lines.append(f"  🟢 MXU {mxu_pct:.1f}% — healthy")
            if dma_pct > 15:
                lines.append(f"  🔴 DMA stall {dma_pct:.1f}% — bandwidth-bound")
            elif dma_pct > 5:
                lines.append(f"  🟡 DMA stall {dma_pct:.1f}% — adequate")
            else:
                lines.append(f"  🟢 DMA stall {dma_pct:.1f}% — sufficient bandwidth")

        return "\n".join(lines)


class CoreTimeline:
    """Single-core event-driven timeline.

    Tracks overlapping events: MXU and DMA can run concurrently,
    SFU follows MXU (data dependency), RISC-V overhead is negligible.
    """

    def __init__(self, core_id: int = 0):
        self.core_id = core_id
        self.events: List[TimelineEvent] = []
        self._current_cycle: int = 0
        self._mxu_busy_until: int = 0
        self._dma_busy_until: int = 0

    def add_mxu(self, op: str, cycles: int, layer: int) -> TimelineEvent:
        """Schedule a matrix multiply operation on the MXU.

        MXU events advance the timeline and set the ``_mxu_busy_until``
        watermark that DMA/NoC events use for overlap detection.
        """
        start = self._current_cycle
        end = start + cycles
        self._mxu_busy_until = max(self._mxu_busy_until, end)
        self._current_cycle = end
        ev = TimelineEvent("mxu", op, start, end, layer)
        self.events.append(ev)
        return ev

    def add_sfu(self, op: str, cycles: int, layer: int) -> TimelineEvent:
        """Schedule a scalar function unit operation.

        SFU runs after MXU for the current layer (data dependency:
        softmax/activation requires MXU output).
        """
        # SFU runs after MXU for current layer (data dependency)
        start = self._current_cycle
        end = start + cycles
        self._current_cycle = end
        ev = TimelineEvent("sfu", op, start, end, layer)
        self.events.append(ev)
        return ev

    def add_vector(self, op: str, cycles: int, layer: int) -> TimelineEvent:
        """Vector unit: can overlap with SFU (separate datapath)."""
        start = self._current_cycle
        end = start + cycles
        self._current_cycle = end
        ev = TimelineEvent("vector", op, start, end, layer)
        self.events.append(ev)
        return ev

    def add_dma_parallel(self, op: str, cycles: int, layer: int) -> TimelineEvent:
        """DMA that can overlap with MXU: starts now, may extend beyond MXU."""
        start = self._current_cycle
        end = start + cycles
        overlapped = cycles <= (self._mxu_busy_until - start)
        ev = TimelineEvent("dma", op, start, end, layer, overlapped=overlapped)
        self.events.append(ev)
        # Only advance timeline if DMA extends beyond current mxu
        if end > self._current_cycle:
            self._current_cycle = end
        return ev

    def add_noc(self, op: str, cycles: int, layer: int) -> TimelineEvent:
        """NoC transfer that can overlap with MXU: starts now, may extend beyond MXU."""
        start = self._current_cycle
        end = start + cycles
        overlapped = cycles <= (self._mxu_busy_until - start)
        ev = TimelineEvent("noc", op, start, end, layer, overlapped=overlapped)
        self.events.append(ev)
        if end > self._current_cycle:
            self._current_cycle = end
        return ev

    def add_kv(self, op: str, cycles: int, layer: int) -> TimelineEvent:
        """Schedule a KV cache access operation.

        KV cache accesses (layer switches, context reads) advance the
        timeline and are serialised with compute (no overlap).
        """
        start = self._current_cycle
        end = start + cycles
        self._current_cycle = end
        ev = TimelineEvent("kv", op, start, end, layer)
        self.events.append(ev)
        return ev

    @property
    def total_cycles(self) -> int:
        return self._current_cycle


def breakdown_events(events: List[TimelineEvent]) -> Dict[str, float]:
    """Aggregate events by module, counting only effective (non-overlapped) cycles."""
    modules: Dict[str, float] = {}
    for ev in events:
        cycles = ev.end_cycle - ev.start_cycle
        if ev.module == "dma" and ev.overlapped:
            key = "DMA (hidden)"
            # Hidden DMA doesn't count toward total
            modules[key] = modules.get(key, 0) + cycles
        elif ev.module == "noc" and ev.overlapped:
            key = "noc_latency"
            modules[key] = modules.get(key, 0) + cycles
        else:
            key = {
                "mxu": "MXU",
                "sfu": "SFU",
                "vector": "Vector",
                "dma": "DMA (stall)",
                "kv": "KV Cache",
                "riscv": "RISC-V",
                "noc": "noc_latency",
            }.get(ev.module, ev.module)
            modules[key] = modules.get(key, 0) + cycles
    modules.setdefault("noc_contention", 0.0)
    return modules
