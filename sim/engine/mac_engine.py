"""MAC Engine 抽象接口 — 可插拔的矩阵乘法引擎

支持:
  - Systolic: weight-stationary, 时空映射，有 pipeline fill/drain
  - Block: 全并行 MAC 阵列，纯空间映射，无 pipeline overhead

所有引擎实现统一的 estimate(M,K,N) → EngineResult 接口。
"""

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from contracts.units import bandwidth_gbps_to_bytes_per_cycle as _bw2bpc
from models.memory_backend import AccessType
from models.memory_hierarchy import build_hierarchy_from_config
from models.residency import MemoryAccessPlan, build_memory_access_plan
from workloads.schema import WorkloadGraphV1


@dataclass
class EngineResult:
    """统一引擎性能结果 — v2 contract.

    Required fields (non-negotiable):
      mac_count           — M × K × N multiply-accumulates
      op_count            — 2 × mac_count
      ideal_compute_cycles — ceil(mac_count / peak_macs_per_cycle)
      raw_dma_cycles       — ceil(raw_transfer_bytes / eff_bytes_per_cycle)

    ``ops`` is a deprecated read-only alias for ``mac_count``.
    """

    compute_cycles: int
    dma_cycles: int  # DRAM ↔ SRAM 数据传输
    total_cycles: int
    utilization: float  # 理论峰值利用率
    mac_count: int  # M × K × N multiply-accumulates
    op_count: int  # 2 × mac_count
    num_tiles: int = 0
    weight_bytes: int = 0
    bottleneck: str = ""  # "compute" | "dma"
    details: dict[str, Any] = field(default_factory=dict)
    stall_cycles_dram: int = 0
    stall_cycles_sram: int = 0
    ideal_compute_cycles: int = 0
    raw_dma_cycles: int = 0

    def __post_init__(self) -> None:
        """Validate finite values, non-negative cycles, positive counts."""
        _validate_finite(self.mac_count, "mac_count")
        _validate_finite(self.op_count, "op_count")
        _validate_finite(self.ideal_compute_cycles, "ideal_compute_cycles")
        _validate_finite(self.raw_dma_cycles, "raw_dma_cycles")
        _validate_finite(self.total_cycles, "total_cycles")
        _validate_finite(self.compute_cycles, "compute_cycles")
        _validate_finite(self.dma_cycles, "dma_cycles")
        _validate_finite(self.utilization, "utilization")
        _validate_finite(self.weight_bytes, "weight_bytes")

        if self.total_cycles < 0:
            raise ValueError(f"total_cycles must be non-negative, got {self.total_cycles}")
        if self.compute_cycles < 0:
            raise ValueError(f"compute_cycles must be non-negative, got {self.compute_cycles}")
        if self.dma_cycles < 0:
            raise ValueError(f"dma_cycles must be non-negative, got {self.dma_cycles}")
        if self.ideal_compute_cycles < 0:
            raise ValueError(f"ideal_compute_cycles must be non-negative, got {self.ideal_compute_cycles}")
        if self.raw_dma_cycles < 0:
            raise ValueError(f"raw_dma_cycles must be non-negative, got {self.raw_dma_cycles}")
        if self.mac_count <= 0:
            raise ValueError(f"mac_count must be positive, got {self.mac_count}")
        if self.op_count <= 0:
            raise ValueError(f"op_count must be positive, got {self.op_count}")

    @property
    def ops(self) -> int:
        """Deprecated: use ``mac_count`` instead.

        Returns the MAC count (M × K × N) for backward compatibility with
        legacy consumers.  Use ``mac_count`` in new code.
        """
        warnings.warn(
            "EngineResult.ops is deprecated; use mac_count instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mac_count

    def __repr__(self):
        return (
            f"Engine(total={self.total_cycles}c, "
            f"compute={self.compute_cycles}c, dma={self.dma_cycles}c, "
            f"util={self.utilization:.1%}, tiles={self.num_tiles}, "
            f"mac_count={self.mac_count}, "
            f"bottleneck={self.bottleneck}, "
            f"stall_dram={self.stall_cycles_dram}, stall_sram={self.stall_cycles_sram})"
        )


def _validate_finite(value: float, name: str) -> None:
    """Reject NaN and Inf values in result fields."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


class MACEngine(ABC):
    """MAC 引擎抽象基类

    Weight DMA follows a sequential access pattern (AccessType.SEQUENTIAL):
    weights are loaded in contiguous bulk transfers with page locality.
    KV cache reads are random (AccessType.RANDOM) due to scattered token
    positions causing row-buffer conflicts.
    """

    def __init__(
        self,
        config: dict[str, Any],
        graph: WorkloadGraphV1 | None = None,
        memory_access_plan: "MemoryAccessPlan | None" = None,
    ):
        self.config = config
        self._parse_config(config)

        if memory_access_plan is not None:
            self.memory_access_plan = memory_access_plan
        elif graph is not None:
            hierarchy = build_hierarchy_from_config(config)
            self.memory_access_plan = build_memory_access_plan(graph, hierarchy)
        else:
            hierarchy = build_hierarchy_from_config(config)
            self.memory_access_plan = build_memory_access_plan(WorkloadGraphV1(), hierarchy)

        self._apply_memory_plan()

    def _parse_config(self, config: dict[str, Any]):
        """解析公共配置参数"""
        mac = config.get("mac_engine", config.get("mxu", {}))
        self.H = int(mac.get("array_height", 128))
        self.W = int(mac.get("array_width", 128))
        self.f_mhz = int(mac.get("frequency_mhz", 1000))
        self.w_bits = int(mac.get("weight_precision_bits", 4))
        self.a_bits = int(mac.get("activation_precision_bits", 8))
        self.ops_per_mac = int(mac.get("ops_per_mac", 2))

        mem = config.get("memory", {})
        bw_gbps = float(mem.get("bandwidth_gbps", 51.2))
        self.bw_raw = _bw2bpc(bw_gbps, self.f_mhz)
        self.dram_efficiency = float(mem.get("dram_efficiency", 0.90))
        self.dram_efficiency_random_bw = float(mem.get("dram_efficiency_random_bw", 0.50))
        self.random_latency_penalty_cycles = int(mem.get("random_latency_penalty_cycles", 40))
        opts = config.get("optimizations", {})
        self.bw_multiplier = float(opts.get("dma_bw_multiplier", 1.0))

        self.eff_bw_weight = self.bw_raw * self.dram_efficiency * self.bw_multiplier
        self.eff_bw_kv = self.bw_raw * self.dram_efficiency_random_bw * self.bw_multiplier
        # Deprecated alias retained for callers that have not migrated to access_type.
        self.eff_bw = self.eff_bw_weight

        # SRAM: 60% weight buffer, 40% KV tile buffer
        sram = config.get("sram", {})
        self.l2_sram_kb = int(sram.get("l2_shared_kb", 2048))
        self.wbuf_kb = int(self.l2_sram_kb * 0.6)
        self.kvbuf_kb = int(self.l2_sram_kb * 0.4)

        # On-chip 3D DRAM (RK1828-style): weights resident on-die
        onchip = config.get("on_chip_memory", {})
        self.on_chip_capacity_gb = float(onchip.get("capacity_gb", 0))
        self.on_chip_bw = float(onchip.get("bandwidth_gbps", 0))  # GB/s

    def _apply_memory_plan(self) -> None:
        """Consume ``self.memory_access_plan`` to set effective bandwidth.

        If all weights are resident in the fastest eligible tier (no spill),
        weight transfers use that tier's bandwidth.  Otherwise the external
        DRAM bandwidth is used.  This replaces the old positive predicate
        ``weight_resident`` and ensures monotonic behavior as capacity shrinks.
        """
        plan = self.memory_access_plan
        if plan is None:
            return

        weight = plan.placements_for_category("weight")
        total_weight_spill = sum(p.spill_bytes for p in weight)
        if total_weight_spill == 0:
            fastest = plan.fastest_allocated_tier("weight")
            if fastest is not None:
                tier = plan.hierarchy.get_tier(fastest)
                self.eff_bw_weight = _bw2bpc(tier.effective_read_bw_gbps(), self.f_mhz)
                self.eff_bw = self.eff_bw_weight
                return

        self.eff_bw_weight = self.bw_raw * self.dram_efficiency * self.bw_multiplier
        self.eff_bw = self.eff_bw_weight

    def _dram_eff_for_bytes(self, transfer_bytes: int) -> float:
        """DRAM utilization factor for a given transfer size (sequential access).

        Weight DMA follows a sequential access pattern, benefiting from
        page locality and burst efficiency (cf. AccessType.SEQUENTIAL).

        - Small transfers (≤ wbuf) → cached, no DRAM needed
        - Large transfers → full DRAM read, efficiency depends on buffer ratio
        Returns 0.0 if fully cached, else [0.55, 0.92] efficiency factor.
        """
        if transfer_bytes <= 0:
            return 1.0
        wbuf_mb = self.wbuf_kb / 1024.0
        weight_mb = transfer_bytes / (1024 * 1024.0)
        if weight_mb <= wbuf_mb:
            return 0.0  # cached — caller should skip DMA
        ratio = wbuf_mb / weight_mb
        return 0.55 + 0.40 * ratio / (0.3 + ratio)

    def _kv_dram_efficiency(self, kv_bytes: int) -> float:
        """DRAM efficiency for KV cache reads (random access; cf. AccessType.RANDOM)."""
        if kv_bytes <= 0:
            return 1.0
        kvbuf_mb = self.kvbuf_kb / 1024.0
        kv_mb = kv_bytes / (1024 * 1024.0)
        ratio = kvbuf_mb / max(kv_mb, 0.001)
        return 0.55 + 0.40 * ratio / (0.3 + ratio)

    def _kv_eff_bw(self, kv_bytes: int) -> float:
        """Effective bytes/cycle for random KV reads including _kv_dram_efficiency."""
        return self.eff_bw_kv * self._kv_dram_efficiency(kv_bytes)

    def _dma_cycles(
        self,
        size_bytes: int,
        access_type: AccessType,
        *,
        kv_bytes: int = 0,
        is_hit: bool = False,
    ) -> float:
        """DMA cycles for a transfer using pattern-aware effective bandwidth.

        Sequential accesses (weights/activations) use eff_bw_weight.
        Random accesses (KV cache) use eff_bw_kv * _kv_dram_efficiency(kv_bytes)
        and add random_latency_penalty_cycles only on miss.
        """
        if size_bytes <= 0:
            return 0.0
        if access_type == AccessType.RANDOM:
            eff_bw = self._kv_eff_bw(kv_bytes)
            penalty = 0 if is_hit else self.random_latency_penalty_cycles
            return size_bytes / eff_bw + penalty if eff_bw > 0 else 0.0
        return size_bytes / self.eff_bw_weight if self.eff_bw_weight > 0 else 0.0

    @property
    def peak_macs_per_cycle(self) -> float:
        """理论峰值 MAC/cycle"""
        return self.H * self.W * self.ops_per_mac

    @abstractmethod
    def estimate(self, M: int, K: int, N: int, weight_preloaded: bool = False) -> EngineResult:
        """估算 (M×K) × (K×N) 矩阵乘法的 cycle 数"""
        ...

    @abstractmethod
    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """估算 gate+up 权重缓存合并的 cycle 数"""
        ...

    @property
    @abstractmethod
    def engine_type(self) -> str:
        """引擎类型标识"""
        ...


def create_engine(
    config: dict[str, Any],
    graph: WorkloadGraphV1 | None = None,
    memory_access_plan: "MemoryAccessPlan | None" = None,
) -> MACEngine:
    """工厂函数：根据配置创建引擎实例.

    Uses the unified engine registry as the single source of truth.
    Raises ``ConfigError`` for unknown engine types.
    """
    from engine.registry import create_engine_by_type

    mac = config.get("mac_engine", config.get("mxu", {}))
    engine_type = mac.get("type", "block")
    return create_engine_by_type(engine_type, config, graph=graph, memory_access_plan=memory_access_plan)
