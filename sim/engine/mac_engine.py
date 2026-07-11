"""MAC Engine 抽象接口 — 可插拔的矩阵乘法引擎

支持:
  - Systolic: weight-stationary, 时空映射，有 pipeline fill/drain
  - Block: 全并行 MAC 阵列，纯空间映射，无 pipeline overhead

所有引擎实现统一的 estimate(M,K,N) → EngineResult 接口。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class EngineResult:
    """统一引擎性能结果"""
    compute_cycles: int
    dma_cycles: int           # DRAM ↔ SRAM 数据传输
    total_cycles: int
    utilization: float        # 理论峰值利用率
    ops: int                  # 总 MAC 操作数
    num_tiles: int = 0
    weight_bytes: int = 0
    bottleneck: str = ""      # "compute" | "dma"
    details: Dict[str, Any] = field(default_factory=dict)
    stall_cycles_dram: int = 0
    stall_cycles_sram: int = 0

    def __repr__(self):
        return (f"Engine(total={self.total_cycles}c, "
                f"compute={self.compute_cycles}c, dma={self.dma_cycles}c, "
                f"util={self.utilization:.1%}, tiles={self.num_tiles}, "
                f"bottleneck={self.bottleneck}, "
                f"stall_dram={self.stall_cycles_dram}, stall_sram={self.stall_cycles_sram})")


class MACEngine(ABC):
    """MAC 引擎抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._parse_config(config)

    def _parse_config(self, config: Dict[str, Any]):
        """解析公共配置参数"""
        mac = config.get("mac_engine", config.get("mxu", {}))
        self.H = int(mac.get("array_height", 128))
        self.W = int(mac.get("array_width", 128))
        self.f_mhz = int(mac.get("frequency_mhz", 1000))
        self.w_bits = int(mac.get("weight_precision_bits", 4))
        self.a_bits = int(mac.get("activation_precision_bits", 8))
        self.ops_per_mac = int(mac.get("ops_per_mac", 2))

        mem = config.get("memory", {})
        self.bw_raw = float(mem.get("bandwidth_bytes_per_cycle", 51.2))
        self.dram_efficiency = float(mem.get("dram_efficiency", 0.85))
        opts = config.get("optimizations", {})
        self.bw_multiplier = float(opts.get("dma_bw_multiplier", 1.0))
        self.eff_bw = self.bw_raw * self.dram_efficiency * self.bw_multiplier

        # SRAM: 60% weight buffer, 40% KV tile buffer
        sram = config.get("sram", {})
        self.l2_sram_kb = int(sram.get("l2_shared_kb", 2048))
        self.wbuf_kb = int(self.l2_sram_kb * 0.6)
        self.kvbuf_kb = int(self.l2_sram_kb * 0.4)

        # On-chip 3D DRAM (RK1828-style): weights resident on-die
        onchip = config.get("on_chip_memory", {})
        self.on_chip_capacity_gb = float(onchip.get("capacity_gb", 0))
        self.on_chip_bw = float(onchip.get("bandwidth_gbps", 0))  # GB/s

    @property
    def weight_resident(self) -> bool:
        """True if all model weights fit in on-chip memory."""
        return self.on_chip_capacity_gb > 0 and self.on_chip_bw > 0

    def _dram_eff_for_bytes(self, transfer_bytes: int) -> float:
        """DRAM utilization factor for a given transfer size.

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
        """DRAM efficiency for KV cache reads (uses 40% SRAM buffer)."""
        if kv_bytes <= 0:
            return 1.0
        kvbuf_mb = self.kvbuf_kb / 1024.0
        kv_mb = kv_bytes / (1024 * 1024.0)
        ratio = kvbuf_mb / max(kv_mb, 0.001)
        return 0.55 + 0.40 * ratio / (0.3 + ratio)

    @property
    def peak_macs_per_cycle(self) -> float:
        """理论峰值 MAC/cycle"""
        return self.H * self.W * self.ops_per_mac

    @abstractmethod
    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
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



def create_engine(config: Dict[str, Any]) -> MACEngine:
    """工厂函数：根据配置创建引擎实例"""
    mac = config.get("mac_engine", config.get("mxu", {}))
    engine_type = mac.get("type", "block")

    if engine_type == "systolic":
        from engine.systolic_engine import SystolicEngine
        return SystolicEngine(config)
    elif engine_type == "os_systolic":
        from engine.os_systolic_engine import OutputStationaryEngine
        return OutputStationaryEngine(config)
    elif engine_type == "input_stationary":
        from engine.is_systolic_engine import InputStationaryEngine
        return InputStationaryEngine(config)
    elif engine_type == "tensor_core":
        from engine.tensor_core_engine import TensorCoreEngine
        return TensorCoreEngine(config)
    elif engine_type == "wmma":
        from engine.wmma_engine import WMMAEngine
        return WMMAEngine(config)
    elif engine_type == "gmma":
        from engine.gmma_engine import GMMAEngine
        return GMMAEngine(config)
    elif engine_type == "fsa":
        from engine.fsa_engine import FSAEngine
        return FSAEngine(config)
    elif engine_type == "block":
        from engine.block_engine import BlockEngine
        return BlockEngine(config)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")
