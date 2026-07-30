"""PPA 模型 — 面积/功耗/性能 综合评估"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts.bitcell import BitcellTable, sram_area_mm2
from models.memory_backend import (
    MemoryAccessPattern,
    MemoryBackend,
    MemoryRequest,
    MemoryTopology,
)
from models.onchip_dram import Parametric3DMemoryBackend


@dataclass
class PPA:
    """Performance, Power, Area"""

    tok_s: float
    area_mm2: float
    power_w: float
    efficiency_tok_per_watt: float = 0.0
    efficiency_tok_per_mm2: float = 0.0
    config_label: str = ""
    sram_spill_mb: float = 0.0
    depthwise_util_pct: float = 0.0

    def __post_init__(self):
        self.efficiency_tok_per_watt = self.tok_s / max(self.power_w, 0.1)
        self.efficiency_tok_per_mm2 = self.tok_s / max(self.area_mm2, 0.1)

    def __repr__(self):
        return (
            f"PPA(tok={self.tok_s:.0f}/s, {self.area_mm2:.0f}mm², "
            f"{self.power_w:.1f}W, {self.efficiency_tok_per_watt:.1f}tok/W)"
        )


def _node_scale_factor(process_node_nm: float) -> float:
    """Return area scale factor relative to the 7nm baseline.

    Uses the TSMC density ratio correction: 12FFC is an optical shrink of
    16FFC, not true 12nm geometry.  The density ratio 91.2/33.8 = 2.70×
    replaces the old geometric scaling (12/7)² = 2.94×.
    """
    if process_node_nm == 12.0:
        return 2.70
    # Fall back to conventional geometric scaling for other nodes; 7nm = 1.0.
    return (process_node_nm / 7.0) ** 2


class AreaModel:
    """面积估算模型 — 基于配置参数。

    PE 面积基线来自公开论文/产品数据，详见 references/area_sources.md。
    所有基线以 7nm 为参考节点，运行时按密度比缩放（12nm 使用 2.70×，
    不是旧版的 (12/7)² = 2.94×）。
    """

    def __init__(self, config: dict[str, Any]):
        am = config.get("area_model", {})
        self.process_node_nm = float(am.get("process_node_nm", am.get("process_node", 7.0)))
        self.node_scale = _node_scale_factor(self.process_node_nm)

        # ── PE 面积基线 @7nm (128×128 array) ──
        # 来源: TPUv1 ISCA 2017 die-shot 反推，见 references/area_sources.md
        self.systolic_pe_baseline = float(am.get("systolic_pe_area_mm2", 2.0)) * self.node_scale
        self.block_pe_baseline = (
            float(am.get("block_pe_area_mm2", 4.0)) * self.node_scale
        )  # 2× systolic (local acc + broadcast)
        self.os_pe_baseline = float(am.get("os_pe_area_mm2", 4.0)) * self.node_scale  # output stationary ≈ block
        self.input_stationary_pe_baseline = (
            float(am.get("is_pe_area_mm2", 4.0)) * self.node_scale
        )  # input stationary ≈ block
        self.tensor_core_pe_baseline = float(am.get("tc_pe_area_mm2", 4.0)) * self.node_scale  # TC ≈ block
        self.wmma_pe_baseline = (
            float(am.get("wmma_pe_area_mm2", 6.0)) * self.node_scale
        )  # ~1.5× block (warp-level control)
        self.gmma_pe_baseline = (
            float(am.get("gmma_pe_area_mm2", 7.0)) * self.node_scale
        )  # ~1.75× block (async copy + TMA)
        self.fsa_pe_baseline = (
            float(am.get("fsa_pe_area_mm2", 2.2)) * self.node_scale
        )  # 1.1× systolic (CMP + Split overhead)
        self.sfu = float(am.get("sfu_area_mm2", 1.5)) * self.node_scale

        # SRAM area is now derived from the foundry bitcell table, not from
        # fixed mm²/KB constants.  l1_per_kb / l2_per_kb are kept as legacy
        # fallback values (unchanged, NOT node-scaled) for backward-compatible
        # config parsing and any downstream code that still reads them.
        # DEPRECATED: do not use for new area calculations.
        self.l1_per_kb = float(am.get("l1_sram_per_kb_mm2", 0.002))
        self.l2_per_kb = float(am.get("l2_sram_per_kb_mm2", 0.0015))
        self.l1_overhead = float(am.get("l1_overhead", 1.5))
        self.l2_overhead = float(am.get("l2_overhead", 1.3))
        self._bitcell_table = BitcellTable()

        self.dma = float(am.get("dma_area_mm2", 1.0)) * self.node_scale
        self.riscv = float(am.get("riscv_area_mm2", 1.0)) * self.node_scale
        self.pcie = float(am.get("pcie_area_mm2", 2.0)) * self.node_scale
        self.dram_phy = float(am.get("dram_phy_area_mm2", 5.0)) * self.node_scale
        self.crossbar = float(am.get("crossbar_area_mm2", 1.0)) * self.node_scale
        self.dma_per_ch = float(am.get("dma_channels_area_per_channel_mm2", 0.5)) * self.node_scale

        # Legacy fixed-percentage TSV overhead is replaced by the parametric
        # memory backend below.  Keep the constant for backward-compatible
        # config parsing only.
        self.tsv_overhead_pct = float(am.get("tsv_overhead_pct", 0.10))

        # CV-specific hardware units
        self.im2col_feeder = float(am.get("im2col_feeder_mm2", 0.002))  # scales with array
        self.pool2d = float(am.get("pool2d_mm2", 0.05))  # fixed cost
        self.conv_sfu = float(am.get("conv_sfu_mm2", 0.10))  # fixed cost

        # Parametric memory backend for memory-dependent area/power.
        self._memory_backend: MemoryBackend = Parametric3DMemoryBackend()

    def _memory_type(self, config: dict[str, Any]) -> str:
        """Classify memory subsystem as on_chip_3d_dram, hbm, or lpddr."""
        onchip = config.get("on_chip_memory", {})
        if float(onchip.get("capacity_gb", 0)) > 0:
            return "on_chip_3d_dram"
        mem_type = str(config.get("memory", {}).get("type", "LPDDR5-6400")).lower()
        if "hbm3" in mem_type:
            return "hbm3"
        if "hbm2e" in mem_type or "hbm2" in mem_type:
            return "hbm2e"
        if "lpddr5x" in mem_type:
            return "lpddr5x"
        return "lpddr5"

    def _memory_area_estimate(self, config: dict[str, Any]) -> dict[str, float]:
        """Estimate memory-die, PHY, TSV, and package area using the backend.

        Returns a dict with memory_die_area_mm2, interface_area_mm2,
        dram_phy_area_mm2, package_area_mm2, tsv_area_mm2.
        """
        mem_type = self._memory_type(config)
        onchip = config.get("on_chip_memory", {})
        memory = config.get("memory", {})

        if mem_type == "on_chip_3d_dram":
            capacity_gb = float(onchip.get("capacity_gb", 0))
            bandwidth_gbps = float(onchip.get("bandwidth_gbps", 500.0))
            include_phy = False
            include_tsv = True
        else:
            capacity_gb = float(memory.get("capacity_gb", 32.0))
            bandwidth_gbps = float(memory.get("bandwidth_gbps", 51.2))
            include_phy = True
            include_tsv = mem_type in {"hbm2e", "hbm3"}

        topology = MemoryTopology(
            tier=mem_type,  # type: ignore[arg-type]
            process_node_nm=12.0,
            include_phy=include_phy,
            include_tsv=include_tsv,
            include_package=True,
        )
        request = MemoryRequest(
            topology=topology,
            capacity_gb=capacity_gb,
            bandwidth_gbps=bandwidth_gbps,
            access=MemoryAccessPattern(
                read_bytes=0,
                write_bytes=0,
                active_time_seconds=1e-6,
            ),
        )
        response = self._memory_backend.estimate(request)
        return {
            "memory_die_area_mm2": response.memory_die_area_mm2,
            "interface_area_mm2": response.interface_area_mm2,
            "dram_phy_area_mm2": response.components.get("phy_area_mm2", 0.0),
            "package_area_mm2": response.components.get("package_area_mm2", 0.0),
            "tsv_area_mm2": response.components.get("tsv_area_mm2", 0.0),
        }

    def estimate(self, config: dict[str, Any], engine_type: str) -> dict[str, float]:
        """估算总面积"""
        mac = config.get("mac_engine", {})
        H = int(mac.get("array_height", 128))
        W = int(mac.get("array_width", 128))
        scale = (H * W) / (128 * 128)

        # PE array
        engine_area_map = {
            "systolic": self.systolic_pe_baseline,
            "os_systolic": self.os_pe_baseline,
            "block": self.block_pe_baseline,
            "tensor_core": self.tensor_core_pe_baseline,
            "wmma": self.wmma_pe_baseline,
            "gmma": self.gmma_pe_baseline,
            "input_stationary": self.input_stationary_pe_baseline,
            "fsa": self.fsa_pe_baseline,
        }
        pe_base = engine_area_map.get(engine_type, self.block_pe_baseline)
        pe_area = pe_base * scale

        # SRAM — derived from foundry bitcell table, not legacy mm²/KB constants.
        sram = config.get("sram", {})
        l1_kb = float(sram.get("l1_per_core_kb", 512))
        l2_kb = float(sram.get("l2_shared_kb", 2048))
        l1 = sram_area_mm2(
            size_bytes=int(l1_kb * 1024),
            node_nm=self.process_node_nm,
            overhead=self.l1_overhead,
            table=self._bitcell_table,
        )
        l2 = sram_area_mm2(
            size_bytes=int(l2_kb * 1024),
            node_nm=self.process_node_nm,
            overhead=self.l2_overhead,
            table=self._bitcell_table,
        )

        # DMA channels
        dma_cfg = config.get("dma", {})
        dma_channels = int(dma_cfg.get("channels", 2))
        opts = config.get("optimizations", {})
        if float(opts.get("dma_bw_multiplier", 1.0)) >= 2.0:
            # 128-bit DRAM or 4ch DMA
            dma_channels = max(dma_channels, 4)

        dma_area = self.dma + (dma_channels - 2) * self.dma_per_ch

        # PCIe is always present (host interface).
        pcie_area = self.pcie

        # Memory subsystem area: on-chip 3D DRAM has no external PHY;
        # HBM keeps PHY + TSV + package; LPDDR keeps PHY + package, no TSV.
        memory = self._memory_area_estimate(config)
        memory_die_area = memory["memory_die_area_mm2"]
        dram_phy_area = memory["dram_phy_area_mm2"]
        tsv_area = memory["tsv_area_mm2"]
        package_area = memory["package_area_mm2"]

        # CV hardware units
        im2col_feeder_area = self.im2col_feeder * scale  # scales with array size
        pool2d_area = self.pool2d
        conv_sfu_area = self.conv_sfu

        total = (
            pe_area
            + self.sfu
            + self.riscv
            + pcie_area
            + self.crossbar
            + l1
            + l2
            + dma_area
            + dram_phy_area
            + memory_die_area
            + tsv_area
            + package_area
            + im2col_feeder_area
            + pool2d_area
            + conv_sfu_area
        )

        return {
            "total_mm2": round(total, 1),
            "memory_die_area_mm2": memory_die_area,
            "dram_phy_area_mm2": dram_phy_area,
            "tsv_area_mm2": tsv_area,
            "package_area_mm2": package_area,
            "im2col_feeder_mm2": im2col_feeder_area,
            "pool2d_mm2": pool2d_area,
            "conv_sfu_mm2": conv_sfu_area,
        }


class PowerModel:
    """功耗估算模型 — 粗略 but proportional"""

    def __init__(self, config: dict[str, Any]):
        # 12nm: ~0.5 W/mm² for logic, ~0.1 W/mm² for SRAM (active)
        self.logic_power_density = 0.5  # W/mm²
        self.sram_power_density = 0.1  # W/mm²
        self.dram_phy_power = 3.0  # W (fixed overhead)
        self._memory_backend: MemoryBackend = Parametric3DMemoryBackend()

    def _memory_type(self, config: dict[str, Any]) -> str:
        onchip = config.get("on_chip_memory", {})
        if float(onchip.get("capacity_gb", 0)) > 0:
            return "on_chip_3d_dram"
        mem_type = str(config.get("memory", {}).get("type", "LPDDR5-6400")).lower()
        if "hbm3" in mem_type:
            return "hbm3"
        if "hbm2e" in mem_type or "hbm2" in mem_type:
            return "hbm2e"
        if "lpddr5x" in mem_type:
            return "lpddr5x"
        return "lpddr5"

    def _memory_power_estimate(self, area_model: AreaModel, config: dict[str, Any]) -> float:
        """Return memory-related static + active power proxy."""
        mem_type = self._memory_type(config)
        onchip = config.get("on_chip_memory", {})
        memory = config.get("memory", {})

        if mem_type == "on_chip_3d_dram":
            capacity_gb = float(onchip.get("capacity_gb", 0))
            bandwidth_gbps = float(onchip.get("bandwidth_gbps", 500.0))
            include_phy = False
            include_tsv = True
        else:
            capacity_gb = float(memory.get("capacity_gb", 32.0))
            bandwidth_gbps = float(memory.get("bandwidth_gbps", 51.2))
            include_phy = True
            include_tsv = mem_type in {"hbm2e", "hbm3"}

        topology = MemoryTopology(
            tier=mem_type,  # type: ignore[arg-type]
            process_node_nm=12.0,
            include_phy=include_phy,
            include_tsv=include_tsv,
            include_package=True,
        )
        # Use a nominal 1% bandwidth utilization for the power proxy.
        bytes_per_s = bandwidth_gbps * 1e9 * 0.01
        active_time = 1.0
        request = MemoryRequest(
            topology=topology,
            capacity_gb=capacity_gb,
            bandwidth_gbps=bandwidth_gbps,
            access=MemoryAccessPattern(
                read_bytes=int(bytes_per_s * active_time * 0.5),
                write_bytes=int(bytes_per_s * active_time * 0.5),
                active_time_seconds=active_time,
            ),
        )
        response = self._memory_backend.estimate(request)
        return response.static_power_w + response.active_power_w

    def estimate(self, area_model: AreaModel, config: dict[str, Any], engine_type: str) -> float:
        """粗略功耗估算"""
        mac = config.get("mac_engine", {})
        H = int(mac.get("array_height", 128))
        W = int(mac.get("array_width", 128))
        scale = (H * W) / (128 * 128)
        freq_scale = float(mac.get("frequency_mhz", 1000)) / 1000

        # Logic power
        engine_area_map = {
            "systolic": area_model.systolic_pe_baseline,
            "os_systolic": area_model.os_pe_baseline,
            "block": area_model.block_pe_baseline,
            "tensor_core": area_model.tensor_core_pe_baseline,
            "wmma": area_model.wmma_pe_baseline,
            "gmma": area_model.gmma_pe_baseline,
            "input_stationary": area_model.input_stationary_pe_baseline,
            "fsa": area_model.fsa_pe_baseline,
        }
        pe_base = engine_area_map.get(engine_type, area_model.block_pe_baseline)
        logic_mm2 = pe_base * scale + area_model.sfu

        logic_power = logic_mm2 * self.logic_power_density * freq_scale

        # SRAM power — use the same bitcell-derived area as AreaModel.
        sram = config.get("sram", {})
        l1_kb = float(sram.get("l1_per_core_kb", 512))
        l2_kb = float(sram.get("l2_shared_kb", 2048))
        sram_mm2 = sram_area_mm2(
            size_bytes=int(l1_kb * 1024),
            node_nm=area_model.process_node_nm,
            overhead=area_model.l1_overhead,
            table=area_model._bitcell_table,
        ) + sram_area_mm2(
            size_bytes=int(l2_kb * 1024),
            node_nm=area_model.process_node_nm,
            overhead=area_model.l2_overhead,
            table=area_model._bitcell_table,
        )
        sram_power = sram_mm2 * self.sram_power_density

        # DRAM PHY fixed overhead for external memory; on-chip has no PHY.
        mem_type = self._memory_type(config)
        dram_phy_power = 0.0 if mem_type == "on_chip_3d_dram" else self.dram_phy_power

        # DRAM bandwidth proportional power via parametric backend proxy.
        mem_power = self._memory_power_estimate(area_model, config)

        # CV unit power
        cv_area = area_model.estimate(config, engine_type)
        im2col_power = cv_area["im2col_feeder_mm2"] * 0.1  # SRAM-dense logic
        pool2d_power = cv_area["pool2d_mm2"] * 0.5  # combinational logic
        conv_sfu_power = cv_area["conv_sfu_mm2"] * 0.3  # LUT + control

        total = (
            logic_power
            + sram_power
            + dram_phy_power
            + mem_power
            + 2.0  # +2W misc
            + im2col_power
            + pool2d_power
            + conv_sfu_power
        )
        return round(total, 1)
