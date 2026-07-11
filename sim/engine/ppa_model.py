"""PPA 模型 — 面积/功耗/性能 综合评估"""

from dataclasses import dataclass, field
from typing import Any, Dict


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
        return (f"PPA(tok={self.tok_s:.0f}/s, {self.area_mm2:.0f}mm², "
                f"{self.power_w:.1f}W, {self.efficiency_tok_per_watt:.1f}tok/W)")


class AreaModel:
    """面积估算模型 — 基于配置参数。

    PE 面积基线来自公开论文/产品数据，详见 references/area_sources.md。
    所有基线以 7nm 为参考节点，运行时按 (process/7nm)^2 缩放。
    """

    def __init__(self, config: Dict[str, Any]):
        am = config.get("area_model", {})
        node = float(am.get("process_node_nm", am.get("process_node", 7.0)))
        self.node_scale = (node / 7.0) ** 2  # area scales with node^2

        # ── PE 面积基线 @7nm (128×128 array) ──
        # 来源: TPUv1 ISCA 2017 die-shot 反推，见 references/area_sources.md
        self.systolic_pe_baseline = float(am.get("systolic_pe_area_mm2", 2.0)) * self.node_scale
        self.block_pe_baseline       = float(am.get("block_pe_area_mm2", 4.0)) * self.node_scale       # 2× systolic (local acc + broadcast)
        self.os_pe_baseline          = float(am.get("os_pe_area_mm2", 4.0)) * self.node_scale          # output stationary ≈ block
        self.input_stationary_pe_baseline = float(am.get("is_pe_area_mm2", 4.0)) * self.node_scale     # input stationary ≈ block
        self.tensor_core_pe_baseline = float(am.get("tc_pe_area_mm2", 4.0)) * self.node_scale          # TC ≈ block
        self.wmma_pe_baseline        = float(am.get("wmma_pe_area_mm2", 6.0)) * self.node_scale        # ~1.5× block (warp-level control)
        self.gmma_pe_baseline        = float(am.get("gmma_pe_area_mm2", 7.0)) * self.node_scale        # ~1.75× block (async copy + TMA)
        self.fsa_pe_baseline         = float(am.get("fsa_pe_area_mm2", 2.2)) * self.node_scale         # 1.1× systolic (CMP + Split overhead)
        self.sfu = float(am.get("sfu_area_mm2", 1.5)) * self.node_scale
        self.l1_per_kb = float(am.get("l1_sram_per_kb_mm2", 0.002)) * self.node_scale
        self.l2_per_kb = float(am.get("l2_sram_per_kb_mm2", 0.0015)) * self.node_scale
        self.dma = float(am.get("dma_area_mm2", 1.0)) * self.node_scale
        self.riscv = float(am.get("riscv_area_mm2", 1.0)) * self.node_scale
        self.pcie = float(am.get("pcie_area_mm2", 2.0)) * self.node_scale
        self.dram_phy = float(am.get("dram_phy_area_mm2", 5.0)) * self.node_scale
        self.crossbar = float(am.get("crossbar_area_mm2", 1.0)) * self.node_scale
        self.dma_per_ch = float(am.get("dma_channels_area_per_channel_mm2", 0.5)) * self.node_scale
        # TSV area overhead for 3D-stacked DRAM (keep-out zones + SerDes + redundancy)
        # ~10% of total die for HBM2/3-class stacking at 500 GB/s, per industry rule-of-thumb
        self.tsv_overhead_pct = float(am.get("tsv_overhead_pct", 0.10))

        # CV-specific hardware units
        self.im2col_feeder = float(am.get("im2col_feeder_mm2", 0.002))   # scales with array
        self.pool2d = float(am.get("pool2d_mm2", 0.05))                   # fixed cost
        self.conv_sfu = float(am.get("conv_sfu_mm2", 0.10))               # fixed cost

    def estimate(self, config: Dict[str, Any], engine_type: str) -> float:
        """估算总面积"""
        mac = config.get("mac_engine", {})
        H = int(mac.get("array_height", 128))
        W = int(mac.get("array_width", 128))
        scale = (H * W) / (128 * 128)

        # PE array
        engine_area_map = {
            "systolic": self.systolic_pe_baseline,
            "os_systolic": self.block_pe_baseline,
            "block": self.block_pe_baseline,
            "tensor_core": self.tensor_core_pe_baseline,
            "wmma": self.wmma_pe_baseline,
            "gmma": self.gmma_pe_baseline,
            "input_stationary": self.input_stationary_pe_baseline,
            "fsa": self.fsa_pe_baseline,
        }
        pe_base = engine_area_map.get(engine_type, self.block_pe_baseline)
        pe_area = pe_base * scale

        # SRAM
        sram = config.get("sram", {})
        l1 = float(sram.get("l1_per_core_kb", 512)) * self.l1_per_kb
        l2 = float(sram.get("l2_shared_kb", 2048)) * self.l2_per_kb

        # DMA channels
        dma_cfg = config.get("dma", {})
        dma_channels = int(dma_cfg.get("channels", 2))
        opts = config.get("optimizations", {})
        if float(opts.get("dma_bw_multiplier", 1.0)) >= 2.0:
            # 128-bit DRAM or 4ch DMA
            dma_channels = max(dma_channels, 4)

        dma_area = self.dma + (dma_channels - 2) * self.dma_per_ch

        # DRAM PHY: skip if on-chip memory used (no external DDR interface)
        # PCIe: still needed even with on-chip memory (host communication)
        onchip = config.get("on_chip_memory", {})
        if float(onchip.get("capacity_gb", 0)) > 0:
            dram_phy_area = 0  # on-chip 3D DRAM doesn't need DDR PHY
            pcie_area = self.pcie  # host interface still required
        else:
            mem = config.get("memory", {})
            dram_width = int(mem.get("dram_width_bits", 64))
            dram_phy_area = self.dram_phy * (dram_width / 64)
            pcie_area = self.pcie

        # CV hardware units
        im2col_feeder_area = self.im2col_feeder * scale     # scales with array size
        pool2d_area = self.pool2d
        conv_sfu_area = self.conv_sfu

        total = (pe_area + self.sfu + self.riscv + pcie_area +
                 self.crossbar + l1 + l2 + dma_area + dram_phy_area +
                 im2col_feeder_area + pool2d_area + conv_sfu_area)

        # TSV overhead for 3D-stacked memory
        if float(onchip.get("capacity_gb", 0)) > 0:
            total *= (1.0 + self.tsv_overhead_pct)

        return {
            "total_mm2": round(total, 1),
            "im2col_feeder_mm2": im2col_feeder_area,
            "pool2d_mm2": pool2d_area,
            "conv_sfu_mm2": conv_sfu_area,
        }


class PowerModel:
    """功耗估算模型 — 粗略 but proportional"""

    def __init__(self, config: Dict[str, Any]):
        # 12nm: ~0.5 W/mm² for logic, ~0.1 W/mm² for SRAM (active)
        self.logic_power_density = 0.5   # W/mm²
        self.sram_power_density = 0.1    # W/mm²
        self.dram_phy_power = 3.0        # W (fixed overhead)

    def estimate(self, area_model: AreaModel, config: Dict[str, Any],
                 engine_type: str) -> float:
        """粗略功耗估算"""
        mac = config.get("mac_engine", {})
        H = int(mac.get("array_height", 128))
        W = int(mac.get("array_width", 128))
        scale = (H * W) / (128 * 128)
        freq_scale = float(mac.get("frequency_mhz", 1000)) / 1000

        # Logic power
        engine_area_map = {
            "systolic": area_model.systolic_pe_baseline,
            "os_systolic": area_model.block_pe_baseline,
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

        # SRAM power
        sram = config.get("sram", {})
        sram_kb = float(sram.get("l1_per_core_kb", 512)) + float(sram.get("l2_shared_kb", 2048))
        sram_mm2 = sram_kb * area_model.l1_per_kb  # rough
        sram_power = sram_mm2 * self.sram_power_density

        # DRAM bandwidth proportional power
        mem = config.get("memory", {})
        bw_ratio = float(mem.get("bandwidth_gbps", 51.2)) / 51.2
        dram_power = self.dram_phy_power * bw_ratio

        # CV unit power
        cv_area = area_model.estimate(config, engine_type)
        im2col_power = cv_area["im2col_feeder_mm2"] * 0.1    # SRAM-dense logic
        pool2d_power = cv_area["pool2d_mm2"] * 0.5           # combinational logic
        conv_sfu_power = cv_area["conv_sfu_mm2"] * 0.3       # LUT + control

        total = (logic_power + sram_power + dram_power + 2.0  # +2W misc
                 + im2col_power + pool2d_power + conv_sfu_power)
        return round(total, 1)
