"""PPA 模型 — 面积/功耗/性能 综合评估"""

from typing import Any, Dict

from dse.types import DSEPoint


PPA = DSEPoint


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
        self.fsa_pe_baseline         = float(am.get("fsa_pe_area_mm2", 2.24)) * self.node_scale        # paper: +12% array only
        self.block_fused_pe_baseline = float(am.get("block_fused_attention_pe_area_mm2", 4.24)) * self.node_scale
        self.os_fused_pe_baseline    = float(am.get("os_fused_attention_pe_area_mm2", 4.24)) * self.node_scale
        # Weight Cache is a hardware variant, not a free performance switch.
        # These PE-array multipliers stay configurable until layout calibration.
        self.weight_cache_pe_overhead = {
            "systolic": float(am.get(
                "systolic_weight_cache_pe_overhead_pct", 0.15,
            )),
            "block": float(am.get(
                "block_weight_cache_pe_overhead_pct", 0.10,
            )),
            "block_fused_attention": float(am.get(
                "block_fused_attention_weight_cache_pe_overhead_pct", 0.10,
            )),
            "gmma": float(am.get(
                "gmma_weight_cache_pe_overhead_pct", 0.05,
            )),
        }
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

    def pe_area(self, config: Dict[str, Any], engine_type: str):
        """Return adjusted PE area and the explicit Weight Cache increment."""
        mac = config.get("mac_engine", {})
        height = int(mac.get("array_height", 128))
        width = int(mac.get("array_width", 128))
        scale = (height * width) / (128 * 128)
        engine_area_map = {
            "systolic": self.systolic_pe_baseline,
            "os_systolic": self.os_pe_baseline,
            "block": self.block_pe_baseline,
            "block_fused_attention": self.block_fused_pe_baseline,
            "os_systolic_fused_attention": self.os_fused_pe_baseline,
            "tensor_core": self.tensor_core_pe_baseline,
            "wmma": self.wmma_pe_baseline,
            "gmma": self.gmma_pe_baseline,
            "input_stationary": self.input_stationary_pe_baseline,
            "fsa": self.fsa_pe_baseline,
        }
        base_area = engine_area_map.get(engine_type, self.block_pe_baseline) * scale
        enabled = bool(
            config.get("optimizations", {}).get("weight_cache", False)
        )
        overhead_pct = (
            self.weight_cache_pe_overhead.get(engine_type, 0.0)
            if enabled else 0.0
        )
        weight_cache_area = base_area * overhead_pct
        return base_area + weight_cache_area, weight_cache_area, overhead_pct

    def estimate(self, config: Dict[str, Any], engine_type: str) -> float:
        """估算总面积"""
        mac = config.get("mac_engine", {})
        H = int(mac.get("array_height", 128))
        W = int(mac.get("array_width", 128))
        scale = (H * W) / (128 * 128)

        # PE array, including an explicit Weight Cache hardware increment.
        pe_area, weight_cache_area, weight_cache_overhead_pct = self.pe_area(
            config, engine_type,
        )

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

        if not bool(config.get("_cv_workload", False)):
            im2col_feeder_area = pool2d_area = conv_sfu_area = 0.0
        total = (pe_area + self.sfu + self.riscv + pcie_area +
                 self.crossbar + l1 + l2 + dma_area + dram_phy_area +
                 im2col_feeder_area + pool2d_area + conv_sfu_area)

        # TSV overhead for 3D-stacked memory
        if float(onchip.get("capacity_gb", 0)) > 0:
            total *= (1.0 + self.tsv_overhead_pct)

        logic_die_mm2 = round(total, 1)
        stack_area_mm2 = float(onchip.get("stack_area_mm2", 0.0))
        package_footprint_mm2 = max(logic_die_mm2, stack_area_mm2)
        return {
            "total_mm2": round(package_footprint_mm2, 1),
            "logic_die_mm2": logic_die_mm2,
            "memory_stack_mm2": round(stack_area_mm2, 1),
            "package_footprint_mm2": round(package_footprint_mm2, 1),
            "pe_array_mm2": round(pe_area, 4),
            "weight_cache_area_mm2": round(weight_cache_area, 4),
            "weight_cache_pe_overhead_pct": round(
                weight_cache_overhead_pct * 100.0, 2,
            ),
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

        # Logic power uses the same WC-adjusted PE area as the area model.
        pe_area, _, _ = area_model.pe_area(config, engine_type)
        logic_mm2 = pe_area + area_model.sfu

        logic_power = logic_mm2 * self.logic_power_density * freq_scale

        # SRAM power
        sram = config.get("sram", {})
        sram_kb = float(sram.get("l1_per_core_kb", 512)) + float(sram.get("l2_shared_kb", 2048))
        sram_mm2 = sram_kb * area_model.l1_per_kb  # rough
        sram_power = sram_mm2 * self.sram_power_density

        # External PHY and on-chip stack use different power models.
        mem = config.get("memory", {})
        onchip = config.get("on_chip_memory", {})
        bandwidth_gbps = float(mem.get("bandwidth_gbps", 51.2))
        if float(onchip.get("capacity_gb", 0.0)) > 0:
            memory_power = bandwidth_gbps * float(
                onchip.get("stack_power_per_gbps_w", 0.015))
        else:
            memory_power = self.dram_phy_power * (bandwidth_gbps / 51.2)

        # CV unit power
        cv_area = area_model.estimate(config, engine_type)
        im2col_power = cv_area["im2col_feeder_mm2"] * 0.1    # SRAM-dense logic
        pool2d_power = cv_area["pool2d_mm2"] * 0.5           # combinational logic
        conv_sfu_power = cv_area["conv_sfu_mm2"] * 0.3       # LUT + control

        total = (logic_power + sram_power + memory_power + 2.0  # +2W misc
                 + im2col_power + pool2d_power + conv_sfu_power)
        return round(total, 2)
