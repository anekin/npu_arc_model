"""LPDDR5 DRAM 时序模型 — 刷新、行周期、带宽效率"""

from typing import Any, Dict


class DRAMModel:
    """LPDDR5-6400 timing model.

    Models DRAM refresh overhead and access efficiency.
    Key parameters from JEDEC LPDDR5 spec:
    - tRFC (refresh cycle time): ~210 ns for 8Gb density
    - tREFI (refresh interval): 3.9 μs (average, distributed)
    - tRC (row cycle time): ~48 ns
    - tRAS (row active time): ~42 ns

    At 1GHz: 1 cycle = 1 ns
    """

    def __init__(self, config: Dict[str, Any]):
        mem = config["memory"]
        self.f_mhz = float(config["mxu"]["frequency_mhz"])
        self.bw_gbps = float(mem["bandwidth_gbps"])

        # Convert ns to cycles at operating frequency
        ns_per_cycle = 1000.0 / self.f_mhz  # 1ns at 1GHz

        self.tRC_cycles = int(float(mem.get("tRC_cycles", 48)))     # 48
        self.tRAS_cycles = int(float(mem.get("tRAS_cycles", 42)))   # 42

        # LPDDR5 refresh: tRFC ≈ 210ns, tREFI ≈ 3900ns
        self.tRFC = 210 / ns_per_cycle     # refresh command time
        self.tREFI = 3900 / ns_per_cycle   # refresh interval
        self.refresh_overhead_pct = (self.tRFC / self.tREFI) * 100  # ~5.4%

        # Effective bandwidth: accounting for refresh + row conflicts
        # Row conflict probability depends on access pattern
        # Conservative: 85% bandwidth efficiency
        self.row_conflict_prob = 0.15  # 15% chance of row miss → need precharge+activate

    def effective_bandwidth_bytes_per_cycle(self) -> float:
        """Effective bandwidth after refresh + timing overhead.

        At 1GHz, LPDDR5-6400 = 51.2 GB/s raw.
        Effective = raw × (1 - refresh_pct) × row_efficiency
        """
        raw_bw = self.bw_gbps  # 51.2
        refresh_efficiency = 1.0 - (self.refresh_overhead_pct / 100)
        row_efficiency = 1.0 - (self.row_conflict_prob * 0.3)  # 30% penalty per conflict
        return raw_bw * refresh_efficiency * row_efficiency

    def add_refresh_overhead(self, total_cycles: int) -> int:
        """Add DRAM refresh cycles proportional to total compute time.

        Refresh happens in background during compute; only a fraction
        of refresh cycles actually stall the pipeline.
        """
        refresh_cycles = int(total_cycles * (self.refresh_overhead_pct / 100))
        return refresh_cycles

    def estimate_access_latency(self, size_bytes: int, is_read: bool = True) -> int:
        """Estimate DRAM access latency for a given transfer size.

        Includes: row activation + CAS + data burst + precharge.
        """
        if size_bytes <= 0:
            return 0

        burst_size = 256  # bytes per burst (32B × 8 beats on 32-bit bus)
        num_bursts = max(1, (size_bytes + burst_size - 1) // burst_size)

        # Base latency: one row activation + N bursts
        # tRCD (RAS-to-CAS) ≈ 18ns → 18 cycles @ 1GHz
        tRCD = 18
        tCAS = 14  # CAS latency
        tBURST = 4  # 4 cycles per burst (32B at 8B/cycle on 64-bit bus)
        tWR = 16   # write recovery

        latency = tRCD + tCAS  # row open + first CAS

        # Burst transfer time
        if is_read:
            latency += num_bursts * tBURST
        else:
            latency += num_bursts * tBURST + tWR

        # Row conflict: if accessing different row, add precharge time
        if self.row_conflict_prob > 0:
            # tRP (precharge) ≈ 18ns → 18 cycles
            latency += int(18 * self.row_conflict_prob)

        return int(latency)
