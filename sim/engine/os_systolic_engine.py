"""Output-stationary systolic-array analytical model.

The array rows and columns map the output matrix M and N dimensions.  Each PE
keeps one output partial sum while K is accumulated temporally.  This differs
from BlockEngine, which maps array rows onto the K reduction dimension and
serializes M.

The cycle model follows the standard systolic wavefront for an active
rows-by-columns output tile:

    K + active_rows + active_columns - 2

A small configurable control cost covers preload/flush sequencing.  External
memory traffic remains a logical GEMM roofline (A and B once); detailed
scratchpad-bank conflicts and transposer stalls require future RTL calibration.

References:
- Gemmini, arXiv:1911.09925, sections 2.1-2.3
- SCALE-Sim, arXiv:1811.02883
"""

import math
from typing import Any, Dict, Tuple

from engine.mac_engine import EngineResult, MACEngine


class OutputStationaryEngine(MACEngine):
    """Gemmini-style output-stationary systolic array."""

    def _parse_config(self, config: Dict[str, Any]):
        super()._parse_config(config)
        mac = config.get("mac_engine", config.get("mxu", {}))
        self.tile_control_cycles = int(mac.get("os_tile_control_cycles", 2))
        if self.tile_control_cycles < 0:
            raise ValueError("os_tile_control_cycles must be non-negative")

    @property
    def engine_type(self) -> str:
        return "os_systolic"

    def _compute_schedule(
        self, M: int, K: int, N: int,
    ) -> Tuple[int, int, int, int]:
        """Return compute cycles, M tiles, N tiles and wavefront-only cycles."""
        if min(M, K, N) <= 0:
            raise ValueError("OS GEMM dimensions must be positive")

        m_tiles = math.ceil(M / self.H)
        n_tiles = math.ceil(N / self.W)
        wavefront_cycles = 0

        for m_index in range(m_tiles):
            active_rows = min(self.H, M - m_index * self.H)
            for n_index in range(n_tiles):
                active_columns = min(self.W, N - n_index * self.W)
                wavefront_cycles += K + active_rows + active_columns - 2

        num_tiles = m_tiles * n_tiles
        compute_cycles = (
            wavefront_cycles + num_tiles * self.tile_control_cycles
        )
        return compute_cycles, m_tiles, n_tiles, wavefront_cycles

    def estimate(
        self,
        M: int,
        K: int,
        N: int,
        weight_preloaded: bool = False,
    ) -> EngineResult:
        """Estimate one GEMM with M/N spatial and K temporal mapping."""
        compute_cycles, m_tiles, n_tiles, wavefront_cycles = (
            self._compute_schedule(M, K, N)
        )

        weight_bytes = math.ceil(K * N * self.w_bits / 8)
        activation_bytes = math.ceil(M * K * self.a_bits / 8)
        transferred_weight_bytes = 0 if weight_preloaded else weight_bytes
        dma_bytes = transferred_weight_bytes + activation_bytes
        dma_cycles = math.ceil(dma_bytes / max(self.eff_bw, 1e-9))
        total_cycles = max(compute_cycles, dma_cycles)
        total_macs = M * K * N
        num_tiles = m_tiles * n_tiles

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total_cycles,
            utilization=min(
                1.0,
                total_macs / max(self.H * self.W * total_cycles, 1),
            ),
            ops=total_macs * self.ops_per_mac,
            num_tiles=num_tiles,
            weight_bytes=weight_bytes,
            bottleneck="compute" if compute_cycles >= dma_cycles else "dma",
            details={
                "dataflow": "output_stationary",
                "mapping": "M_by_N_spatial_K_temporal",
                "M_tiles": m_tiles,
                "N_tiles": n_tiles,
                "K_accumulation_cycles": K,
                "K_streaming_assumption": "continuous_partial_sum_residency",
                "wavefront_cycles": wavefront_cycles,
                "tile_control_cycles": self.tile_control_cycles,
                "transposer": "pipelined_A_input",
                "output_partial_sum_location": "PE_accumulator",
                "weight_bytes": weight_bytes,
                "weight_transfer_bytes": transferred_weight_bytes,
                "activation_bytes": activation_bytes,
                "weight_preloaded": weight_preloaded,
                "roofline_overlap": True,
            },
        )

    def estimate_weight_cache_pair(
        self, M: int, K: int, N: int,
    ) -> EngineResult:
        """Estimate Gate+Up as two OS outputs with shared activation fetch.

        OS does not use BlockEngine's dual-weight register mechanism.  The two
        output matrices require separate wavefronts and PE accumulations, while
        the common activation matrix may remain in the scratchpad.
        """
        single_compute, m_tiles, n_tiles, wavefront_cycles = (
            self._compute_schedule(M, K, N)
        )
        compute_cycles = 2 * single_compute
        one_weight_bytes = math.ceil(K * N * self.w_bits / 8)
        weight_bytes = 2 * one_weight_bytes
        activation_bytes = math.ceil(M * K * self.a_bits / 8)
        dma_bytes = weight_bytes + activation_bytes
        dma_cycles = math.ceil(dma_bytes / max(self.eff_bw, 1e-9))
        total_cycles = max(compute_cycles, dma_cycles)
        total_macs = 2 * M * K * N
        num_tiles = 2 * m_tiles * n_tiles

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total_cycles,
            utilization=min(
                1.0,
                total_macs / max(self.H * self.W * total_cycles, 1),
            ),
            ops=total_macs * self.ops_per_mac,
            num_tiles=num_tiles,
            weight_bytes=weight_bytes,
            bottleneck="compute" if compute_cycles >= dma_cycles else "dma",
            details={
                "dataflow": "output_stationary",
                "mapping": "M_by_N_spatial_K_temporal",
                "M_tiles": m_tiles,
                "N_tiles": n_tiles,
                "wavefront_cycles_per_output": wavefront_cycles,
                "tile_control_cycles": self.tile_control_cycles,
                "shared_activation": True,
                "weight_cache_mechanism": "scratchpad_activation_reuse",
                "weight_bytes": weight_bytes,
                "activation_bytes": activation_bytes,
                "roofline_overlap": True,
            },
        )
