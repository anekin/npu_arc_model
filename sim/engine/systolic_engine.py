"""Systolic Engine — weight-stationary systolic array (现有 MXU v2 模型)"""

import math

from engine.mac_engine import EngineResult, MACEngine


class SystolicEngine(MACEngine):
    """Weight-stationary systolic array — 时空映射.

    权重对角加载到 PE 阵列，激活流式穿过。
    Decode per-tile compute follows MXUModel v2 interleaving: H*(M+1)+W.
    Prefill drain is conditional on partial vs full M-tiles.
    """

    @property
    def engine_type(self) -> str:
        return "systolic"

    def _per_ktile_compute(self, M: int) -> int:
        M_tiles = max(1, (M + self.H - 1) // self.H)
        last_rows = M - (M_tiles - 1) * self.H
        if last_rows <= 0:
            last_rows = self.H
        return (M_tiles - 1) * (2 * self.H + self.W) + (self.H + self.W + last_rows)

    def estimate(self, M: int, K: int, N: int, weight_preloaded: bool = False) -> EngineResult:
        if M <= 0:
            raise ValueError(f"SystolicEngine.estimate requires M > 0, got {M}")

        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)
        tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)

        per_tile_compute = self._per_ktile_compute(M)
        pipeline_fill = self.H + self.W
        pipeline_drain = per_tile_compute - pipeline_fill
        per_tile_dma = (tile_weight_bytes + tile_act_bytes) / self.eff_bw

        bottleneck_per_tile = max(per_tile_compute, per_tile_dma)
        first_tile_cold = per_tile_dma + per_tile_compute

        if total_tiles > 1:
            total_compute_cycles = first_tile_cold + (total_tiles - 1) * bottleneck_per_tile
        else:
            total_compute_cycles = first_tile_cold

        total_dma_bytes = total_tiles * (tile_weight_bytes + tile_act_bytes)
        total_macs = M * K * N
        ideal_cycles = math.ceil(total_macs / self.peak_macs_per_cycle)
        utilization = ideal_cycles / total_compute_cycles if total_compute_cycles > 0 else 0.0

        total = int(total_compute_cycles)
        compute_cycles = int(per_tile_compute * total_tiles)
        dma_cycles = total - compute_cycles

        raw_dma = K * N * self.w_bits // 8 + M * K * self.a_bits // 8
        raw_dma_cycles = math.ceil(raw_dma / self.eff_bw) if self.eff_bw > 0 else 0

        M_tiles = max(1, (M + self.H - 1) // self.H)

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total,
            utilization=utilization,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal_cycles,
            raw_dma_cycles=raw_dma_cycles,
            num_tiles=total_tiles,
            weight_bytes=total_dma_bytes,
            bottleneck="compute" if per_tile_compute > per_tile_dma else "dma",
            details={
                "K_tiles": K_tiles,
                "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "per_tile_dma": round(per_tile_dma, 1),
                "pipeline_fill": pipeline_fill,
                "pipeline_drain": pipeline_drain,
                "M_tiles": M_tiles,
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up with PE dual weight register."""
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_dual = K_tiles * N_tiles

        dual_weight_bytes = 2 * math.ceil(self.H * self.W * self.w_bits / 8)
        dual_act_bytes = math.ceil(M * self.H * self.a_bits / 8)
        dual_dma = (dual_weight_bytes + dual_act_bytes) / self.eff_bw

        per_matm_drain = M + self.W
        dual_compute = 2 * per_matm_drain + 1

        fill = self.H + self.W
        drain = M + self.H

        bottleneck = max(dual_dma, dual_compute)
        first_cold = dual_dma + dual_compute

        if N_tiles >= 2:
            per_Ktile = fill + first_cold + (N_tiles - 1) * bottleneck + drain
        else:
            per_Ktile = fill + first_cold + drain

        total = int(K_tiles * per_Ktile)

        total_macs = M * K * N * 2
        total_weight_bytes = total_dual * (dual_weight_bytes + dual_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        raw_dma = (K * N * self.w_bits // 8 + M * K * self.a_bits // 8) * 2
        raw_dma_cycles = math.ceil(raw_dma / self.eff_bw) if self.eff_bw > 0 else 0

        return EngineResult(
            compute_cycles=int(dual_compute * total_dual),
            dma_cycles=int(total - dual_compute * total_dual),
            total_cycles=total,
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_cycles,
            num_tiles=total_dual,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if dual_compute > dual_dma else "dma",
            details={
                "K_tiles": K_tiles,
                "N_tiles": N_tiles,
                "dual_dma": round(dual_dma, 1),
                "dual_compute": dual_compute,
                "weight_cache": True,
            },
        )
