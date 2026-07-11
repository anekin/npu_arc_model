"""Input-Stationary Engine — 第三经典数据流

参考: SCALE-Sim 三种数据流对比 (arXiv 2410.22595)

Input-stationary: 激活常驻 PE，权重流式穿过。
对 M=1 decode: 极差 — 只有 1 行激活，权重重复流过 K 次。
主要是为了完整性对比，确认 WS/OS/IS 三者的 M=1 性能排序。
"""

import math
from typing import Any, Dict
from engine.mac_engine import MACEngine, EngineResult


class InputStationaryEngine(MACEngine):
    """Input-stationary systolic array.

    激活常驻 PE，权重从上方流入，部分和从左方/下方流出。
    适合: 激活复用率高（大 M）、权重相对小的场景。
    不适合: M=1 decode（几乎没有激活复用）。
    """

    @property
    def engine_type(self) -> str:
        return "input_stationary"


    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """IS GEMM estimate with M-dependent activation-reuse scaling.

        Input-stationary keeps the M×K activation matrix resident in the PE
        array and streams K×N weights through.  Reuse is high when many
        activation rows fill the array (large M / prefill) and negligible
        when only one row is active (M=1 / decode).

        Reuse model:
          reuse_factor = min(M, H) / H
          - M=H: array fully occupied, each activation row is reused by many
            weight passes; compute is the pipelined base cost.
          - M=1: only one of H rows is active, so fill/drain overhead and the
            K-weight stream are amortized over a single row.

        Per-tile compute therefore scales inversely with reuse:
          base_compute = K_tiles + H + W
          per_tile_compute = base_compute / reuse_factor
        where K_tiles = ceil(K/H) is the number of systolic weight passes.

        DMA stays tile-local: each tile loads its stationary activation rows
        once plus the weights for the current N-tile.
        """
        # IS: M maps to array rows (H), N to columns (W)
        M_tiles = math.ceil(M / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = M_tiles * N_tiles

        # Architecture-derived activation reuse factor.
        reuse_factor = min(M, self.H) / self.H

        # Base compute assumes the array is fully occupied (reuse_factor=1).
        # K_tiles weight passes, plus H+W fill/drain overhead.
        K_tiles = math.ceil(K / self.H)
        base_compute = K_tiles + self.H + self.W
        per_tile_compute = base_compute / reuse_factor

        # Weight data: K × N_tile columns = K × min(N,W)
        tile_weight_bytes = math.ceil(K * min(N, self.W) * self.w_bits / 8)
        # Activation data: the stationary rows loaded for this M-tile
        tile_act_bytes = math.ceil(min(M, self.H) * K * self.a_bits / 8)
        per_tile_dma = (tile_weight_bytes + tile_act_bytes) / self.eff_bw

        bottleneck = max(per_tile_compute, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute

        if total_tiles > 1:
            total = int(first_cold + (total_tiles - 1) * bottleneck)
        else:
            total = int(first_cold)

        total_macs = M * K * N
        total_weight_bytes = total_tiles * (tile_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        return EngineResult(
            compute_cycles=int(per_tile_compute * total_tiles),
            dma_cycles=int(total - per_tile_compute * total_tiles),
            total_cycles=total,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if per_tile_compute > per_tile_dma else "dma",
            details={
                "M_tiles": M_tiles, "N_tiles": N_tiles,
                "K_tiles": K_tiles,
                "per_tile_compute": per_tile_compute,
                "per_tile_dma": round(per_tile_dma, 1),
                "reuse_factor": round(reuse_factor, 4),
                "dataflow": "input_stationary",
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up pair with IS activation stationarity.

        The M×K activation stays resident while two sets of weights (gate and
        up) stream through, so we pay the activation DMA only once per tile
        but fetch twice the weight bytes and perform twice the compute.
        """
        M_tiles = math.ceil(M / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = M_tiles * N_tiles

        reuse_factor = min(M, self.H) / self.H
        K_tiles = math.ceil(K / self.H)
        base_compute = K_tiles + self.H + self.W
        per_tile_compute_single = base_compute / reuse_factor
        per_tile_compute_pair = 2 * per_tile_compute_single

        # Weight data: two sets of gate+up weights per tile.
        tile_weight_bytes = math.ceil(K * min(N, self.W) * self.w_bits / 8)
        tile_act_bytes = math.ceil(min(M, self.H) * K * self.a_bits / 8)
        per_tile_dma = (2 * tile_weight_bytes + tile_act_bytes) / self.eff_bw

        bottleneck = max(per_tile_compute_pair, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute_pair

        if total_tiles > 1:
            total = int(first_cold + (total_tiles - 1) * bottleneck)
        else:
            total = int(first_cold)

        total_macs = M * K * N * 2
        total_weight_bytes = total_tiles * (2 * tile_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        # Activation-load savings vs two independent single estimates.
        activation_savings = total_tiles * tile_act_bytes / self.eff_bw

        return EngineResult(
            compute_cycles=int(per_tile_compute_pair * total_tiles),
            dma_cycles=int(total - per_tile_compute_pair * total_tiles),
            total_cycles=total,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if per_tile_compute_pair > per_tile_dma else "dma",
            details={
                "M_tiles": M_tiles, "N_tiles": N_tiles,
                "K_tiles": K_tiles,
                "per_tile_compute": per_tile_compute_pair,
                "per_tile_dma": round(per_tile_dma, 1),
                "reuse_factor": round(reuse_factor, 4),
                "weight_cache_savings": int(activation_savings),
                "dataflow": "input_stationary",
            },
        )
