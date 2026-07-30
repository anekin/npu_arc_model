"""Input-Stationary Engine — 第三经典数据流

参考: SCALE-Sim 三种数据流对比 (arXiv 2410.22595)

Input-stationary: 激活常驻 PE，权重流式穿过。
对 M=1 decode: 极差 — 只有 1 行激活，权重重复流过 K 次。
主要是为了完整性对比，确认 WS/OS/IS 三者的 M=1 性能排序。
"""

import math

from engine.mac_engine import EngineResult, MACEngine
from models.memory_backend import AccessType


class InputStationaryEngine(MACEngine):
    """Input-stationary systolic array.

    激活常驻 PE，权重从上方流入，部分和从左方/下方流出。
    适合: 激活复用率高（大 M）、权重相对小的场景。
    不适合: M=1 decode（几乎没有激活复用）。
    """

    @property
    def engine_type(self) -> str:
        return "input_stationary"

    def estimate(self, M: int, K: int, N: int, weight_preloaded: bool = False) -> EngineResult:
        M_tiles = math.ceil(M / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = M_tiles * N_tiles

        K_tiles = math.ceil(K / self.H)
        per_tile_compute = K_tiles + self.H + self.W

        tile_weight_bytes = math.ceil(K * min(N, self.W) * self.w_bits / 8)
        full_act_bytes = math.ceil(self.H * K * self.a_bits / 8)
        per_tile_dma = self._dma_cycles(tile_weight_bytes + full_act_bytes, AccessType.SEQUENTIAL)

        bottleneck = max(per_tile_compute, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute

        total = int(first_cold + (total_tiles - 1) * bottleneck) if total_tiles > 1 else int(first_cold)

        total_macs = M * K * N
        total_weight_bytes = total_tiles * (tile_weight_bytes + full_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        raw_dma = K * N * self.w_bits // 8 + M * K * self.a_bits // 8
        raw_dma_cycles = math.ceil(raw_dma / self.eff_bw) if self.eff_bw > 0 else 0

        return EngineResult(
            compute_cycles=int(per_tile_compute * total_tiles),
            dma_cycles=int(total - per_tile_compute * total_tiles),
            total_cycles=total,
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_cycles,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if per_tile_compute > per_tile_dma else "dma",
            details={
                "M_tiles": M_tiles,
                "N_tiles": N_tiles,
                "K_tiles": K_tiles,
                "per_tile_compute": per_tile_compute,
                "per_tile_dma": round(per_tile_dma, 1),
                "dataflow": "input_stationary",
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        M_tiles = math.ceil(M / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = M_tiles * N_tiles

        K_tiles = math.ceil(K / self.H)
        per_tile_compute_pair = 2 * (K_tiles + self.H + self.W)

        tile_weight_bytes = math.ceil(K * min(N, self.W) * self.w_bits / 8)
        full_act_bytes = math.ceil(self.H * K * self.a_bits / 8)
        per_tile_dma = self._dma_cycles(2 * tile_weight_bytes + full_act_bytes, AccessType.SEQUENTIAL)

        bottleneck = max(per_tile_compute_pair, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute_pair

        total = int(first_cold + (total_tiles - 1) * bottleneck) if total_tiles > 1 else int(first_cold)

        total_macs = M * K * N * 2
        total_weight_bytes = total_tiles * (2 * tile_weight_bytes + full_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        activation_savings = total_tiles * self._dma_cycles(full_act_bytes, AccessType.SEQUENTIAL)
        raw_dma = (K * N * self.w_bits // 8 + M * K * self.a_bits // 8) * 2
        raw_dma_cycles = math.ceil(raw_dma / self.eff_bw) if self.eff_bw > 0 else 0

        return EngineResult(
            compute_cycles=int(per_tile_compute_pair * total_tiles),
            dma_cycles=int(total - per_tile_compute_pair * total_tiles),
            total_cycles=total,
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_cycles,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if per_tile_compute_pair > per_tile_dma else "dma",
            details={
                "M_tiles": M_tiles,
                "N_tiles": N_tiles,
                "K_tiles": K_tiles,
                "per_tile_compute": per_tile_compute_pair,
                "per_tile_dma": round(per_tile_dma, 1),
                "weight_cache_savings": int(activation_savings),
                "dataflow": "input_stationary",
            },
        )
