"""WMMA Engine — 16×16×16 Warp Matrix Multiply Accumulate

参考: NVIDIA Volta/Ampere Tensor Core WMMA API

每个 warp (32 threads) 协作算一个 16×16×16 块。
对 M=1 decode 来说，16×16 粒度产生灾难性 DMA 碎片。

与 Block Engine 的本质区别:
  - Block: 全阵列 128×128 一个大 tile
  - WMMA: 阵列被切成 64 个独立的 16×16 小块，各自 DMA
  - 同样 total bytes 但 64× 更多的 DMA 事务 → 启动开销爆炸
"""

import math
from typing import Any, Dict

from engine.mac_engine import MACEngine, EngineResult


class WMMAEngine(MACEngine):
    """Warp Matrix Multiply Accumulate — 16×16 TC blocks."""

    # WMMA fragment shape: one warp (32 threads) computes a 16×16×16 MMA.
    FRAG_M = 16
    FRAG_N = 16
    FRAG_K = 16
    WARP_THREADS = 32

    # Per-fragment warp synchronization + compute.
    #  - warp_sync: barrier across 32 threads (~32 cycles)
    #  - frag_mac: 16×16 MAC inside one warp (~16 cycles)
    #  - frag_serialization: single-die NPU has no GPU-style warp scheduler;
    #    each 16×16 fragment is dispatched serially and pays full issue/DMA
    #    setup overhead.  This dominates large GEMMs.
    WARP_SYNC_CYCLES = 32
    FRAG_MAC_CYCLES = 16
    WARP_FRAGMENT_SERIALIZATION_CYCLES = 1600

    # Base DMA startup per fragment.  Real fragmented 16×16 loads hit DRAM
    # with terrible burst efficiency; the serialization constant above
    # captures most of that penalty.
    DMA_STARTUP_CYCLES = 10

    @property
    def engine_type(self) -> str:
        return "wmma"

    @property
    def num_warps(self) -> int:
        """Number of independent 16×16 blocks that fit in the array."""
        return (self.H // self.FRAG_M) * (self.W // self.FRAG_N)

    def _per_fragment_compute(self, ops_multiplier: int = 1) -> int:
        base = (
            self.WARP_FRAGMENT_SERIALIZATION_CYCLES
            + self.WARP_SYNC_CYCLES
            + self.FRAG_MAC_CYCLES
        )
        return base * ops_multiplier

    def _fragment_counts(self, M: int, K: int, N: int) -> Dict[str, int]:
        """Return fragment counts per (H,W) tile and across the full GEMM."""
        frag_M_total = math.ceil(M / self.FRAG_M)
        frag_K_total = math.ceil(K / self.FRAG_K)
        frag_N_total = math.ceil(N / self.FRAG_N)
        total_fragments = frag_M_total * frag_K_total * frag_N_total

        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        M_tiles = math.ceil(M / self.H)
        total_tiles = M_tiles * K_tiles * N_tiles

        fragments_per_tile = total_fragments // total_tiles if total_tiles else 0

        return {
            "frag_M_total": frag_M_total,
            "frag_K_total": frag_K_total,
            "frag_N_total": frag_N_total,
            "total_fragments": total_fragments,
            "M_tiles": M_tiles,
            "K_tiles": K_tiles,
            "N_tiles": N_tiles,
            "total_tiles": total_tiles,
            "fragments_per_tile": fragments_per_tile,
        }

    def _estimate(self, M: int, K: int, N: int,
                  weight_multiplier: int = 1, ops_multiplier: int = 1) -> EngineResult:
        """Core WMMA estimator with explicit per-fragment serialization."""
        frag = self._fragment_counts(M, K, N)
        fragments_per_tile = frag["fragments_per_tile"]
        total_tiles = frag["total_tiles"]

        per_frag_compute = self._per_fragment_compute(ops_multiplier=ops_multiplier)

        per_tile_weight_bytes = weight_multiplier * math.ceil(
            self.H * self.W * self.w_bits / 8
        )
        per_tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)
        per_tile_dma = (
            fragments_per_tile * self.DMA_STARTUP_CYCLES
            + (per_tile_weight_bytes + per_tile_act_bytes) / self.eff_bw
        )

        per_tile_compute = fragments_per_tile * per_frag_compute

        # Double-buffered pipeline across tiles.
        bottleneck = max(per_tile_compute, per_tile_dma)
        first_cold = per_tile_dma + per_tile_compute

        if total_tiles > 1:
            total = int(first_cold + (total_tiles - 1) * bottleneck)
        else:
            total = int(first_cold)

        total_macs = M * K * N * ops_multiplier
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        compute_cycles = int(per_tile_compute * total_tiles)
        dma_cycles = total - compute_cycles

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=int(total_tiles * (per_tile_weight_bytes + per_tile_act_bytes)),
            bottleneck="compute" if per_tile_compute > per_tile_dma else "dma",
            details={
                "frag_M_total": frag["frag_M_total"],
                "frag_K_total": frag["frag_K_total"],
                "frag_N_total": frag["frag_N_total"],
                "total_fragments": frag["total_fragments"],
                "fragments_per_tile": fragments_per_tile,
                "per_fragment_compute": per_frag_compute,
                "per_tile_compute": per_tile_compute,
                "per_tile_dma": round(per_tile_dma, 1),
                "tile_size": f"{self.FRAG_M}×{self.FRAG_K}×{self.FRAG_N}",
                "num_warps": self.num_warps,
            },
        )

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """WMMA GEMM — massive tile fragmentation for small M."""
        return self._estimate(M, K, N, weight_multiplier=1, ops_multiplier=1)

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up pair: dual weight fragments, shared activation fragment.

        A single-die NPU cannot hide DMA behind warp scheduling, so the pair
        still pays nearly twice the compute.  The only saving is the shared
        activation load, which is tiny compared to the serialization cost.
        """
        return self._estimate(M, K, N, weight_multiplier=2, ops_multiplier=2)
