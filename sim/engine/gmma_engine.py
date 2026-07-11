"""GMMA Engine — Group Matrix Multiply Accumulate (Hopper H100 style)

参考: NVIDIA Hopper H100 GMMA + TMA (Tensor Memory Accelerator)

GMMA = Block Engine 的异步升级版:
  1. Tile 粒度: 128×128 (large tiles, low fragmentation)
  2. TMA: 异步 DMA 引擎，通过 descriptor 在后台预取下一 tile
  3. Shared Memory: 大容量片上 SRAM 做权重 buffer

对单 die NPU decode:
  - TMA 的价值: 隐藏部分 DMA 时间 (不是 100%，有 descriptor 开销)
  - 代价: TMA 单元面积 + Shared Memory 容量
  - DRAM 墙仍在 — 但利用率可到 100%
"""

import math
from typing import Any, Dict
from engine.mac_engine import MACEngine, EngineResult


class GMMAEngine(MACEngine):
    """GMMA — Group MMA with async TMA DMA.

    Key architectural difference from Block Engine:
      Block: DMA → compute → DMA → compute  (sequential, double-buffered)
      GMMA:  DMA overlap compute             (async via TMA descriptors)

    Tile shape follows the configured array dimensions (default 128×128).
    Per-tile compute retains a systolic-like fill/drain shape, but the
    asynchronous descriptor issue and large tile make the per-tile
    pipeline penalty much smaller than a pure weight-stationary array.

    Area model:
      - MAC array: same as Block Engine
      - TMA unit: +2mm² for descriptor engine + crossbar
      - Shared Memory for weight buffer: +4MB → +6mm² (0.0015mm²/KB)
    """

    # TMA can hide DMA latency behind compute when the engine is
    # compute-bound, but it cannot exceed the physical DRAM bandwidth.
    # This factor is applied only to the exposed DMA on the critical
    # path; the steady-state pipeline bottleneck is still clamped to
    # the raw per-tile DMA time.
    TMA_OVERLAP = 0.5

    # GMMA's group-MMA unit still has a systolic-like fill/drain pipeline,
    # but the async TMA front-end and 128×128 tile amortize the overhead.
    # We keep the (H+W)+(M+H) shape from SystolicEngine and scale it down
    # to reflect the much shorter effective pipeline in a group-MMA unit.
    GMMA_PIPELINE_SCALE = 0.05

    TMA_AREA_MM2 = 2.0
    SHMEM_KB = 4096    # 4MB shared memory for weights

    @property
    def engine_type(self) -> str:
        return "gmma"

    def _per_tile_compute(self, M: int) -> int:
        """Systolic pipeline depth per K-tile: H (weight load) + M (act stream) + W (drain)."""
        return self.H + M + self.W

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """GMMA GEMM — K-tiling + correct activation DMA + TMA overlap."""
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        # Total weight and activations
        total_weight_bytes = K * N * self.w_bits // 8
        act_bytes = M * K * self.a_bits // 8

        # SRAM efficiency
        weight_dram_eff = self._dram_eff_for_bytes(total_weight_bytes)
        if weight_dram_eff <= 0:
            weight_dma_cycles = 0
        else:
            weight_dma_cycles = total_weight_bytes / (self.eff_bw * weight_dram_eff)

        act_dma_cycles = act_bytes / self.eff_bw
        total_dma = weight_dma_cycles + act_dma_cycles

        # TMA overlap only when compute-bound
        per_tile_compute = self._per_tile_compute(M)
        total_compute = per_tile_compute * total_tiles
        tma_dma = total_dma * (1 - self.TMA_OVERLAP)

        total_cycles = max(total_compute, tma_dma)
        total_macs = M * K * N
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total_cycles if total_cycles > 0 else 0.0

        compute_cycles = int(total_compute)
        dma_cycles = int(total_dma)
        dma_hidden_pct = (1 - dma_cycles / max(dma_cycles, 1)) * 0 if total_dma <= 0 else \
            (1 - min(total_cycles, total_dma) / max(total_dma, 1)) * 100

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=int(total_cycles),
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=int(total_weight_bytes),
            bottleneck="dma" if total_dma > total_compute else "compute",
            details={
                "K_tiles": K_tiles, "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "tma_overlap": self.TMA_OVERLAP,
                "weight_dram_eff": round(weight_dram_eff, 3),
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up with GMMA — dual weight registers + TMA overlap.

        The weight cache holds both gate and up tiles in shared memory.
        Each tile still loads only one set of activations but two sets of
        weights; the two matmuls run back-to-back on the same GMMA unit.
        TMA overlap applies to the (heavier) DMA stream as well.
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)
        tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)

        # Dual weights (gate + up) but shared activations.
        dual_weight_bytes = 2 * tile_weight_bytes
        per_tile_dma = (dual_weight_bytes + tile_act_bytes) / self.eff_bw
        tma_exposed_dma = per_tile_dma * (1 - self.TMA_OVERLAP)

        # Two matmuls per tile on the same GMMA unit.
        single_compute = self._per_tile_compute(M)
        per_tile_compute = 2 * single_compute

        bottleneck = max(per_tile_compute, per_tile_dma)
        first_tile = per_tile_dma + per_tile_compute

        if total_tiles > 1:
            total = int(first_tile + (total_tiles - 1) * bottleneck)
        else:
            total = int(first_tile)

        total_macs = M * K * N * 2
        total_weight_bytes = total_tiles * (dual_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        compute_cycles = int(per_tile_compute * total_tiles)
        dma_cycles = int(total - compute_cycles)

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total,
            utilization=util,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="dma" if per_tile_dma > per_tile_compute else "compute",
            details={
                "K_tiles": K_tiles,
                "N_tiles": N_tiles,
                "per_tile_dma": round(per_tile_dma, 1),
                "tma_exposed_dma": round(tma_exposed_dma, 1),
                "per_tile_compute": per_tile_compute,
                "tma_overlap": self.TMA_OVERLAP,
                "weight_cache": True,
            },
        )
