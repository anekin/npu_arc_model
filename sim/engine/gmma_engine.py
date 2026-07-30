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
from typing import Any

from engine.mac_engine import EngineResult, MACEngine
from models.memory_backend import AccessType


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
    SHMEM_KB = 4096  # 4MB shared memory for weights

    def _parse_config(self, config: dict[str, Any]) -> None:
        """Parse common config plus GMMA-specific calibration parameters."""
        super()._parse_config(config)
        scale = config.get("gmma", {}).get("pipeline_scale", self.GMMA_PIPELINE_SCALE)
        try:
            scale = float(scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gmma.pipeline_scale must be a number, got {scale!r}") from exc
        if not (0 < scale <= 1):
            raise ValueError(f"gmma.pipeline_scale must be in (0, 1], got {scale}")
        self.pipeline_scale = scale

    @property
    def engine_type(self) -> str:
        return "gmma"

    def _per_tile_compute(self, M: int) -> int:
        """Systolic pipeline depth per K-tile, scaled by GMMA's async pipeline."""
        return max(1, math.ceil((self.H + M + self.W) * self.pipeline_scale))

    def estimate(self, M: int, K: int, N: int, weight_preloaded: bool = False) -> EngineResult:
        """GMMA GEMM — K-tiling + correct activation DMA + TMA overlap."""
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        # Total weight and activations
        total_weight_bytes = K * N * self.w_bits // 8
        act_bytes = M * K * self.a_bits // 8

        # SRAM efficiency
        weight_dram_eff = self._dram_eff_for_bytes(total_weight_bytes)
        weight_dma_cycles = 0 if weight_dram_eff <= 0 else total_weight_bytes / (self.eff_bw_weight * weight_dram_eff)

        act_dma_cycles = self._dma_cycles(act_bytes, AccessType.SEQUENTIAL)
        total_dma = weight_dma_cycles + act_dma_cycles

        # TMA overlap — only for diagnostics; total_cycles uses raw DMA floor
        per_tile_compute = self._per_tile_compute(M)
        total_compute = per_tile_compute * total_tiles
        per_tile_dma_val = total_dma / total_tiles if total_tiles > 0 else 0.0
        tma_exposed_dma_val = per_tile_dma_val * (1 - self.TMA_OVERLAP)
        tma_hidden_dma_val = per_tile_dma_val * self.TMA_OVERLAP

        # Enforce all three physical floors: compute, DMA ceil, raw DMA.
        raw_dma_bytes = K * N * self.w_bits // 8 + M * K * self.a_bits // 8
        raw_dma_floor = math.ceil(raw_dma_bytes / self.bw_raw) if self.bw_raw > 0 else 0
        total_dma_ceil = math.ceil(total_dma)

        total_macs = M * K * N
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)

        total_cycles = max(int(total_compute), ideal, raw_dma_floor, total_dma_ceil)
        util = ideal / total_cycles if total_cycles > 0 else 0.0

        compute_cycles = int(total_compute)
        dma_cycles = total_dma_ceil

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=int(total_cycles),
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_floor,
            num_tiles=total_tiles,
            weight_bytes=int(total_weight_bytes),
            bottleneck="dma" if total_dma > total_compute else "compute",
            details={
                "K_tiles": K_tiles,
                "N_tiles": N_tiles,
                "per_tile_compute": per_tile_compute,
                "per_tile_dma": round(per_tile_dma_val, 1),
                "raw_dma_cycles": int(total_dma),
                "tma_hidden_dma": round(tma_hidden_dma_val, 1),
                "tma_exposed_dma": round(tma_exposed_dma_val, 1),
                "pipeline_scale": self.pipeline_scale,
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
        per_tile_dma_raw = self._dma_cycles(dual_weight_bytes + tile_act_bytes, AccessType.SEQUENTIAL)
        tma_exposed_dma = per_tile_dma_raw * (1 - self.TMA_OVERLAP)
        tma_hidden_dma = per_tile_dma_raw * self.TMA_OVERLAP

        # Two matmuls per tile on the same GMMA unit, with pipeline scaling.
        single_compute = self._per_tile_compute(M)
        per_tile_compute = 2 * single_compute

        # Enforce raw-DMA floor: bottleneck is max of compute and raw DMA.
        bottleneck = max(per_tile_compute, per_tile_dma_raw)
        first_tile = per_tile_dma_raw + per_tile_compute

        total = int(first_tile + (total_tiles - 1) * bottleneck) if total_tiles > 1 else int(first_tile)

        total_macs = M * K * N * 2
        total_weight_bytes = total_tiles * (dual_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)

        raw_dma_bytes = (K * N * self.w_bits // 8 + M * K * self.a_bits // 8) * 2
        raw_dma_floor = math.ceil(raw_dma_bytes / self.bw_raw) if self.bw_raw > 0 else 0
        total_dma_ceil = math.ceil(total_tiles * per_tile_dma_raw)

        total = max(total, ideal, raw_dma_floor, total_dma_ceil)
        util = ideal / total if total > 0 else 0.0

        compute_cycles = int(per_tile_compute * total_tiles)
        dma_cycles = total_dma_ceil

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total,
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_floor,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
            bottleneck="dma" if per_tile_dma_raw > per_tile_compute else "compute",
            details={
                "K_tiles": K_tiles,
                "N_tiles": N_tiles,
                "per_tile_dma": round(per_tile_dma_raw, 1),
                "per_tile_compute": per_tile_compute,
                "raw_dma_cycles": int(per_tile_dma_raw * total_tiles),
                "tma_hidden_dma": round(tma_hidden_dma, 1),
                "tma_exposed_dma": round(tma_exposed_dma, 1),
                "pipeline_scale": self.pipeline_scale,
                "tma_overlap": self.TMA_OVERLAP,
                "weight_cache": True,
            },
        )
