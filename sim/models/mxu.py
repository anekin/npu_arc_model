"""MXU 性能模型 (legacy systolic, v2) — 64×64 array (superseded by Block 64×64 broadcast; preserved for systolic regression)

v2 changes:
- 使用 weight streaming 假设（3B 模型权重 > 片上 SRAM，必须每 token 从 DRAM 流式加载）
- 加入 128×128 tile 粒度建模
- DMA/MXU double-buffer overlap
- DRAM 有效带宽（85% 效率，含刷新 + 行冲突）
"""

import math
from dataclasses import dataclass
from typing import Any, Dict

# Marker consumed by overnight_loop.py consistency checks
V2_BANDWIDTH_AWARE = True


@dataclass
class MXUResult:
    compute_cycles: int
    stall_cycles_dram: int
    stall_cycles_sram: int
    total_cycles: int
    utilization: float
    ops: int
    num_tiles: int = 0
    weight_bytes: int = 0

    def __repr__(self):
        return (f"MXU(compute={self.compute_cycles}, stall_dram={self.stall_cycles_dram}, "
                f"stall_sram={self.stall_cycles_sram}, util={self.utilization:.1%}, "
                f"tiles={self.num_tiles})")


class MXUModel:
    """Weight-stationary systolic array v2 — bandwidth-aware."""

    def __init__(self, config: Dict[str, Any]):
        mxu = config["mxu"]
        self.H = int(mxu["array_height"])       # 128
        self.W = int(mxu["array_width"])        # 128
        self.f_mhz = int(mxu["frequency_mhz"])  # 1000
        self.w_bits = int(mxu["weight_precision_bits"])     # 4
        self.a_bits = int(mxu["activation_precision_bits"]) # 8
        self.ops_per_mac = int(mxu["ops_per_mac"])          # 2
        self.double_buffer = bool(mxu.get("double_buffer", True))

        mem = config["memory"]
        self.bw_bytes_per_cycle = float(mem["bandwidth_bytes_per_cycle"])  # 51.2
        self.dram_efficiency = float(mem.get("dram_efficiency", 0.85))    # 85%

        # DMA bandwidth multiplier (L2 optimization: 128-bit DRAM or 4ch DMA)
        opts = config.get("optimizations", {})
        self.bw_multiplier = float(opts.get("dma_bw_multiplier", 1.0))

        # Effective bandwidth (with multiplier)
        self.eff_bw = (self.bw_bytes_per_cycle * self.dram_efficiency
                       * self.bw_multiplier)

    @property
    def macs_per_cycle(self) -> int:
        return self.H * self.W * self.ops_per_mac  # 32768

    def _estimate_decode(
        self, M: int, K: int, N: int
    ) -> MXUResult:
        """Decode mode (M=1): weight-stationary with tiled weight streaming.

        For weight-stationary decode on a 128×128 array:
        - Tile along K: ceil(K/H) tiles
        - Tile along N: ceil(N/W) tiles
        - Per (k_tile, n_tile): load 128×128 INT4 weights (8KB), compute 128 MACs

        With double-buffering: DMA next tile while MXU computes current tile.
        Bottleneck = max(per_tile_compute, per_tile_dma).
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        # Per-tile weight: H×W×w_bits/8 bytes
        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)  # 8KB

        # Per-tile activation: M×H×a_bits/8 bytes (M=1 for decode)
        tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)

        # --- Per-tile compute ---
        # V2 interleaving: each of M tokens streams through stationary weights
        # Fill (H+W) + first drain (H) + (M-1)*H for subsequent tokens
        # Equivalently: H*(M+1) + W
        per_tile_compute = self.H * (M + 1) + self.W

        # --- Per-tile DMA ---
        per_tile_dma = (tile_weight_bytes + tile_act_bytes) / self.eff_bw

        # --- With double-buffering ---
        # For N_tiles within same K_tile: activation shared, only weight changes
        # For first tile: cold start (DMA + compute sequential)
        # For subsequent tiles: overlap DMA_next with compute_current
        #   bottleneck_per_tile = max(per_tile_compute, per_tile_dma)

        bottleneck_per_tile = max(per_tile_compute, per_tile_dma)

        # Total cycles: first tile sequential + (N-1) overlapping tiles
        # For each K_tile, first N_tile needs cold DMA+compute
        # Remaining N_tiles-1 overlap

        # Actually, with proper pipelining:
        # DMA for first tile, then while computing tile 0, DMA for tile 1
        pipeline_depth = 2  # double buffer

        # Simplified model: total = first_tile_dma + total_tiles × bottleneck
        # (first DMA is not overlapped, all subsequent are)
        first_tile_cold = per_tile_dma + per_tile_compute
        if total_tiles > 1:
            total_compute_cycles = first_tile_cold + (total_tiles - 1) * bottleneck_per_tile
        else:
            total_compute_cycles = first_tile_cold

        # Total weight data for bandwidth utilization tracking
        total_weight_bytes = total_tiles * tile_weight_bytes + total_tiles * tile_act_bytes

        # MAC operations
        total_macs = M * K * N

        ideal_cycles = math.ceil(total_macs / self.macs_per_cycle)
        utilization = ideal_cycles / total_compute_cycles if total_compute_cycles > 0 else 0.0

        return MXUResult(
            compute_cycles=int(total_compute_cycles),
            stall_cycles_dram=0,  # Already accounted in bottleneck model
            stall_cycles_sram=0,
            total_cycles=int(total_compute_cycles),
            utilization=utilization,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
        )

    def _estimate_prefill(
        self, M: int, K: int, N: int
    ) -> MXUResult:
        """Prefill mode (M>1): compute-bound, but weights must still stream.

        Prefill has large M → good array utilization.
        Weights streamed once, reused across all M rows of the tile.
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_tiles = K_tiles * N_tiles

        tile_weight_bytes = math.ceil(self.H * self.W * self.w_bits / 8)
        tile_act_bytes = math.ceil(M * self.H * self.a_bits / 8)

        # For M>1, the M dimension is tiled across the array height
        M_tiles = math.ceil(M / self.H)
        pipeline_fill = self.H + self.W

        # Per (k,n,m) tile compute:
        # With weight-stationary, weights loaded once, activations stream through
        # For M < H: only M rows are active, partial tile — drain = M
        # For M >= H: full tile(s) — drain = H per tile
        if M_tiles == 1 and M < self.H:
            # Partial tile: pipeline fill (H+W) + drain (M)
            per_tile_compute = pipeline_fill + M
        else:
            # Full tile(s): each M-tile is H rows, drain = H
            pipeline_drain = self.H
            per_m_tile_compute = pipeline_fill + pipeline_drain
            per_tile_compute = M_tiles * per_m_tile_compute

        # DMA per tile: weight + activation for all M rows
        per_tile_dma = (tile_weight_bytes + M_tiles * tile_act_bytes) / self.eff_bw

        bottleneck_per_tile = max(per_tile_compute, per_tile_dma)

        first_tile_cold = per_tile_dma + per_tile_compute
        if total_tiles > 1:
            total_cycles = first_tile_cold + (total_tiles - 1) * bottleneck_per_tile
        else:
            total_cycles = first_tile_cold

        total_weight_bytes = total_tiles * tile_weight_bytes + total_tiles * M_tiles * tile_act_bytes
        total_macs = M * K * N
        ideal_cycles = math.ceil(total_macs / self.macs_per_cycle)
        utilization = ideal_cycles / total_cycles if total_cycles > 0 else 0.0

        return MXUResult(
            compute_cycles=int(total_cycles),
            stall_cycles_dram=0,
            stall_cycles_sram=0,
            total_cycles=int(total_cycles),
            utilization=utilization,
            ops=total_macs,
            num_tiles=total_tiles,
            weight_bytes=total_weight_bytes,
        )

    def estimate(
        self, M: int, K: int, N: int
    ) -> MXUResult:
        """Estimate cycles for GEMM: (M×K) × (K×N) → (M×N).

        v2: always models weight streaming from DRAM. 3B model weights
        (1.5GB) cannot fit in 2.5MB on-chip SRAM.
        """
        # Batch decode (M ≤ 8): tokens stream sequentially through
        # weight-stationary array. Interleaving model: H*(M+1)+W.
        # Actual prefill (M > 8): all tokens arrive simultaneously.
        if M <= 8:
            return self._estimate_decode(M, K, N)
        else:
            return self._estimate_prefill(M, K, N)

    def estimate_weight_cache_pair(
        self, M: int, K: int, N: int
    ) -> MXUResult:
        """Estimate Gate+Up combined with PE dual weight register caching.

        Hardware: each PE has dual weight reg (reg_w0, reg_w1).
        Loads both W_gate and W_up for a (k,n) tile simultaneously,
        computes gate, switches reg (1 cycle), computes up —
        no pipeline drain/fill between.

        This is used for FFN gate/up pairs that share (M,K) dimensions.
        """
        K_tiles = math.ceil(K / self.H)
        N_tiles = math.ceil(N / self.W)
        total_dual_tiles = K_tiles * N_tiles

        # Per dual-tile: 2×H×W weights + 1×H activation (shared)
        dual_weight_bytes = 2 * math.ceil(self.H * self.W * self.w_bits / 8)
        dual_act_bytes = math.ceil(M * self.H * self.a_bits / 8)
        dual_dma = (dual_weight_bytes + dual_act_bytes) / self.eff_bw

        # Compute per dual-tile: gate drain + switch + up drain
        # gate: M+H, switch: 1, up: M+H
        per_matm_drain = M + self.W
        dual_compute = 2 * per_matm_drain + 1  # +1 for weight reg switch

        # Pipeline overhead: fill once per K-tile, drain once per K-tile
        fill = self.H + self.W
        drain = M + self.H

        bottleneck = max(dual_dma, dual_compute)
        first_cold = dual_dma + dual_compute

        if N_tiles >= 2:
            per_Ktile = fill + first_cold + (N_tiles - 1) * bottleneck + drain
        else:
            per_Ktile = fill + first_cold + drain

        total_cycles = int(K_tiles * per_Ktile)

        # Total weight data
        total_weight_bytes = total_dual_tiles * (dual_weight_bytes + dual_act_bytes)
        total_macs = M * K * N * 2  # both matmuls

        ideal_cycles = math.ceil(total_macs / self.macs_per_cycle)
        utilization = ideal_cycles / total_cycles if total_cycles > 0 else 0.0

        return MXUResult(
            compute_cycles=total_cycles,
            stall_cycles_dram=0,
            stall_cycles_sram=0,
            total_cycles=total_cycles,
            utilization=utilization,
            ops=total_macs,
            num_tiles=total_dual_tiles,
            weight_bytes=total_weight_bytes,
        )

    # Backward-compat aliases
    @property
    def array_height(self) -> int:
        return self.H

    @property
    def array_width(self) -> int:
        return self.W

    @property
    def frequency_mhz(self) -> int:
        return self.f_mhz
