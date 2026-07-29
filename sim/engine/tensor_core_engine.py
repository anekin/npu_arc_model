"""Tensor Core 风格引擎 — 64×16×16 子块碎片化模型

参考: NVIDIA A100 Tensor Core

NVIDIA 实际 Tensor Core 以 64×16×16 (K×M×N) 子块处理 GEMM：
  - K 方向累计深度 64，输出块 16×16
  - 一个 128×128 PE 阵列可容 64 个独立 16×16 TC
  - 每个 sub-tile 有 pipeline fill (64+16=80) + 小开销

对 M=1 decode:
  - 子块数 = ceil(K/64) × ceil(N/16) 巨大
  - 每个子块独立 DMA → 事务碎片化
  - 因此 Tensor Core 单 die NPU 下比 Block Engine 慢
"""

import math
from typing import Any, Dict

from engine.mac_engine import MACEngine, EngineResult


class TensorCoreEngine(MACEngine):
    """Tensor Core engine — 64×16×16 sub-tile fragmentation model.

    A 128×128 MAC array is partitioned into 64 independent 16×16 Tensor Cores.
    Each TC processes a (K=64, M=16, N=16) sub-tile per invocation.
    The small sub-tile granularity creates many DMA transactions, which is
    the primary source of fragmentation overhead vs. a monolithic Block Engine.
    """

    SUBTILE_K = 64   # accumulation depth per sub-tile
    SUBTILE_M = 16   # output rows per sub-tile
    SUBTILE_N = 16   # output columns per sub-tile
    SUBTILE_PIPELINE_FILL = 80  # 64 (K) + 16 (N) systolic fill/drain
    SUBTILE_OVERHEAD_CYCLES = 4  # small sync/startup overhead
    DEFAULT_DESCRIPTOR_OVERHEAD_CYCLES = 5

    def _parse_config(self, config: Dict[str, Any]) -> None:
        """Parse common config plus Tensor Core descriptor overhead cycles."""
        super()._parse_config(config)
        overhead = config.get("dma", {}).get(
            "descriptor_overhead_cycles", self.DEFAULT_DESCRIPTOR_OVERHEAD_CYCLES
        )
        if not isinstance(overhead, int) or isinstance(overhead, bool):
            raise ValueError(
                f"dma.descriptor_overhead_cycles must be an integer, got {overhead!r}"
            )
        if overhead < 0:
            raise ValueError(
                f"dma.descriptor_overhead_cycles must be >= 0, got {overhead}"
            )
        self.descriptor_overhead_cycles = overhead

    @property
    def engine_type(self) -> str:
        return "tensor_core"

    @property
    def num_tcs(self) -> int:
        """Number of independent 16×16 Tensor Cores in the array."""
        return (self.H * self.W) // (self.SUBTILE_M * self.SUBTILE_N)

    @property
    def per_subtile_compute(self) -> int:
        """Cycles to execute one 64×16×16 sub-tile on a TC."""
        return self.SUBTILE_PIPELINE_FILL + self.SUBTILE_OVERHEAD_CYCLES

    def estimate(self, M: int, K: int, N: int,
                 weight_preloaded: bool = False) -> EngineResult:
        """TC-style GEMM estimate with 64×16×16 sub-tile fragmentation."""
        sub_K = math.ceil(K / self.SUBTILE_K)
        sub_M = math.ceil(M / self.SUBTILE_M)
        sub_N = math.ceil(N / self.SUBTILE_N)
        total_invocations = sub_K * sub_M * sub_N

        # Per-sub-tile data movement
        tile_weight_bytes = math.ceil(
            self.SUBTILE_K * self.SUBTILE_N * self.w_bits / 8
        )
        M_eff = min(M, self.SUBTILE_M)
        tile_act_bytes = math.ceil(
            M_eff * self.SUBTILE_K * self.a_bits / 8
        )

        num_tcs = self.num_tcs
        waves = math.ceil(total_invocations / num_tcs)

        # Per wave: all TCs fire in parallel and share the DRAM bus
        per_wave_compute = self.per_subtile_compute

        per_wave_payload_cycles = (
            num_tcs * (tile_weight_bytes + tile_act_bytes) / self.eff_bw
        )

        descriptor_cycles_per_wave = num_tcs * self.descriptor_overhead_cycles

        invocations_last = total_invocations - num_tcs * (waves - 1)
        active_tcs = invocations_last
        last_wave_descriptor = active_tcs * self.descriptor_overhead_cycles
        total_descriptor_cycles = (
            descriptor_cycles_per_wave * (waves - 1) + last_wave_descriptor
        )

        per_wave_dma = per_wave_payload_cycles + descriptor_cycles_per_wave

        # Double-buffering: overlap DMA of wave N+1 with compute of wave N
        bottleneck = max(per_wave_compute, per_wave_dma)

        if waves == 1:
            last_wave_payload = active_tcs * (tile_weight_bytes + tile_act_bytes) / self.eff_bw
            total = int(last_wave_payload + last_wave_descriptor + per_wave_compute)
        elif active_tcs == num_tcs:
            first_cold = per_wave_dma + per_wave_compute
            total = int(first_cold + (waves - 1) * bottleneck)
        else:
            last_wave_payload = active_tcs * (tile_weight_bytes + tile_act_bytes) / self.eff_bw
            last_wave_dma_total = last_wave_payload + last_wave_descriptor
            total = int(
                per_wave_dma + per_wave_compute
                + (waves - 2) * bottleneck
                + max(per_wave_compute, last_wave_dma_total)
            )

        total_macs = M * K * N
        total_weight_bytes = total_invocations * (tile_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        compute_cycles = waves * per_wave_compute
        dma_cycles = total - compute_cycles
        raw_dma = (K * N * self.w_bits // 8 + M * K * self.a_bits // 8)
        raw_dma_cycles = math.ceil(raw_dma / self.eff_bw) if self.eff_bw > 0 else 0

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total,
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_cycles,
            num_tiles=total_invocations,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if per_wave_compute > per_wave_dma else "dma",
            details={
                "sub_K": sub_K,
                "sub_N": sub_N,
                "sub_M": sub_M,
                "total_invocations": total_invocations,
                "num_tcs": num_tcs,
                "waves": waves,
                "active_tcs": active_tcs,
                "num_waves": waves,
                "per_wave_payload_cycles": round(per_wave_payload_cycles, 1),
                "per_wave_dma": round(per_wave_dma, 1),
                "per_wave_compute": per_wave_compute,
                "per_subtile_compute": per_wave_compute,
                "descriptor_cycles_per_wave": descriptor_cycles_per_wave,
                "total_descriptor_cycles": total_descriptor_cycles,
                "subtile_size": f"{self.SUBTILE_K}×{self.SUBTILE_M}×{self.SUBTILE_N}",
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up with Tensor Core weight cache.

        TC can keep both gate and up weights in registers while reusing the
        same activation slice for a given K sub-tile.  This saves one activation
        load per sub-tile vs. two separate GEMMs.
        """
        sub_K = math.ceil(K / self.SUBTILE_K)
        sub_M = math.ceil(M / self.SUBTILE_M)
        sub_N = math.ceil(N / self.SUBTILE_N)
        total_invocations = sub_K * sub_M * sub_N

        tile_weight_bytes = math.ceil(
            self.SUBTILE_K * self.SUBTILE_N * self.w_bits / 8
        )
        M_eff = min(M, self.SUBTILE_M)
        tile_act_bytes = math.ceil(
            M_eff * self.SUBTILE_K * self.a_bits / 8
        )

        # Load both gate+up weights, one activation slice per K sub-tile
        dual_weight_bytes = 2 * tile_weight_bytes
        per_subtile_payload = (dual_weight_bytes + tile_act_bytes) / self.eff_bw

        # Compute gate then up sequentially on the same TC
        per_subtile_compute = 2 * self.per_subtile_compute

        num_tcs = self.num_tcs
        waves = math.ceil(total_invocations / num_tcs)

        per_wave_payload_cycles = num_tcs * per_subtile_payload
        per_wave_compute = per_subtile_compute

        descriptor_cycles_per_wave = num_tcs * self.descriptor_overhead_cycles

        invocations_last = total_invocations - num_tcs * (waves - 1)
        active_tcs = invocations_last
        last_wave_descriptor = active_tcs * self.descriptor_overhead_cycles
        total_descriptor_cycles = (
            descriptor_cycles_per_wave * (waves - 1) + last_wave_descriptor
        )

        per_wave_dma = per_wave_payload_cycles + descriptor_cycles_per_wave

        bottleneck = max(per_wave_compute, per_wave_dma)

        if waves == 1:
            last_wave_payload = active_tcs * per_subtile_payload
            total = int(last_wave_payload + last_wave_descriptor + per_wave_compute)
        elif active_tcs == num_tcs:
            first_cold = per_wave_dma + per_wave_compute
            total = int(first_cold + (waves - 1) * bottleneck)
        else:
            last_wave_payload = active_tcs * per_subtile_payload
            last_wave_dma_total = last_wave_payload + last_wave_descriptor
            total = int(
                per_wave_dma + per_wave_compute
                + (waves - 2) * bottleneck
                + max(per_wave_compute, last_wave_dma_total)
            )

        total_macs = M * K * N * 2
        total_weight_bytes = total_invocations * (dual_weight_bytes + tile_act_bytes)
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        # Activation sharing savings vs. two separate estimates
        activation_savings = total_invocations * tile_act_bytes / self.eff_bw

        compute_cycles = waves * per_wave_compute
        dma_cycles = total - compute_cycles
        raw_dma = (K * N * self.w_bits // 8 + M * K * self.a_bits // 8) * 2
        raw_dma_cycles = math.ceil(raw_dma / self.eff_bw) if self.eff_bw > 0 else 0

        return EngineResult(
            compute_cycles=compute_cycles,
            dma_cycles=dma_cycles,
            total_cycles=total,
            utilization=util,
            mac_count=total_macs,
            op_count=total_macs * 2,
            ideal_compute_cycles=ideal,
            raw_dma_cycles=raw_dma_cycles,
            num_tiles=total_invocations,
            weight_bytes=total_weight_bytes,
            bottleneck="compute" if per_wave_compute > per_wave_dma else "dma",
            details={
                "sub_K": sub_K,
                "sub_N": sub_N,
                "sub_M": sub_M,
                "total_invocations": total_invocations,
                "num_tcs": num_tcs,
                "waves": waves,
                "active_tcs": active_tcs,
                "num_waves": waves,
                "per_wave_payload_cycles": round(per_wave_payload_cycles, 1),
                "per_wave_dma": round(per_wave_dma, 1),
                "per_wave_compute": per_wave_compute,
                "descriptor_cycles_per_wave": descriptor_cycles_per_wave,
                "total_descriptor_cycles": total_descriptor_cycles,
                "weight_cache_savings": int(activation_savings),
            },
        )
