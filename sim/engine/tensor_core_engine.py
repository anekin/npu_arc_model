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
        """TC-style GEMM estimate with 64x16x16 sub-tile fragmentation."""
        sub_K = math.ceil(K / self.SUBTILE_K)
        sub_M = math.ceil(M / self.SUBTILE_M)
        sub_N = math.ceil(N / self.SUBTILE_N)
        total_invocations = sub_K * sub_M * sub_N

        last_k = K - (sub_K - 1) * self.SUBTILE_K
        if last_k <= 0:
            last_k = self.SUBTILE_K
        last_m = M - (sub_M - 1) * self.SUBTILE_M
        if last_m <= 0:
            last_m = self.SUBTILE_M
        last_n = N - (sub_N - 1) * self.SUBTILE_N
        if last_n <= 0:
            last_n = self.SUBTILE_N

        full_weight_bytes = math.ceil(
            self.SUBTILE_K * self.SUBTILE_N * self.w_bits / 8
        )
        full_act_bytes = math.ceil(
            self.SUBTILE_M * self.SUBTILE_K * self.a_bits / 8
        )
        full_payload = (full_weight_bytes + full_act_bytes) / self.eff_bw

        num_tcs = self.num_tcs
        waves = math.ceil(total_invocations / num_tcs)

        per_wave_compute = self.per_subtile_compute
        per_wave_payload_cycles = num_tcs * full_payload
        descriptor_cycles_per_wave = num_tcs * self.descriptor_overhead_cycles

        invocations_last = total_invocations - num_tcs * (waves - 1)
        active_tcs = invocations_last

        last_wave_weight_bytes = 0
        last_wave_act_bytes = 0
        for i in range(invocations_last):
            global_idx = num_tcs * (waves - 1) + i
            n_idx = global_idx % sub_N
            k_idx = (global_idx // sub_N) % sub_K
            m_idx = global_idx // (sub_N * sub_K)
            m_eff = last_m if m_idx == sub_M - 1 else self.SUBTILE_M
            k_eff = last_k if k_idx == sub_K - 1 else self.SUBTILE_K
            n_eff = last_n if n_idx == sub_N - 1 else self.SUBTILE_N
            last_wave_weight_bytes += math.ceil(
                k_eff * n_eff * self.w_bits / 8
            )
            last_wave_act_bytes += math.ceil(
                m_eff * k_eff * self.a_bits / 8
            )

        last_wave_payload = (last_wave_weight_bytes + last_wave_act_bytes) / self.eff_bw
        last_wave_descriptor = active_tcs * self.descriptor_overhead_cycles
        total_descriptor_cycles = (
            descriptor_cycles_per_wave * (waves - 1) + last_wave_descriptor
        )

        per_wave_dma = per_wave_payload_cycles + descriptor_cycles_per_wave
        bottleneck = max(per_wave_compute, per_wave_dma)

        if waves == 1:
            total = int(last_wave_payload + last_wave_descriptor + per_wave_compute)
        elif active_tcs == num_tcs:
            first_cold = per_wave_dma + per_wave_compute
            total = int(first_cold + (waves - 1) * bottleneck)
        else:
            last_wave_dma_total = last_wave_payload + last_wave_descriptor
            total = int(
                per_wave_dma + per_wave_compute
                + (waves - 2) * bottleneck
                + max(per_wave_compute, last_wave_dma_total)
            )

        total_macs = M * K * N
        total_weight_bytes = (
            (waves - 1) * num_tcs * (full_weight_bytes + full_act_bytes)
            + last_wave_weight_bytes + last_wave_act_bytes
        )
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
                "subtile_size": f"{self.SUBTILE_K}x{self.SUBTILE_M}x{self.SUBTILE_N}",
            },
        )

    def estimate_weight_cache_pair(self, M: int, K: int, N: int) -> EngineResult:
        """Gate+Up with TC weight cache — partial-tile-aware."""
        sub_K = math.ceil(K / self.SUBTILE_K)
        sub_M = math.ceil(M / self.SUBTILE_M)
        sub_N = math.ceil(N / self.SUBTILE_N)
        total_invocations = sub_K * sub_M * sub_N

        last_k = K - (sub_K - 1) * self.SUBTILE_K
        if last_k <= 0:
            last_k = self.SUBTILE_K
        last_m = M - (sub_M - 1) * self.SUBTILE_M
        if last_m <= 0:
            last_m = self.SUBTILE_M
        last_n = N - (sub_N - 1) * self.SUBTILE_N
        if last_n <= 0:
            last_n = self.SUBTILE_N

        full_weight_bytes = math.ceil(
            self.SUBTILE_K * self.SUBTILE_N * self.w_bits / 8
        )
        full_act_bytes = math.ceil(
            self.SUBTILE_M * self.SUBTILE_K * self.a_bits / 8
        )
        full_payload = (2 * full_weight_bytes + full_act_bytes) / self.eff_bw

        num_tcs = self.num_tcs
        waves = math.ceil(total_invocations / num_tcs)

        per_wave_compute = 2 * self.per_subtile_compute
        per_wave_payload_cycles = num_tcs * full_payload
        descriptor_cycles_per_wave = num_tcs * self.descriptor_overhead_cycles

        invocations_last = total_invocations - num_tcs * (waves - 1)
        active_tcs = invocations_last

        last_wave_weight_bytes = 0
        last_wave_act_bytes = 0
        for i in range(invocations_last):
            global_idx = num_tcs * (waves - 1) + i
            n_idx = global_idx % sub_N
            k_idx = (global_idx // sub_N) % sub_K
            m_idx = global_idx // (sub_N * sub_K)
            m_eff = last_m if m_idx == sub_M - 1 else self.SUBTILE_M
            k_eff = last_k if k_idx == sub_K - 1 else self.SUBTILE_K
            n_eff = last_n if n_idx == sub_N - 1 else self.SUBTILE_N
            last_wave_weight_bytes += math.ceil(
                k_eff * n_eff * self.w_bits / 8
            )
            last_wave_act_bytes += math.ceil(
                m_eff * k_eff * self.a_bits / 8
            )

        last_wave_payload = (2 * last_wave_weight_bytes + last_wave_act_bytes) / self.eff_bw
        last_wave_descriptor = active_tcs * self.descriptor_overhead_cycles
        total_descriptor_cycles = (
            descriptor_cycles_per_wave * (waves - 1) + last_wave_descriptor
        )

        per_wave_dma = per_wave_payload_cycles + descriptor_cycles_per_wave
        bottleneck = max(per_wave_compute, per_wave_dma)

        if waves == 1:
            total = int(last_wave_payload + last_wave_descriptor + per_wave_compute)
        elif active_tcs == num_tcs:
            first_cold = per_wave_dma + per_wave_compute
            total = int(first_cold + (waves - 1) * bottleneck)
        else:
            last_wave_dma_total = last_wave_payload + last_wave_descriptor
            total = int(
                per_wave_dma + per_wave_compute
                + (waves - 2) * bottleneck
                + max(per_wave_compute, last_wave_dma_total)
            )

        total_macs = M * K * N * 2
        total_weight_bytes = (
            (waves - 1) * num_tcs * (2 * full_weight_bytes + full_act_bytes)
            + 2 * last_wave_weight_bytes + last_wave_act_bytes
        )
        ideal = math.ceil(total_macs / self.peak_macs_per_cycle)
        util = ideal / total if total > 0 else 0.0

        activation_savings = total_invocations * full_act_bytes / self.eff_bw

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
