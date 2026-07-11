"""KV Cache Manager 性能模型 — SRAM 命中率 + DRAM 访问延迟"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class KVCacheResult:
    hit: bool          # cache hit?
    access_cycles: int # actual access time
    sram_hit_rate: float  # overall hit rate


class KVCacheModel:
    """Per-layer KV cache with sliding window SRAM + LPDDR5 backing.

    Key insight (from the design doc): caching ALL layers' KV in 256KB SRAM
    gives only ~9 tokens. Caching PER-LAYER gives ~512 tokens — >85% hit rate.

    Strategy:
    - SRAM 256KB holds KV for the CURRENT LAYER only
    - Sliding window: most recent N tokens stay in SRAM
    - Older tokens in LPDDR5 KV region
    - On layer switch: SRAM is reloaded with that layer's KV window
    """

    def __init__(self, config: Dict[str, Any]):
        kv = config["kv_cache"]
        self.sram_bytes = int(kv["sram_kb"]) * 1024       # 256 KB
        self.dram_region_mb = int(kv["dram_region_mb"])    # 96 MB
        self.element_bits = int(kv["precision_bits"])       # 8 (INT8)
        self.element_bytes = self.element_bits // 8          # 1

        mem = config["memory"]
        self.bw_bytes_per_cycle = float(mem["bandwidth_bytes_per_cycle"])

        # Timing parameters
        # SRAM access: ~2 cycles (1 read + 1 write port)
        self.sram_access_cycles = 2
        # DRAM access: row open + CAS + data transfer
        self.dram_access_cycles = 80  # tRC ≈ 48ns @ 1GHz → ~48 cycles + overhead

        # Current state (reset per layer)
        self.total_tokens = 0
        self.sram_tokens = 0
        self._per_layer_kv_bytes = 0

    def configure_for_model(self, num_kv_heads: int, head_dim: int,
                            num_layers: int, max_context: int = 2048):
        """Set model-specific parameters.

        KV per token per layer = num_kv_heads × head_dim × 2(K+V) × element_bytes
        """
        kv_per_token_per_layer = num_kv_heads * head_dim * 2 * self.element_bytes
        self._per_layer_kv_bytes = kv_per_token_per_layer

        # How many tokens fit in SRAM per layer?
        self.max_sram_tokens = self.sram_bytes // kv_per_token_per_layer

        # How many tokens fit in total DRAM region?
        total_kv_per_token = kv_per_token_per_layer * num_layers
        self.max_total_tokens = min(
            (self.dram_region_mb * 1024 * 1024) // total_kv_per_token,
            max_context,
        )
        self.num_layers = num_layers

    def access(self, token_pos: int, total_tokens: int) -> KVCacheResult:
        """Estimate KV cache access latency for one layer at token_pos.

        During decode: each new token needs KV of ALL previous tokens for attention.
        The model predicts which tokens hit SRAM vs DRAM.

        Args:
            token_pos: current token position (0-indexed)
            total_tokens: total tokens in context so far (including current)

        Returns:
            KVCacheResult with hit/miss and cycle count
        """
        # How many previous tokens' KV do we need to access?
        # For causal attention: token_pos previous tokens
        num_kv_entries = token_pos

        if num_kv_entries <= 0:
            return KVCacheResult(hit=True, access_cycles=0, sram_hit_rate=1.0)

        # SRAM holds most recent N tokens for this layer
        sram_window = min(self.max_sram_tokens, num_kv_entries)
        sram_hits = sram_window  # all window tokens hit
        dram_misses = max(0, num_kv_entries - sram_window)

        hit_rate = sram_hits / num_kv_entries if num_kv_entries > 0 else 1.0

        # Access time: hit → SRAM, miss → DRAM
        sram_time = sram_hits * self.sram_access_cycles
        dram_time = dram_misses * self.dram_access_cycles

        total_cycles = sram_time + dram_time

        # Amortize over the token — we're accessing many KVs in parallel
        # during attention score computation. The MXU already handles
        # the matmul; KV access happens concurrently.
        # Return the per-token amortized cost.
        # In practice, KV access is pipelined and hidden behind MXU.

        return KVCacheResult(
            hit=(dram_misses == 0),
            access_cycles=int(total_cycles),
            sram_hit_rate=hit_rate,
        )

    def estimate_per_decode(self, token_pos: int, total_tokens: int) -> int:
        """Per-decode-step KV access cycles, amortized.

        During decode (token by token), we access KV of all previous tokens
        for attention. But this is heavily pipelined with MXU.

        Returns amortized cycles per layer.
        """
        result = self.access(token_pos, total_tokens)

        # Amortize: if hit_rate > 90%, overhead is negligible
        # If many DRAM misses, add stall penalty
        if result.sram_hit_rate >= 0.85:
            return self.sram_access_cycles * 10  # ~20 cycles, negligible
        else:
            # Penalty proportional to miss rate
            miss_rate = 1.0 - result.sram_hit_rate
            return int(result.access_cycles * miss_rate * 0.1)  # partial stall


    def layer_switch_cost(self) -> int:
        """Cost to reload SRAM with a new layer's KV window.

        When switching layers, the SRAM must be repopulated with
        the new layer's KV entries. This is a DMA operation.
        """
        # Load SRAM's worth of KV: sram_bytes at DRAM bandwidth
        # But we can pipeline this with the layer computation
        load_bytes = self.sram_bytes
        raw_cycles = load_bytes / self.bw_bytes_per_cycle
        # Assume 70% hidden behind MXU
        return int(raw_cycles * 0.3)
