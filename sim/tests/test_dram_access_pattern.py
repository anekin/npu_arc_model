"""DRAM access-pattern validation tests.

Validates that the engine/DMA model treats sequential and random accesses
with different efficiencies, that KV cache reads use random efficiency, and
that all eight engine types route the expected AccessType into DMA.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from contracts.units import bandwidth_gbps_to_bytes_per_cycle as _bw2bpc
from engine.mac_engine import MACEngine, create_engine
from models.dma import DMAModel
from models.kv_cache import KVCacheModel
from models.memory_backend import AccessType

# ── Test constants ──────────────────────────────────────────────────────────

_ENGINE_TYPES: list[str] = [
    "systolic",
    "block",
    "os_systolic",
    "input_stationary",
    "tensor_core",
    "wmma",
    "gmma",
    "fsa",
]

_FREQS_MHZ: list[int] = [1000, 2000]

_DEFAULT_DRAM_EFFICIENCY = 0.85
_DEFAULT_RANDOM_EFFICIENCY = 0.50
_DEFAULT_RANDOM_LATENCY = 40


# ── Config builders ─────────────────────────────────────────────────────────


def _dma_config(bandwidth_gbps: float, freq_mhz: int) -> dict[str, Any]:
    """Build a config usable by DMAModel and MACEngine."""
    return {
        "mac_engine": {
            "type": "block",
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": freq_mhz,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
            "ops_per_mac": 2,
        },
        "memory": {
            "type": "LPDDR5",
            "bandwidth_gbps": bandwidth_gbps,
            "dram_efficiency": _DEFAULT_DRAM_EFFICIENCY,
            "dram_efficiency_random_bw": _DEFAULT_RANDOM_EFFICIENCY,
            "random_latency_penalty_cycles": _DEFAULT_RANDOM_LATENCY,
        },
        "dma": {
            "channels": 2,
            "burst_size_bytes": 256,
            "descriptor_overhead_cycles": 5,
            "num_channels": 2,
            "per_channel_fifo_depth": 64,
            "max_burst_length": 8,
            "multi_block_mode": "linked_list",
            "ll_prefetch_en": True,
            "arbitration": "round_robin",
        },
    }


def _engine_config(engine_type: str, bandwidth_gbps: float, freq_mhz: int) -> dict[str, Any]:
    """Build a complete engine config with explicit access-pattern fields."""
    return {
        "mac_engine": {
            "type": engine_type,
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": freq_mhz,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
            "ops_per_mac": 2,
        },
        "memory": {
            "type": "LPDDR5",
            "bandwidth_gbps": bandwidth_gbps,
            "dram_efficiency": _DEFAULT_DRAM_EFFICIENCY,
            "dram_efficiency_random_bw": _DEFAULT_RANDOM_EFFICIENCY,
            "random_latency_penalty_cycles": _DEFAULT_RANDOM_LATENCY,
        },
        "sram": {"l2_shared_kb": 2048},
        "on_chip_memory": {"capacity_gb": 0.0, "bandwidth_gbps": 0.0},
        "kv_cache": {"sram_kb": 256, "dram_region_mb": 96, "precision_bits": 8},
        "dma": {
            "channels": 2,
            "burst_size_bytes": 256,
            "descriptor_overhead_cycles": 5,
            "num_channels": 2,
            "per_channel_fifo_depth": 64,
            "max_burst_length": 8,
            "multi_block_mode": "linked_list",
            "ll_prefetch_en": True,
            "arbitration": "round_robin",
        },
    }


def _kv_config(bandwidth_gbps: float, freq_mhz: int) -> dict[str, Any]:
    """Build a config for KVCacheModel."""
    return {
        "mac_engine": {"frequency_mhz": freq_mhz},
        "memory": {
            "bandwidth_gbps": bandwidth_gbps,
            "dram_efficiency_random_bw": _DEFAULT_RANDOM_EFFICIENCY,
            "random_latency_penalty_cycles": _DEFAULT_RANDOM_LATENCY,
        },
        "kv_cache": {"sram_kb": 256, "dram_region_mb": 96, "precision_bits": 8},
    }


# ── DMAModel pattern-aware transfer tests ───────────────────────────────────


class TestDMAModelAccessPattern:
    """DMAModel.estimate_transfer respects access_type efficiency."""

    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    @pytest.mark.parametrize("bw_gbps", [51.2, 102.4])
    def test_sequential_fewer_cycles_than_random(self, bw_gbps: float, freq_mhz: int) -> None:
        """Sequential access is faster than random for the same bytes."""
        cfg = _dma_config(bw_gbps, freq_mhz)
        dma = DMAModel(cfg)
        size = 256 * 1024

        seq_cycles = dma.estimate_transfer(size, direction="load", access_type=AccessType.SEQUENTIAL)
        rand_cycles = dma.estimate_transfer(size, direction="load", access_type=AccessType.RANDOM)

        assert seq_cycles > 0
        assert rand_cycles > 0
        assert seq_cycles < rand_cycles

    def test_bandwidth_dominant_portion_halves(self) -> None:
        """Doubling bandwidth roughly halves the bandwidth-dominated portion."""
        size = 2 * 1024 * 1024
        low = DMAModel(_dma_config(51.2, 1000))
        high = DMAModel(_dma_config(102.4, 1000))

        low_seq = low.estimate_transfer(size, access_type=AccessType.SEQUENTIAL)
        high_seq = high.estimate_transfer(size, access_type=AccessType.SEQUENTIAL)
        low_rand = low.estimate_transfer(size, access_type=AccessType.RANDOM)
        high_rand = high.estimate_transfer(size, access_type=AccessType.RANDOM)

        # Descriptor and burst overheads are fixed; strip them to isolate BW.
        def bw_only(model: DMAModel, total: int) -> float:
            return float(total - model.descriptor_overhead - math.ceil(size / model.burst_size))

        ratio_seq = bw_only(low, low_seq) / bw_only(high, high_seq)
        ratio_rand = bw_only(low, low_rand) / bw_only(high, high_rand)

        assert 1.7 < ratio_seq < 2.2
        assert 1.7 < ratio_rand < 2.2


# ── MACEngine._dma_cycles pattern-aware tests ───────────────────────────────


class TestMACEngineDMACycles:
    """MACEngine._dma_cycles applies sequential/random efficiency correctly."""

    @pytest.mark.parametrize("engine_type", _ENGINE_TYPES)
    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_sequential_weight_cycles_less_than_random(self, engine_type: str, freq_mhz: int) -> None:
        """Weight-sized sequential transfers are cheaper than random ones."""
        engine = create_engine(_engine_config(engine_type, 51.2, freq_mhz))
        size = 64 * 64 * 4 // 8  # one 64x64 INT4 weight tile

        seq_cycles = engine._dma_cycles(size, AccessType.SEQUENTIAL)
        rand_cycles = engine._dma_cycles(size, AccessType.RANDOM, kv_bytes=size)

        assert seq_cycles > 0
        assert rand_cycles > 0
        assert seq_cycles < rand_cycles

    @pytest.mark.parametrize("engine_type", _ENGINE_TYPES)
    def test_random_hit_skips_latency_penalty(self, engine_type: str) -> None:
        """Random KV hits avoid the fixed random-latency penalty."""
        engine = create_engine(_engine_config(engine_type, 51.2, 1000))
        size = 1024

        hit_cycles = engine._dma_cycles(size, AccessType.RANDOM, kv_bytes=size, is_hit=True)
        miss_cycles = engine._dma_cycles(size, AccessType.RANDOM, kv_bytes=size, is_hit=False)

        assert miss_cycles > hit_cycles


# ── KV cache random-efficiency tests ────────────────────────────────────────


class TestKVCacheRandomEfficiency:
    """KVCacheModel uses dram_efficiency_random_bw and the two-layer miss cost."""

    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_kv_cache_applies_random_efficiency(self, freq_mhz: int) -> None:
        """KV bandwidth is raw BW multiplied by random efficiency, not sequential."""
        bw_gbps = 51.2
        cfg = _kv_config(bw_gbps, freq_mhz)
        kv = KVCacheModel(cfg)

        expected = _bw2bpc(bw_gbps * _DEFAULT_RANDOM_EFFICIENCY, freq_mhz)
        assert kv.bw_bytes_per_cycle == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_hit_has_no_random_latency_penalty(self, freq_mhz: int) -> None:
        """A fully SRAM-resident KV access pays only SRAM cycles."""
        cfg = _kv_config(51.2, freq_mhz)
        kv = KVCacheModel(cfg)
        kv.configure_for_model(num_kv_heads=8, head_dim=64, num_layers=32)

        # 64 previous tokens fit inside the 256KB per-layer SRAM window.
        result = kv.access(token_pos=64, total_tokens=65)
        assert result.hit
        assert result.access_cycles == 64 * kv.sram_access_cycles

    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_miss_adds_random_latency_penalty(self, freq_mhz: int) -> None:
        """DRAM misses include the fixed random-latency penalty."""
        cfg = _kv_config(51.2, freq_mhz)
        kv = KVCacheModel(cfg)
        kv.configure_for_model(num_kv_heads=8, head_dim=64, num_layers=32)

        # 512 previous tokens exceed the SRAM window.
        result = kv.access(token_pos=512, total_tokens=513)
        assert not result.hit
        per_miss = math.ceil(kv._per_layer_kv_bytes / kv.bw_bytes_per_cycle) + kv.random_latency_penalty_cycles
        assert result.access_cycles >= per_miss

    def test_bandwidth_doubles_miss_cost_bandwidth_part(self) -> None:
        """Doubling bandwidth roughly halves the KV bandwidth portion."""
        cfg_low = _kv_config(51.2, 1000)
        cfg_high = _kv_config(102.4, 1000)
        kv_low = KVCacheModel(cfg_low)
        kv_high = KVCacheModel(cfg_high)
        kv_low.configure_for_model(num_kv_heads=8, head_dim=64, num_layers=32)
        kv_high.configure_for_model(num_kv_heads=8, head_dim=64, num_layers=32)

        miss_low = kv_low.access(token_pos=512, total_tokens=513).access_cycles
        miss_high = kv_high.access(token_pos=512, total_tokens=513).access_cycles

        # Latency penalty is fixed; subtract it from both to compare BW parts.
        bw_low = miss_low - kv_low.random_latency_penalty_cycles * (512 - kv_low.max_sram_tokens)
        bw_high = miss_high - kv_high.random_latency_penalty_cycles * (512 - kv_high.max_sram_tokens)

        ratio = bw_low / bw_high
        assert 1.7 < ratio < 2.2


# ── Engine-level access_type routing tests ──────────────────────────────────


def _recorded_dma_call_types(engine: MACEngine, method_name: str, *args: Any, **kwargs: Any) -> list[AccessType]:
    """Call ``method_name`` on ``engine`` and return the AccessType values passed to _dma_cycles."""
    recorded: list[AccessType] = []
    original = engine._dma_cycles

    def wrapper(size: int, access_type: AccessType, *, kv_bytes: int = 0, is_hit: bool = False) -> float:
        if size > 0:
            recorded.append(access_type)
        return original(size, access_type, kv_bytes=kv_bytes, is_hit=is_hit)

    engine._dma_cycles = wrapper  # type: ignore[method-assign]
    getattr(engine, method_name)(*args, **kwargs)
    engine._dma_cycles = original
    return recorded


class TestEngineAccessTypeRouting:
    """All engines pass SEQUENTIAL for weights/activations; FSA attention uses RANDOM for KV."""

    @pytest.mark.parametrize("engine_type", _ENGINE_TYPES)
    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_estimate_uses_sequential_for_weight_and_activation(self, engine_type: str, freq_mhz: int) -> None:
        """Weight/activation DMA paths in estimate() are SEQUENTIAL."""
        engine = create_engine(_engine_config(engine_type, 51.2, freq_mhz))
        calls = _recorded_dma_call_types(engine, "estimate", 8, 64, 64)

        assert len(calls) > 0, f"{engine_type} should call _dma_cycles during estimate()"
        assert all(a == AccessType.SEQUENTIAL for a in calls), (
            f"{engine_type} used non-sequential access: {calls}"
        )

    @pytest.mark.parametrize("engine_type", _ENGINE_TYPES)
    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_weight_cache_pair_uses_sequential(self, engine_type: str, freq_mhz: int) -> None:
        """Weight-cache pair paths are also SEQUENTIAL."""
        engine = create_engine(_engine_config(engine_type, 51.2, freq_mhz))
        calls = _recorded_dma_call_types(engine, "estimate_weight_cache_pair", 8, 64, 64)

        assert len(calls) > 0, f"{engine_type} should call _dma_cycles during estimate_weight_cache_pair()"
        assert all(a == AccessType.SEQUENTIAL for a in calls)

    @pytest.mark.parametrize("freq_mhz", _FREQS_MHZ)
    def test_fsa_attention_kv_uses_random(self, freq_mhz: int) -> None:
        """FSA attention loads K/V with RANDOM and Q with SEQUENTIAL."""
        engine = create_engine(_engine_config("fsa", 51.2, freq_mhz))
        calls = _recorded_dma_call_types(engine, "estimate_attention", 64, 64, 64, 4, 4)

        assert any(a == AccessType.RANDOM for a in calls), "KV DMA in FSA attention must be RANDOM"
        assert any(a == AccessType.SEQUENTIAL for a in calls), "Q DMA in FSA attention must be SEQUENTIAL"


# ── Fail-closed tests ───────────────────────────────────────────────────────


class TestFailClosed:
    """Missing or incorrect access_type must not silently pass."""

    def test_missing_access_type_raises_type_error(self) -> None:
        """MACEngine._dma_cycles requires an explicit access_type."""
        engine = create_engine(_engine_config("systolic", 51.2, 1000))
        with pytest.raises(TypeError):
            engine._dma_cycles(1024)  # type: ignore[call-arg]

    @pytest.mark.parametrize("engine_type", _ENGINE_TYPES)
    def test_random_weight_path_increases_cycles(self, engine_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """If an engine incorrectly routed weights as RANDOM, cycles would increase."""
        engine = create_engine(_engine_config(engine_type, 51.2, 1000))
        baseline = engine.estimate(8, 64, 64).total_cycles

        original = engine._dma_cycles

        def evil_dma(size: int, access_type: AccessType, *, kv_bytes: int = 0, is_hit: bool = False) -> float:
            return original(size, AccessType.RANDOM, kv_bytes=size, is_hit=False)

        monkeypatch.setattr(engine, "_dma_cycles", evil_dma)
        bad = engine.estimate(8, 64, 64).total_cycles

        assert bad >= baseline, f"{engine_type}: incorrect RANDOM routing did not increase cycles"
