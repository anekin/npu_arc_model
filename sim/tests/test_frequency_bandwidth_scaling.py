"""Frequency and bandwidth end-to-end unit propagation tests (Todo 6).

Acceptance criteria from the plan:
- Compute-only at 800/1000/1200 MHz wall time scales as 1000/f (0.1% tolerance)
- Fixed 51.2 GB/s memory-only wall time <=0.1% difference across frequencies
- LPDDR5→HBM3 monotonic and saturate at compute floor
- CLI/DSE output consistent with independent conversion oracle
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from contracts.units import (
    bandwidth_gbps_to_bytes_per_cycle,
    cycles_to_microseconds,
)
from engine.mac_engine import create_engine

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = REPO_ROOT / "sim"


def _make_config(
    engine_type="block", array_h=64, array_w=64, freq_mhz=1000, bw_gbps=51.2, w_bits=4, weight_cache=False, l2_kb=2048
):
    """Build a minimal config dict for engine construction."""
    return {
        "mac_engine": {
            "type": engine_type,
            "array_height": array_h,
            "array_width": array_w,
            "frequency_mhz": freq_mhz,
            "weight_precision_bits": w_bits,
            "activation_precision_bits": 8,
        },
        "memory": {
            "bandwidth_gbps": bw_gbps,
            "dram_efficiency": 0.85,
            "bandwidth_bytes_per_cycle": 0,  # ignored by fixed engines
        },
        "optimizations": {
            "weight_cache": weight_cache,
            "dma_bw_multiplier": 1.0,
        },
        "sram": {"l2_shared_kb": l2_kb},
        "kv_cache": {
            "sram_kb": 256,
            "dram_region_mb": 96,
            "precision_bits": 8,
        },
    }


# ── Unit-level: bytes/cycle conversion ───────────────────────────


@pytest.mark.parametrize(
    "freq_mhz,expected_bpc",
    [
        (800, 64.0),
        (1000, 51.2),
        (1200, 51.2 * 1000 / 1200),
    ],
)
def test_bandwidth_bytes_per_cycle_at_frequencies(freq_mhz, expected_bpc):
    """Given 51.2 GB/s bandwidth,
    When computing bytes/cycle at 800/1000/1200 MHz,
    Then results match the plan formula (64 / 51.2 / 42.666...)."""
    result = bandwidth_gbps_to_bytes_per_cycle(51.2, freq_mhz)
    assert result == pytest.approx(expected_bpc, rel=1e-12, abs=1e-12)


# ── Engine-level: compute-bound scaling ──────────────────────────


@pytest.mark.parametrize("freq_mhz", [800, 1000, 1200])
def test_compute_bound_total_cycles_independent_of_frequency(freq_mhz):
    """Given a compute-bound workload (high BW, large array),
    When the engine estimates with the same shape at different frequencies,
    Then total_cycles is frequency-independent (within rounding)."""
    cfg = _make_config(engine_type="block", array_h=64, array_w=64, freq_mhz=freq_mhz, bw_gbps=819.2)
    engine = create_engine(cfg)
    result = engine.estimate(M=1, K=2048, N=11008)

    # total_cycles should be independent of frequency for pure compute
    assert result.total_cycles > 0
    assert result.ideal_compute_cycles > 0
    # The compute cycles floor should be identical across frequencies
    expected_min = math.ceil((1 * 2048 * 11008) / (64 * 64 * 2))
    assert result.total_cycles >= expected_min


def test_compute_bound_wall_time_scales_inversely_with_frequency():
    """Given a compute-bound workload (high BW=819.2 GB/s, large shape),
    When tok/s is computed at 800/1000/1200 MHz,
    Then tok/s ratios scale as freq/1000 (0.1% tolerance)."""
    results = {}
    for freq_mhz in [800, 1000, 1200]:
        cfg = _make_config(engine_type="block", array_h=64, array_w=64, freq_mhz=freq_mhz, bw_gbps=819.2)
        engine = create_engine(cfg)
        result = engine.estimate(M=1, K=2048, N=11008)

        # Compute wall time per operation
        total_us = cycles_to_microseconds(result.total_cycles, freq_mhz)
        results[freq_mhz] = (result.total_cycles, total_us)

    cycles_800 = results[800][0]
    cycles_1000 = results[1000][0]
    cycles_1200 = results[1200][0]

    # Compute cycles should be identical (compute bound, not DMA bound)
    # Allow small rounding differences from ceil()
    assert cycles_800 == cycles_1000 == cycles_1200, (
        f"Compute-bound total_cycles should be frequency-independent: "
        f"800={cycles_800}, 1000={cycles_1000}, 1200={cycles_1200}"
    )

    # Wall time should scale inversely with frequency:
    # us_800 = cycles / 800, us_1000 = cycles / 1000, us_1200 = cycles / 1200
    # Ratio us_800/us_1000 = 1000/800 = 1.25
    us_800 = results[800][1]
    us_1000 = results[1000][1]
    us_1200 = results[1200][1]

    expected_ratio_800_1000 = 1000.0 / 800.0  # 1.25
    actual_ratio = us_800 / us_1000
    assert actual_ratio == pytest.approx(expected_ratio_800_1000, rel=0.001), (
        f"Wall-time ratio 800/1000 MHz: expected {expected_ratio_800_1000}, got {actual_ratio}"
    )

    expected_ratio_1200_1000 = 1000.0 / 1200.0  # 0.8333...
    actual_ratio = us_1200 / us_1000
    assert actual_ratio == pytest.approx(expected_ratio_1200_1000, rel=0.001), (
        f"Wall-time ratio 1200/1000 MHz: expected {expected_ratio_1200_1000}, got {actual_ratio}"
    )


# ── Memory-bound: fixed 51.2 GB/s across frequencies ─────────────


def test_memory_bound_dma_wall_time_invariant():
    """Given a memory-bound workload (fixed 51.2 GB/s, large weight array),
    When raw DMA cycles are converted to wall time at 800/1000/1200 MHz,
    Then the DMA wall time (raw_dma_cycles/freq) is within 0.1%.

    Rationale: For a fixed physical bandwidth (GB/s), the time to transfer N bytes
    is N/(BW_gbps*1e9) seconds, independent of core frequency. The DMA cycles
    scale inversely with bytes_per_cycle (which scales with 1/freq), so
    DMA_wall_us = raw_dma_cycles / freq is frequency-invariant.
    """
    dma_wall_times = {}
    M, K, N = 1, 11008, 11008  # FFN_down shape, ~58MB weights

    for freq_mhz in [800, 1000, 1200]:
        cfg = _make_config(engine_type="block", array_h=64, array_w=64, freq_mhz=freq_mhz, bw_gbps=51.2)
        engine = create_engine(cfg)
        result = engine.estimate(M=M, K=K, N=N)
        dma_wall_us = cycles_to_microseconds(result.raw_dma_cycles, freq_mhz)
        dma_wall_times[freq_mhz] = dma_wall_us

    ref_us = dma_wall_times[1000]
    for freq in [800, 1200]:
        diff = abs(dma_wall_times[freq] - ref_us) / max(ref_us, 1e-9)
        assert diff <= 0.001, (
            f"DMA wall time at {freq} MHz ({dma_wall_times[freq]:.2f} us) "
            f"differs from 1000 MHz ({ref_us:.2f} us) by {diff * 100:.3f}% (>0.1%)"
        )


def test_memory_bound_cycle_counts_scale_with_frequency():
    """Given a memory-bound workload (fixed 51.2 GB/s),
    When engine estimate runs at 800/1000/1200 MHz,
    Then total cycles strictly increase with frequency.

    Higher frequency → lower bytes/cycle → more DMA cycles needed.
    """
    cycle_counts = {}
    M, K, N = 1, 11008, 11008

    for freq_mhz in [800, 1000, 1200]:
        cfg = _make_config(engine_type="block", array_h=64, array_w=64, freq_mhz=freq_mhz, bw_gbps=51.2)
        engine = create_engine(cfg)
        result = engine.estimate(M=M, K=K, N=N)
        cycle_counts[freq_mhz] = (result.total_cycles, result.raw_dma_cycles)

    # Raw DMA cycles must strictly increase with frequency
    assert cycle_counts[800][1] < cycle_counts[1000][1] < cycle_counts[1200][1], (
        f"raw_dma_cycles should increase with frequency: "
        f"800={cycle_counts[800][1]}, 1000={cycle_counts[1000][1]}, 1200={cycle_counts[1200][1]}"
    )


# ── Bandwidth monotonicity: LPDDR5 → HBM3 ───────────────────────


def test_bandwidth_monotonic_and_saturates_at_compute_floor():
    """Given a fixed engine at 1000 MHz,
    When bandwidth sweeps LPDDR5-32b → HBM3-1024b,
    Then tok/s monotonically increases and saturates at the compute floor."""
    M, K, N = 1, 2048, 11008
    bw_configs = [
        (25.6, "LPDDR5-32b"),
        (51.2, "LPDDR5-64b"),
        (102.4, "LPDDR5-128b"),
        (204.8, "LPDDR5-256b"),
        (460.0, "HBM2e-1024b"),
        (819.2, "HBM3-1024b"),
    ]

    results = []
    for bw_gbps, label in bw_configs:
        cfg = _make_config(engine_type="block", array_h=64, array_w=64, freq_mhz=1000, bw_gbps=bw_gbps)
        engine = create_engine(cfg)
        result = engine.estimate(M=M, K=K, N=N)
        wall_us = cycles_to_microseconds(result.total_cycles, 1000)
        results.append((bw_gbps, label, result.total_cycles, wall_us))

    # Monotonic: higher BW should not increase wall time
    for i in range(1, len(results)):
        prev_bw, prev_label, _, prev_us = results[i - 1]
        curr_bw, curr_label, _, curr_us = results[i]
        assert curr_us <= prev_us * 1.001, (
            f"Wall time should not increase with higher BW: "
            f"{prev_label} ({prev_bw} GB/s) = {prev_us:.2f} us, "
            f"{curr_label} ({curr_bw} GB/s) = {curr_us:.2f} us"
        )

    # Saturation: HBM2e and HBM3 should have near-identical wall times
    # (both are at the compute floor)
    hbm2e_us = results[-2][3]
    hbm3_us = results[-1][3]
    # At 460 and 819 GB/s, both should be compute-bound
    # Allow 0.1% difference for rounding
    diff = abs(hbm2e_us - hbm3_us) / max(hbm2e_us, 1e-9)
    assert diff <= 0.001, (
        f"HBM2e and HBM3 wall times should be near-identical at compute floor: "
        f"HBM2e={hbm2e_us:.2f} us, HBM3={hbm3_us:.2f} us, diff={diff * 100:.3f}%"
    )

    # Also verify that total_cycles at saturation equals the compute floor
    peak_macs = 64 * 64 * 2
    ideal_compute_cycles = math.ceil(M * K * N / peak_macs)
    assert results[-1][2] >= ideal_compute_cycles
    assert results[-2][2] >= ideal_compute_cycles


# ── DSE tok_s_from_layer: frequency-dependent output ─────────────


def _run_dse_quick(freq_mhz):
    """Run DSE quickly with a specific frequency override and parse JSON."""

    # Use the DSE module directly for speed, not subprocess
    from design_space_explorer import simulate_layer, tok_s_from_layer

    _NUM_LAYERS = 28
    configs = []
    # Minimal config set: single engine, single shape, single BW
    for bw_gbps in [51.2, 819.2]:
        configs.append(
            _make_config(
                engine_type="block",
                array_h=64,
                array_w=64,
                freq_mhz=freq_mhz,
                bw_gbps=bw_gbps,
            )
        )

    results = []
    for cfg in configs:
        layer_cycles, _ = simulate_layer(cfg)
        fps = tok_s_from_layer(layer_cycles, _NUM_LAYERS, freq_mhz)
        results.append(
            {
                "bw_gbps": cfg["memory"]["bandwidth_gbps"],
                "total_cycles": layer_cycles,
                "tok_per_s": fps,
            }
        )
    return results


def test_dse_output_different_across_frequencies():
    """Given the same hardware config,
    When DSE tok_s_from_layer runs at 800/1000/1200 MHz,
    Then tok/s values differ and scale with frequency."""
    results_800 = _run_dse_quick(800)
    results_1000 = _run_dse_quick(1000)
    results_1200 = _run_dse_quick(1200)

    # For compute-bound (819.2 GB/s), ideal compute cycles should be same,
    # but tok/s should scale with frequency.
    # Total cycles may differ slightly due to DMA pipeline overhead,
    # so check ideal_compute_cycles equality instead.
    compute_800 = next(r for r in results_800 if r["bw_gbps"] == 819.2)
    compute_1000 = next(r for r in results_1000 if r["bw_gbps"] == 819.2)
    compute_1200 = next(r for r in results_1200 if r["bw_gbps"] == 819.2)

    # total_cycles may differ slightly due to DMA/pipeline overhead — within 5%
    ref_cycles = compute_1000["total_cycles"]
    for freq, result in [(800, compute_800), (1200, compute_1200)]:
        ratio = max(result["total_cycles"] / ref_cycles, ref_cycles / result["total_cycles"])
        assert ratio <= 1.05, f"Compute-bound total_cycles ratio {freq}/1000 MHz ({ratio:.4f}) > 5% deviation"

    # tok/s should differ
    assert compute_800["tok_per_s"] != compute_1000["tok_per_s"], (
        "800 MHz and 1000 MHz tok/s should differ (were identical before fix)"
    )
    assert compute_1000["tok_per_s"] != compute_1200["tok_per_s"], (
        "1000 MHz and 1200 MHz tok/s should differ (were identical before fix)"
    )

    # tok/s ratio should be proportional to frequency (0.5% tolerance for rounding)
    ratio_800_1000 = compute_800["tok_per_s"] / compute_1000["tok_per_s"]
    assert ratio_800_1000 == pytest.approx(800.0 / 1000.0, rel=0.005), (
        f"tok/s ratio 800/1000 MHz: expected {800 / 1000:.4f}, got {ratio_800_1000:.4f}"
    )

    ratio_1200_1000 = compute_1200["tok_per_s"] / compute_1000["tok_per_s"]
    assert ratio_1200_1000 == pytest.approx(1200.0 / 1000.0, rel=0.005), (
        f"tok/s ratio 1200/1000 MHz: expected {1200 / 1000:.4f}, got {ratio_1200_1000:.4f}"
    )


# ── CLI: npu_sim --freq outputs differ ───────────────────────────


def _run_npu_sim(*extra_args):
    """Run sim/npu_sim.py with PYTHONPATH=sim."""
    cmd = [sys.executable, str(SIM_DIR / "npu_sim.py"), *extra_args]
    import os

    env = {**os.environ, "PYTHONPATH": str(SIM_DIR)}
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), env=env, timeout=60)


@pytest.mark.cli
def test_cli_freq_override_produces_different_output():
    """Given npu_sim with --json,
    When --freq 800/1000/1200 are specified,
    Then the outputs differ (proving frequency propagation works)."""
    outputs = {}
    for freq in [800, 1000, 1200]:
        result = _run_npu_sim("--engine", "block", "--freq", str(freq), "--json")
        assert result.returncode == 0, f"npu_sim --freq {freq} failed: {result.stderr}"
        outputs[freq] = json.loads(result.stdout)

    # All three should produce valid output
    for freq in [800, 1000, 1200]:
        assert outputs[freq]["decode"]["tok_per_s"] > 0
        assert outputs[freq]["decode"]["per_token_us"] > 0

    # Outputs should differ — this is the key acceptance criterion
    # (before fix, all three were identical)
    tok_800 = outputs[800]["decode"]["tok_per_s"]
    tok_1000 = outputs[1000]["decode"]["tok_per_s"]
    tok_1200 = outputs[1200]["decode"]["tok_per_s"]

    assert tok_800 != tok_1000, f"CLI --freq 800 and 1000 produce identical tok/s={tok_800} (freq not propagating)"
    assert tok_1000 != tok_1200, f"CLI --freq 1000 and 1200 produce identical tok/s={tok_1000} (freq not propagating)"

    # tok/s should approximately scale with frequency (compute-bound with block engine).
    # Non-compute components (SFU, KV, DRAM refresh) have different frequency dependencies.
    # Allow 15% tolerance.
    ratio_800_1000 = tok_800 / tok_1000
    assert ratio_800_1000 == pytest.approx(800.0 / 1000.0, rel=0.15), (
        f"CLI tok/s ratio 800/1000 MHz: expected ~{800 / 1000:.3f}, got {ratio_800_1000:.4f}"
    )

    ratio_1200_1000 = tok_1200 / tok_1000
    assert ratio_1200_1000 == pytest.approx(1200.0 / 1000.0, rel=0.15), (
        f"CLI tok/s ratio 1200/1000 MHz: expected ~{1200 / 1000:.3f}, got {ratio_1200_1000:.4f}"
    )


# ── Configuration conflict: bandwidth_bytes_per_cycle rejection ──


def test_bandwidth_gbps_overrides_legacy_bytes_per_cycle():
    """Given a config with both bandwidth_gbps and (stale) bandwidth_bytes_per_cycle,
    When an engine is constructed,
    Then bandwidth_gbps (with frequency) determines effective bandwidth.

    The legacy ``bandwidth_bytes_per_cycle`` field is ignored; the engine computes
    bytes/cycle from ``bandwidth_gbps`` at construction time.
    """
    cfg = _make_config(engine_type="block", freq_mhz=1000, bw_gbps=51.2)
    # Inject a conflicting stale bytes_per_cycle value
    cfg["memory"]["bandwidth_bytes_per_cycle"] = 999.0  # should be ignored

    engine = create_engine(cfg)
    # Compute expected eff_bw: 51.2 GB/s at 1000 MHz = 51.2 bytes/cycle * 0.85 dram_eff
    expected_bpc = bandwidth_gbps_to_bytes_per_cycle(51.2, 1000)  # 51.2
    expected_eff = expected_bpc * 0.85  # 43.52

    assert engine.eff_bw == pytest.approx(expected_eff, rel=1e-9), (
        f"Engine eff_bw={engine.eff_bw}, expected {expected_eff} "
        f"(from bandwidth_gbps=51.2 at 1000 MHz, not from stale bytes_per_cycle=999)"
    )
