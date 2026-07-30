"""Unit-conversion tests for sim.contracts.units.

Covers the plan acceptance criteria:

* 800 / 1000 / 1200 MHz × 25.6–819.2 GB/s conversions.
* ``bytes_per_cycle = bandwidth_gbps * 1000 / frequency_mhz``.
* Wall-time round-trip error < 1e-12.
* Fixed 51.2 GB/s: 800 MHz → 64, 1000 MHz → 51.2, 1200 MHz → 42.666… bytes/cycle.
* Edge cases: zero / negative frequency, zero bandwidth.
"""

import pytest
from contracts.units import (
    bandwidth_gbps_to_bytes_per_cycle,
    bytes_per_cycle_to_bandwidth_gbps,
    bytes_to_gib,
    cycles_to_microseconds,
    cycles_to_seconds,
    gib_to_bytes,
    microseconds_to_cycles,
    seconds_to_cycles,
)

# ── bandwidth_gbps_to_bytes_per_cycle ──────────────────────────────────────


@pytest.mark.parametrize(
    "bw_gbps, freq_mhz, expected",
    [
        # Fixed 51.2 GB/s × three frequencies (plan acceptance criteria)
        (51.2, 800, 64.0),
        (51.2, 1000, 51.2),
        (51.2, 1200, 51.2 * 1000 / 1200),  # 42.666...
        # Sweep 25.6–819.2 GB/s at 1000 MHz
        (25.6, 1000, 25.6),
        (51.2, 1000, 51.2),
        (102.4, 1000, 102.4),
        (204.8, 1000, 204.8),
        (460.0, 1000, 460.0),
        (819.2, 1000, 819.2),
        # 800 MHz
        (25.6, 800, 32.0),
        (51.2, 800, 64.0),
        (102.4, 800, 128.0),
        (204.8, 800, 256.0),
        (819.2, 800, 1024.0),
        # 1200 MHz
        (25.6, 1200, 25.6 * 1000 / 1200),
        (51.2, 1200, 51.2 * 1000 / 1200),
        (102.4, 1200, 102.4 * 1000 / 1200),
    ],
)
def test_bandwidth_gbps_to_bytes_per_cycle(bw_gbps: float, freq_mhz: float, expected: float):
    """Given bandwidth in GB/s and frequency in MHz,
    When converting to bytes/cycle,
    Then result matches formula, with float round-trip error < 1e-12.
    """
    result = bandwidth_gbps_to_bytes_per_cycle(bw_gbps, freq_mhz)
    assert result == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_bandwidth_gbps_to_bytes_per_cycle_round_trip():
    """Given a byte-per-cycle value at a given frequency,
    When converting to GB/s and back to bytes/cycle,
    Then the result is recovered with error < 1e-12.
    """
    freqs = [800.0, 1000.0, 1200.0]
    bandwidts_gbps = [25.6, 51.2, 102.4, 204.8, 460.0, 819.2]

    for freq in freqs:
        for bw in bandwidts_gbps:
            bpc = bandwidth_gbps_to_bytes_per_cycle(bw, freq)
            recovered = bytes_per_cycle_to_bandwidth_gbps(bpc, freq)
            assert recovered == pytest.approx(bw, rel=1e-12, abs=1e-12)


def test_bandwidth_gbps_to_bytes_per_cycle_rejects_non_positive_frequency():
    """Given a non-positive frequency,
    When calling the conversion,
    Then ValueError is raised.
    """
    with pytest.raises(ValueError):
        bandwidth_gbps_to_bytes_per_cycle(51.2, 0.0)
    with pytest.raises(ValueError):
        bandwidth_gbps_to_bytes_per_cycle(51.2, -100.0)


# ── bytes_per_cycle_to_bandwidth_gbps ───────────────────────────────────────


def test_bytes_per_cycle_to_bandwidth_gbps_round_trip():
    """Given a bytes/cycle value at a frequency,
    When converting round-trip (BW → bpc → BW),
    Then the original bandwidth is recovered.
    """
    freqs = [800.0, 1000.0, 1200.0]
    bandwidts_gbps = [25.6, 51.2, 102.4, 204.8, 819.2]

    for freq in freqs:
        for bw in bandwidts_gbps:
            bpc = bandwidth_gbps_to_bytes_per_cycle(bw, freq)
            recovered = bytes_per_cycle_to_bandwidth_gbps(bpc, freq)
            assert recovered == pytest.approx(bw, rel=1e-12, abs=1e-12)


def test_bytes_per_cycle_to_bandwidth_gbps_rejects_non_positive_frequency():
    """Given a non-positive frequency,
    When calling the reverse conversion,
    Then ValueError is raised.
    """
    with pytest.raises(ValueError):
        bytes_per_cycle_to_bandwidth_gbps(64.0, 0.0)
    with pytest.raises(ValueError):
        bytes_per_cycle_to_bandwidth_gbps(64.0, -100.0)


# ── cycles_to_seconds / cycles_to_microseconds ──────────────────────────────


@pytest.mark.parametrize(
    "cycles, freq_mhz, expected_seconds, expected_us",
    [
        (1_000_000, 1000, 0.001, 1_000.0),
        (1_000, 1000, 1e-6, 1.0),
        (1, 1000, 1e-9, 0.001),
        (1_200_000, 800, 0.0015, 1_500.0),
        (800_000, 1200, 800_000.0 / (1200.0 * 1e6), 800_000.0 / 1200.0),
    ],
)
def test_cycles_to_seconds_and_us(
    cycles: float,
    freq_mhz: float,
    expected_seconds: float,
    expected_us: float,
):
    """Given cycles and frequency,
    When converting to wall-clock seconds and microseconds,
    Then results match closed-form calculations.
    """
    assert cycles_to_seconds(cycles, freq_mhz) == pytest.approx(expected_seconds, rel=1e-12)
    assert cycles_to_microseconds(cycles, freq_mhz) == pytest.approx(expected_us, rel=1e-12)


def test_cycles_to_seconds_round_trip():
    """Given a wall-clock time in seconds,
    When converting to cycles and back,
    Then the original time is recovered with error < 1e-12.
    """
    freqs = [800.0, 1000.0, 1200.0]
    seconds_values = [0.0, 0.001, 1.0, 10.0, 100.0]

    for freq in freqs:
        for s in seconds_values:
            cycles = seconds_to_cycles(s, freq)
            recovered = cycles_to_seconds(cycles, freq)
            assert recovered == pytest.approx(s, rel=1e-12, abs=1e-12)


def test_cycles_to_seconds_rejects_non_positive_frequency():
    """Given a non-positive frequency,
    When converting cycles to seconds,
    Then ValueError is raised.
    """
    with pytest.raises(ValueError):
        cycles_to_seconds(1000, 0.0)
    with pytest.raises(ValueError):
        cycles_to_seconds(1000, -100.0)


# ── microseconds round-trip ────────────────────────────────────────────────


def test_microseconds_round_trip():
    """Given a wall-clock time in microseconds,
    When converting to cycles and back,
    Then the original time is recovered with error < 1e-12.
    """
    freqs = [800.0, 1000.0, 1200.0]
    us_values = [0.0, 0.1, 1.0, 100.0, 1000.0]

    for freq in freqs:
        for us in us_values:
            cycles = microseconds_to_cycles(us, freq)
            recovered = cycles_to_microseconds(cycles, freq)
            assert recovered == pytest.approx(us, rel=1e-12, abs=1e-12)


# ── bytes_to_gib / gib_to_bytes ─────────────────────────────────────────────


def test_bytes_to_gib():
    """Given a byte count,
    When converting to GiB,
    Then result is bytes / 2^30."""
    assert bytes_to_gib(0) == 0.0
    assert bytes_to_gib(1024**3) == 1.0
    assert bytes_to_gib(2 * 1024**3) == 2.0
    assert bytes_to_gib(1024**2) == pytest.approx(1.0 / 1024, rel=1e-12)


def test_gib_to_bytes():
    """Given a GiB count,
    When converting to bytes,
    Then result is GiB * 2^30."""
    assert gib_to_bytes(0) == 0.0
    assert gib_to_bytes(1.0) == 1024.0**3
    assert gib_to_bytes(2.0) == 2 * 1024.0**3


def test_bytes_gib_round_trip():
    """Given a GiB value,
    When converting to bytes and back,
    Then the original value is recovered."""
    for g in [0.0, 0.5, 1.0, 4.0, 16.0, 100.0]:
        assert bytes_to_gib(gib_to_bytes(g)) == pytest.approx(g, rel=1e-12)


# ── wall-time round-trip (plan acceptance) ──────────────────────────────────


def test_wall_time_round_trip_within_1e12():
    """Given bandwidth, frequency, and a byte transfer size,
    When computing wall time through bytes/cycle → cycles → seconds,
    Then the computed wall time is closed-form consistent with error < 1e-12.

    This is the plan acceptance criterion: for 51.2 GB/s at 800/1000/1200 MHz,
    the computed values must round-trip correctly.
    """
    bw_gbps = 51.2
    transfer_bytes = 1_000_000  # arbitrary transfer
    freqs = [800.0, 1000.0, 1200.0]

    for freq in freqs:
        # Direct wall-time: time = bytes / (BW_gbps * 1e9 / 1e9) = bytes / (BW * 1e9) * freq
        # Not simplified — compute through our unit pipeline
        bpc = bandwidth_gbps_to_bytes_per_cycle(bw_gbps, freq)
        cycles = transfer_bytes / bpc
        wall_seconds = cycles_to_seconds(cycles, freq)
        wall_us = cycles_to_microseconds(cycles, freq)

        # Closed-form: wall_seconds = transfer_bytes / (bw_gbps * 1e9)
        expected_seconds = transfer_bytes / (bw_gbps * 1e9)
        expected_us = expected_seconds * 1e6

        assert wall_seconds == pytest.approx(expected_seconds, rel=1e-12, abs=1e-12)
        assert wall_us == pytest.approx(expected_us, rel=1e-12, abs=1e-12)
