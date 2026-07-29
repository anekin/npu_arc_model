"""Canonical unit conversions for the Arc Model.

All conversions are explicit, using decimal GB (powers of 1000) for bandwidth
and binary GiB (powers of 1024) for byte quantities. The module is the single
source of truth; production code must never inline conversions.

Key relationships (from plan):
  bytes_per_cycle = bandwidth_gbps * 1000.0 / frequency_mhz
  seconds = cycles / (frequency_mhz * 1e6)
  us = cycles / frequency_mhz
"""

from __future__ import annotations

__all__ = [
    "bandwidth_gbps_to_bytes_per_cycle",
    "bytes_per_cycle_to_bandwidth_gbps",
    "cycles_to_seconds",
    "cycles_to_microseconds",
    "seconds_to_cycles",
    "microseconds_to_cycles",
    "bytes_to_gib",
    "gib_to_bytes",
]


def bandwidth_gbps_to_bytes_per_cycle(
    bandwidth_gbps: float,
    frequency_mhz: float,
) -> float:
    """Convert GB/s bandwidth to bytes per core clock cycle.

    Formula: bytes_per_cycle = bandwidth_gbps * 1000 / frequency_mhz

    This uses decimal GB (GB/s = 10^9 bytes/s) and MHz (10^6 cycles/s):
      (BW * 10^9 bytes/s) / (freq * 10^6 cycles/s) = BW * 1000 / freq bytes/cycle
    """
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be > 0, got {frequency_mhz}")
    return bandwidth_gbps * 1000.0 / frequency_mhz


def bytes_per_cycle_to_bandwidth_gbps(
    bytes_per_cycle: float,
    frequency_mhz: float,
) -> float:
    """Reverse conversion: bytes/cycle → GB/s bandwidth."""
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be > 0, got {frequency_mhz}")
    return bytes_per_cycle * frequency_mhz / 1000.0


def cycles_to_seconds(cycles: float, frequency_mhz: float) -> float:
    """Convert core cycles to wall-clock seconds."""
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be > 0, got {frequency_mhz}")
    return cycles / (frequency_mhz * 1e6)


def cycles_to_microseconds(cycles: float, frequency_mhz: float) -> float:
    """Convert core cycles to wall-clock microseconds."""
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be > 0, got {frequency_mhz}")
    return cycles / frequency_mhz


def seconds_to_cycles(seconds: float, frequency_mhz: float) -> float:
    """Convert wall-clock seconds to core cycles."""
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be > 0, got {frequency_mhz}")
    return seconds * frequency_mhz * 1e6


def microseconds_to_cycles(us: float, frequency_mhz: float) -> float:
    """Convert wall-clock microseconds to core cycles."""
    if frequency_mhz <= 0:
        raise ValueError(f"frequency_mhz must be > 0, got {frequency_mhz}")
    return us * frequency_mhz


def bytes_to_gib(size_bytes: float) -> float:
    """Convert bytes to gibibytes (GiB = 2^30 bytes)."""
    return size_bytes / (1024**3)


def gib_to_bytes(size_gib: float) -> float:
    """Convert gibibytes to bytes (GiB = 2^30 bytes)."""
    return size_gib * (1024**3)
