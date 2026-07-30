"""Independent hand-audited oracle for scheduler transfer times.

Recomputes expected byte-transfer completion times from first principles
without importing scheduler code.  Used to verify ByteServer equal-share and
strict-priority behavior.
"""

from __future__ import annotations

import math


def bytes_per_us_to_bytes_per_ps(bytes_per_us: float) -> float:
    """Convert bytes/microsecond to bytes/picosecond."""
    return bytes_per_us / 1_000_000.0


def transfer_duration_ps(bytes_: int, bytes_per_us: float) -> int:
    """Duration for ``bytes_`` at ``bytes_per_us`` bandwidth."""
    if bytes_ < 0:
        raise ValueError(f"bytes must be non-negative, got {bytes_}")
    if bytes_per_us <= 0:
        raise ValueError(f"bandwidth must be positive, got {bytes_per_us}")
    if bytes_ == 0:
        return 0
    bw_ps = bytes_per_us_to_bytes_per_ps(bytes_per_us)
    return math.ceil(bytes_ / bw_ps)


def single_transfer_completion(
    bytes_: int,
    bytes_per_us: float,
    start_ps: int = 0,
) -> int:
    """Absolute completion time for a single transfer."""
    return start_ps + transfer_duration_ps(bytes_, bytes_per_us)


def equal_share_completion_times(
    requests: list[tuple[str, int]],
    total_bytes_per_us: float,
    start_ps: int = 0,
) -> dict[str, int]:
    """Oracle for equal-share bandwidth: each active request shares equally.

    All requests start together and each gets ``total / N`` bandwidth,
    so all complete at the same time.
    """
    if not requests:
        return {}
    n = len(requests)
    per_member_bw = total_bytes_per_us / n
    if per_member_bw <= 0:
        raise ValueError("per-member bandwidth must be positive")
    duration = max(transfer_duration_ps(bytes_, per_member_bw) for _, bytes_ in requests)
    return {rid: start_ps + duration for rid, _ in requests}


def strict_priority_completion_times(
    requests: list[tuple[str, int, int]],
    total_bytes_per_us: float,
    start_ps: int = 0,
) -> dict[str, int]:
    """Oracle for strict-priority bandwidth: highest priority runs first.

    ``requests`` is a list of ``(request_id, bytes, priority)`` where higher
    priority values run first.  Ties are broken by request_id for stability.
    """
    ordered = sorted(requests, key=lambda item: (-item[2], item[0]))
    now = start_ps
    result: dict[str, int] = {}
    for rid, bytes_, _ in ordered:
        duration = transfer_duration_ps(bytes_, total_bytes_per_us)
        now += duration
        result[rid] = now
    return result
