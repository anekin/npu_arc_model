#!/usr/bin/env python3
"""
MXU Performance Analyzer — per-tile cycle formula + VCS log parser.

Implements the module-level MXU per-tile cycle formula and parses
``PERF|...`` lines from ``tb_mxu_perf.v`` output, reporting per-state
and per-tile cycle counts with a PASS/FAIL verdict against expected cycles.

Usage:
    # Dry-run mode (no VCS required)
    python3 CaduceusCore/scripts/analyze_perf.py --dry-run --shape 64,64,64

    # Normal mode — parse a VCS log
    python3 CaduceusCore/scripts/analyze_perf.py --case MX-P01 --log <vcs_log>
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO


# ══════════════════════════════════════════════════════════════════════
# Cycle formula — golden reference
# ══════════════════════════════════════════════════════════════════════


def expected_cycles(M: int, K: int, N: int) -> int:
    """Compute expected total MXU cycles using the per-tile summation formula.

    Formula (golden reference):
        Total = 1                               # READ_DIMS
               + Σ(M_tiles,N_tiles,K_tiles) [
                     LOAD_W   = 1
                 +   LOAD_A   = 1
                 +   COMPUTE  = k_cur + 2        # k_cur = min(64, remaining K)
                 +   STORE_OUT = (is_last_ktile ? m_cur + 1 : 0)
               ]

    Args:
        M: Number of output rows (M dimension).
        K: Shared (reduction) dimension.
        N: Number of output columns (N dimension).

    Returns:
        Expected total cycle count.
    """
    M_tiles = (M + 63) // 64
    N_tiles = (N + 63) // 64
    K_tiles = (K + 63) // 64

    total = 1  # READ_DIMS
    for mt in range(M_tiles):
        m_cur = min(64, M - mt * 64)
        for nt in range(N_tiles):
            for kt in range(K_tiles):
                k_cur = min(64, K - kt * 64)
                total += 1 + 1 + (k_cur + 2)  # LOAD_W + LOAD_A + COMPUTE
                if kt == K_tiles - 1:
                    total += m_cur + 1  # STORE_OUT
    return total


# ══════════════════════════════════════════════════════════════════════
# VCS log parsing
# ══════════════════════════════════════════════════════════════════════

# PERF|case=MX-P01|shape=64,64,64|event=READ_DIMS|cycles=1
_RE_PERF = re.compile(
    r"^PERF\|case=(?P<case>[^|]+)"
    r"\|shape=(?P<M>\d+),(?P<N>\d+),(?P<K>\d+)"
    r"\|event=(?P<event>[^|]+)"
    r"\|cycles=(?P<cycles>\d+)"
)

# TILE|tile=0|cycles=12
_RE_TILE = re.compile(r"^TILE\|tile=(?P<tile>\d+)\|cycles=(?P<cycles>\d+)")

# GAP|op=1|cycles=0
_RE_GAP = re.compile(r"^GAP\|op=(?P<op>\d+)\|cycles=(?P<cycles>\d+)")


@dataclass(frozen=True)
class PerfEvent:
    """A single parsed PERF event line."""

    case_id: str
    M: int
    N: int
    K: int
    event: str
    cycles: int


@dataclass(frozen=True)
class CaseResult:
    """Aggregated analysis result for one test case."""

    case_id: str
    M: int
    N: int
    K: int
    state_cycles: dict[str, int]
    total_measured: int | None
    tile_cycles: dict[int, int]
    gap_cycles: dict[int, int]


def parse_perf_log(log_io: IO[str]) -> list[CaseResult]:
    """Parse a VCS log file handle and return a list of case results.

    Handles multiple test cases in a single log. PERF lines define the
    case boundary; TILE and GAP lines belong to the most recent case seen.

    Args:
        log_io: Open text stream (file or StringIO).

    Returns:
        List of CaseResult in order of appearance.
    """
    events: dict[str, list[PerfEvent]] = {}
    tiles: dict[str, dict[int, int]] = {}
    gaps: dict[str, dict[int, int]] = {}
    case_order: list[str] = []
    current_case: str | None = None

    for raw_line in log_io:
        line = raw_line.rstrip("\n")

        # PERF line
        m = _RE_PERF.match(line)
        if m:
            case_id = m.group("case")
            event = PerfEvent(
                case_id=case_id,
                M=int(m.group("M")),
                N=int(m.group("N")),
                K=int(m.group("K")),
                event=m.group("event"),
                cycles=int(m.group("cycles")),
            )
            if case_id not in events:
                events[case_id] = []
                tiles[case_id] = {}
                gaps[case_id] = {}
                case_order.append(case_id)
            events[case_id].append(event)
            current_case = case_id
            continue

        # TILE line
        m = _RE_TILE.match(line)
        if m and current_case is not None:
            tile_idx = int(m.group("tile"))
            cycles = int(m.group("cycles"))
            tiles[current_case][tile_idx] = cycles
            continue

        # GAP line
        m = _RE_GAP.match(line)
        if m and current_case is not None:
            op_idx = int(m.group("op"))
            cycles = int(m.group("cycles"))
            gaps[current_case][op_idx] = cycles
            continue

    results: list[CaseResult] = []
    for case_id in case_order:
        case_events = events[case_id]
        if not case_events:
            continue
        # dimensions from the first event
        M = case_events[0].M
        N = case_events[0].N
        K = case_events[0].K

        state_cycles: dict[str, int] = {}
        total_measured: int | None = None
        for ev in case_events:
            if ev.event == "TOTAL":
                total_measured = ev.cycles
            state_cycles[ev.event] = ev.cycles

        results.append(
            CaseResult(
                case_id=case_id,
                M=M,
                N=N,
                K=K,
                state_cycles=state_cycles,
                total_measured=total_measured,
                tile_cycles=tiles.get(case_id, {}),
                gap_cycles=gaps.get(case_id, {}),
            )
        )

    return results


# ══════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════


def _state_summary(state_cycles: dict[str, int]) -> str:
    """Build a compact per-state summary string."""
    parts: list[str] = []
    for name in ("READ_DIMS", "LOAD_W", "LOAD_A", "COMPUTE", "STORE_OUT", "TOTAL"):
        val = state_cycles.get(name)
        if val is not None:
            parts.append(f"{name}={val}")
    return " ".join(parts)


def report_case(result: CaseResult) -> str:
    """Format a single case result as a human-readable string.

    Includes the expected cycles, delta, and PASS/FAIL verdict.
    """
    expected = expected_cycles(result.M, result.K, result.N)
    measured = result.total_measured
    state_line = f"  [{result.case_id}] per-state: {_state_summary(result.state_cycles)}"

    if measured is not None:
        delta = measured - expected
        verdict = "PASS" if abs(delta) <= 1 else "FAIL"
        header = (
            f"[{result.case_id}] shape={result.M},{result.N},{result.K} "
            f"expected={expected} measured={measured} delta={delta} {verdict}"
        )
    else:
        header = (
            f"[{result.case_id}] shape={result.M},{result.N},{result.K} "
            f"expected={expected} (TOTAL line not found in log)"
        )

    lines = [header, state_line]

    if result.tile_cycles:
        tile_str = " ".join(
            f"tile[{i}]={c}" for i, c in sorted(result.tile_cycles.items())
        )
        lines.append(f"  [{result.case_id}] tiles: {tile_str}")

    if result.gap_cycles:
        gap_str = " ".join(
            f"gap[{i}]={c}" for i, c in sorted(result.gap_cycles.items())
        )
        lines.append(f"  [{result.case_id}] gaps: {gap_str}")

    return "\n".join(lines)


def _dry_run_report(case_id: str, M: int, N: int, K: int) -> str:
    """Format the dry-run output (computes expected cycles only)."""
    total = expected_cycles(M, K, N)
    return f"[{case_id}] dry-run shape={M},{N},{K} expected_cycles={total}"


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    """Parse a comma-separated shape string ``M,N,K`` into integers."""
    parts = shape_str.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"shape must be M,N,K (3 integers, got {len(parts)}: {shape_str!r})"
        )
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="MXU Performance Analyzer — per-tile cycle formula + log parser",
    )
    parser.add_argument(
        "--case",
        default="MX-P01",
        help="Test case identifier (default: MX-P01)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Path to VCS simulation log (PERF|... lines expected)",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        help="Shape as M,N,K (e.g. 64,64,64). Required for dry-run; accepted in normal mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute expected cycles only; no log parsing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        if args.shape is None:
            print("error: --shape M,N,K is required with --dry-run", file=sys.stderr)
            return 1
        M, N, K = args.shape
        print(_dry_run_report(args.case, M, N, K))
        return 0

    # Normal mode: parse log
    if args.log is None:
        print(
            "error: --log <path> is required in normal mode (or use --dry-run)",
            file=sys.stderr,
        )
        return 1

    if not args.log.exists():
        print(f"error: log file not found: {args.log}", file=sys.stderr)
        return 1

    # If --shape is provided, we can compute expected even if TOTAL is missing
    M, N, K = args.shape if args.shape else (0, 0, 0)

    with args.log.open("r", encoding="utf-8", errors="replace") as f:
        results = parse_perf_log(f)

    if not results:
        print(f"No PERF|... lines found in {args.log}", file=sys.stderr)
        return 1

    for result in results:
        print(report_case(result))

    return 0


if __name__ == "__main__":
    sys.exit(main())
