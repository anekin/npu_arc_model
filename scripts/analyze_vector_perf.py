#!/usr/bin/env python3
"""Vector Module-Level Performance Analyzer.

Cycle formula calculator + VCS PERF log parser for Vector operations.
Parses standardized PERF|case=X|op=...|event=E|cycles=N lines and compares
measured cycles against expected formulas derived from Vector RTL FSM.

Expected cycle formulas (from rtl/testcase-list-sfu-vector-perf.md §4):
    ADD/MUL/MAX/RESID: ceil(N/128) * 4 + 2
    SUM:               ceil(N/128) * 10 + 2
    CONV:              ceil(N/128) * 132 + 2

Tolerances:
    All ops: |delta| <= 1

Usage:
    python3 scripts/analyze_vector_perf.py --case SFV-P08 --log sim.log
    python3 scripts/analyze_vector_perf.py --case SFV-P08 --op add --dim 128 --dry-run
    python3 scripts/analyze_vector_perf.py --log sim.log  (parse from PERF lines)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Cycle formulas
# ══════════════════════════════════════════════════════════════════════

VECTOR_FORMULAS: dict[str, str] = {
    "add":   "ceil(N/128) * 4 + 2",
    "mul":   "ceil(N/128) * 4 + 2",
    "max":   "ceil(N/128) * 10 + 2",  # MAX routes through reduce_tree, not ALU
    "resid": "ceil(N/128) * 4 + 2",
    "sum":   "ceil(N/128) * 10 + 2",
    "conv":  "ceil(N/128) * 259 + 2", # CONV_FEED(N) + CONV_CAPTURE(N) = 2N, not N
}


def expected_cycles(op: str, dim: int) -> int:
    """Return the expected cycle count for a Vector operation and dimension."""
    chunks = math.ceil(dim / 128)

    formulas: dict[str, int] = {
        "add":   chunks * 4 + 2,
        "mul":   chunks * 4 + 2,
        "max":   chunks * 10 + 2,   # MAX routes through reduce_tree, not ALU
        "resid": chunks * 4 + 2,
        "sum":   chunks * 10 + 2,
        "conv":  chunks * 259 + 2,  # CONV_FEED(N) + CONV_CAPTURE(N) sequential
    }
    result = formulas.get(op.lower())
    if result is None:
        raise ValueError(f"Unknown Vector op: {op} (valid: {', '.join(formulas.keys())})")
    return result


# ══════════════════════════════════════════════════════════════════════
# PERF log parsing
# ══════════════════════════════════════════════════════════════════════

PERF_LINE_RE = re.compile(
    r"PERF\|case=([^|]+)\|op=([^|]+)\|event=([^|]+)\|cycles=(\d+)"
)


def parse_perf_log(log_path: str) -> list[dict]:
    """Parse PERF| lines from a VCS simulation log."""
    entries: list[dict] = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = PERF_LINE_RE.search(line)
            if m:
                entries.append({
                    "case": m.group(1),
                    "op": m.group(2),
                    "event": m.group(3),
                    "cycles": int(m.group(4)),
                })
    return entries


def analyze_entries(entries: list[dict], case_id: Optional[str] = None) -> dict:
    """Analyze parsed PERF entries and return a verdict dict."""
    results: dict = {
        "total_cycles": None,
        "expected": None,
        "delta": None,
        "pass": False,
        "per_state": {},
        "chunks": None,
        "verdict": "FAIL (no PERF data)",
    }

    # Filter by case_id if provided
    filtered = entries
    if case_id:
        filtered = [e for e in entries if e["case"] == case_id]

    if not filtered:
        results["verdict"] = "FAIL (no PERF entries found)"
        return results

    # Extract TOTAL cycles
    total_entry = next((e for e in filtered if e["event"] == "TOTAL"), None)
    if total_entry is None:
        results["verdict"] = "FAIL (no TOTAL event found)"
        return results

    measured = total_entry["cycles"]

    # Collect per-state breakdown
    for e in filtered:
        if e["event"] not in ("TOTAL", "GAP"):
            results["per_state"][e["event"]] = e["cycles"]

    # Extract chunk count
    chunk_entry = next((e for e in filtered if e["event"] == "CHUNKS"), None)
    if chunk_entry:
        results["chunks"] = chunk_entry["cycles"]

    # Extract op and dim from the op field
    op_field = filtered[0]["op"]
    op_match = re.match(r"op=(\w+),dim=(\d+)", op_field)
    if not op_match:
        results["total_cycles"] = measured
        results["verdict"] = f"PASS (measured={measured}, no formula available)"
        return results

    op_name = op_match.group(1).lower()
    dim = int(op_match.group(2))
    expected = expected_cycles(op_name, dim)
    tol = 1  # all Vector ops: |delta| <= 1
    delta = measured - expected

    passed = abs(delta) <= tol
    results["total_cycles"] = measured
    results["expected"] = expected
    results["delta"] = delta
    results["tolerance"] = tol
    results["op"] = op_name
    results["dim"] = dim
    results["pass"] = passed
    results["verdict"] = (
        f"{'PASS' if passed else 'FAIL'}"
        f" (measured={measured}, expected={expected}, delta={delta}, tol={tol})"
    )
    return results


# ══════════════════════════════════════════════════════════════════════
# Output formatting
# ══════════════════════════════════════════════════════════════════════


def print_dry_run(op: str, dim: int, case_id: str = "") -> None:
    """Print a dry-run formula check."""
    expected = expected_cycles(op, dim)
    chunks = math.ceil(dim / 128)
    formula_str = VECTOR_FORMULAS.get(op.lower(), "unknown")

    print(f"Case: {case_id or 'N/A'}")
    print(f"Op: {op}, Dim: {dim}, Chunks: ceil({dim}/128) = {chunks}")
    print(f"Expected cycle formula: {formula_str}")
    print(f"  = ceil({dim}/128) * per_chunk_cycles + 2")
    print(f"  = {chunks} * {expected - 2} + 2")
    print(f"  = {expected}")
    print(f"Tolerance: |delta| <= 1")
    print(f"[{case_id or op}] op={op},dim={dim} expected={expected} — PASS (dry-run, formula check only)")


def print_report(results: dict, case_id: str = "") -> None:
    """Print a formatted analysis report."""
    if results["total_cycles"] is None:
        print(f"[{case_id}] No valid PERF data found.")
        return

    measured = results["total_cycles"]
    expected = results.get("expected")
    delta = results.get("delta")
    tol = results.get("tolerance", 0)
    passed = results.get("pass", False)
    op_name = results.get("op", "?")
    dim = results.get("dim", 0)
    chunks = results.get("chunks")

    print(f"[{case_id}] op={op_name},dim={dim} expected={expected} measured={measured} delta={delta} {'PASS' if passed else 'FAIL'}")
    if chunks is not None:
        print(f"  Chunk count: {chunks}")

    if results.get("per_state"):
        print("  Per-state breakdown:")
        for state, cycles in sorted(results["per_state"].items()):
            print(f"    {state}: {cycles}")

    if not passed and expected is not None:
        excess = abs(delta) - tol
        print(f"  FAIL: measured={measured} exceeds expected={expected} by {delta} (tolerance {tol})")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vector module-level performance analyzer (cycle formula + PERF log parser)",
    )
    parser.add_argument("--case", default="", help="Case ID for filtering PERF lines")
    parser.add_argument("--log", help="Path to VCS simulation log with PERF| lines")
    parser.add_argument("--op", help="Vector operation name (add, mul, max, sum, conv, resid)")
    parser.add_argument("--dim", type=int, help="Element dimension")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Formula check only (no log parsing required)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only PASS/FAIL verdict line",
    )
    args = parser.parse_args()

    if args.dry_run:
        if not args.op or args.dim is None:
            print("ERROR: --op and --dim required for dry-run", file=sys.stderr)
            return 1
        print_dry_run(args.op, args.dim, args.case)
        return 0

    if not args.log:
        print("ERROR: --log required (or use --dry-run)", file=sys.stderr)
        return 1

    entries = parse_perf_log(args.log)

    if args.op and args.dim is not None:
        op_field = f"op={args.op},dim={args.dim}"
        filtered = [e for e in entries if e["op"] == op_field]

        if not filtered:
            print(f"[{args.case}] FAIL: no PERF entries for op={op_field}")
            return 1

        if args.case:
            filtered = [e for e in filtered if e["case"] == args.case]

        results = analyze_entries(filtered, args.case)
    else:
        # Full log mode: group by case
        cases = sorted(set(e["case"] for e in entries))
        all_pass = True
        for case in cases:
            results = analyze_entries(entries, case)
            if not args.summary_only:
                print_report(results, case)
            elif results.get("pass", False):
                print(f"[{case}] PASS")
            else:
                print(f"[{case}] FAIL")
                all_pass = False
        return 0 if all_pass else 1

    if not args.summary_only:
        print_report(results, args.case)

    passed = results.get("pass", False)
    print(f"PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
