#!/usr/bin/env python3
"""SFU Module-Level Performance Analyzer.

Cycle formula calculator + VCS PERF log parser for SFU operations.
Parses standardized PERF|case=X|op=...|event=E|cycles=N lines and compares
measured cycles against expected formulas derived from SFU RTL FSM/pipeline.

Expected cycle formulas (from rtl/testcase-list-sfu-vector-perf.md §4):
    gelu:    N + 7
    silu:    N + 7
    rope:    N + 19
    softmax: 3N + 33
    layernorm: 3N + 17
    rmsnorm: 2N + 21

Tolerances:
    Streaming ops (gelu, silu, rope): |delta| <= 1
    Reduction ops (softmax, layernorm, rmsnorm): |delta| <= 5

Usage:
    python3 scripts/analyze_sfu_perf.py --case SFV-P01 --log sim.log
    python3 scripts/analyze_sfu_perf.py --case SFV-P01 --op softmax --dim 64 --dry-run
    python3 scripts/analyze_sfu_perf.py --log sim.log  (parse from PERF lines)
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# Cycle formulas
# ══════════════════════════════════════════════════════════════════════

SFU_FORMULAS: dict[str, str] = {
    "gelu":      "N + 7",
    "silu":      "N + 7",
    "rope":      "N + 19",
    "softmax":   "3*N + 33",
    "layernorm": "3*N + 17",
    "rmsnorm":   "2*N + 21",
}


def expected_cycles(op: str, dim: int) -> int:
    """Return the expected cycle count for an SFU operation and dimension."""
    formulas = {
        "gelu":      lambda n: n + 7,
        "silu":      lambda n: n + 7,
        "rope":      lambda n: n + 19,
        "softmax":   lambda n: 3 * n + 33,
        "layernorm": lambda n: 3 * n + 17,
        "rmsnorm":   lambda n: 2 * n + 21,
    }
    fn = formulas.get(op.lower())
    if fn is None:
        raise ValueError(f"Unknown SFU op: {op} (valid: {', '.join(formulas.keys())})")
    return fn(dim)


def tolerance_for(op: str) -> int:
    """Return the max allowed |delta| for a given SFU operation type."""
    streaming = {"gelu", "silu", "rope"}
    if op.lower() in streaming:
        return 1
    return 5  # reduction ops


# ══════════════════════════════════════════════════════════════════════
# PERF log parsing
# ══════════════════════════════════════════════════════════════════════

PERF_LINE_RE = re.compile(
    r"PERF\|case=([^|]+)\|op=([^|]+)\|event=([^|]+)\|cycles=(\d+)"
)


def parse_perf_log(log_path: str) -> list[dict]:
    """Parse PERF| lines from a VCS simulation log.

    Returns list of dicts with keys: case, op, event, cycles.
    """
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
        if e["event"] not in ("TOTAL", "GAP", "CHUNKS"):
            results["per_state"][e["event"]] = e["cycles"]

    # Extract op and dim from the op field (format: "op=softmax,dim=64")
    op_field = filtered[0]["op"]
    op_match = re.match(r"op=(\w+),dim=(\d+)(?:,pos=(\d+))?", op_field)
    if not op_match:
        results["total_cycles"] = measured
        results["verdict"] = f"PASS (measured={measured}, no formula available)"
        return results

    op_name = op_match.group(1).lower()
    dim = int(op_match.group(2))
    expected = expected_cycles(op_name, dim)
    tol = tolerance_for(op_name)
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
    tol = tolerance_for(op)
    formula_str = SFU_FORMULAS.get(op.lower(), "unknown")

    print(f"Case: {case_id or 'N/A'}")
    print(f"Op: {op}, Dim: {dim}")
    print(f"Expected cycle formula: {formula_str}")
    print(f"  = {formula_str.replace('N', str(dim))}")
    print(f"  = {expected}")
    print(f"Tolerance: |delta| <= {tol}")
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

    print(f"[{case_id}] op={op_name},dim={dim} expected={expected} measured={measured} delta={delta} {'PASS' if passed else 'FAIL'}")

    if results.get("per_state"):
        print("  Per-state breakdown:")
        for state, cycles in sorted(results["per_state"].items()):
            print(f"    {state}: {cycles}")

    if not passed and expected is not None:
        excess = abs(delta) - tol
        print(f"  FAIL: measured={measured} exceeds expected={expected} by {delta} (tolerance {tol}, excess {excess})")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SFU module-level performance analyzer (cycle formula + PERF log parser)",
    )
    parser.add_argument("--case", default="", help="Case ID for filtering PERF lines")
    parser.add_argument("--log", help="Path to VCS simulation log with PERF| lines")
    parser.add_argument("--op", help="SFU operation name (softmax, layernorm, etc.)")
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
        # Specific op/dim mode: filter entries and compare
        # Use startswith to handle RoPE with pos suffix (e.g., "op=rope,dim=64,pos=0")
        op_prefix = f"op={args.op},dim={args.dim}"
        filtered = [e for e in entries if e["op"].startswith(op_prefix)]

        if not filtered:
            print(f"[{args.case}] FAIL: no PERF entries for op={op_field}")
            return 1

        # If we have all entries with a case filter, use that
        if args.case:
            filtered = [e for e in filtered if e["case"] == args.case]

        results = analyze_entries(filtered, args.case)
    else:
        # Full log mode: group by case and analyze each
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
