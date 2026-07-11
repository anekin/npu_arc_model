#!/usr/bin/env python3
"""
SFU result comparator used by tb_sfu.v.

Reads the scenario manifest, loads the golden reference and the RTL result,
and compares float16 values with a tolerance matched to the RTL's FP16
approximations (including RoPE fixed-point trig).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def read_hex_float16(path: Path) -> np.ndarray:
    """Read a file of 4-hex-digit FP16 values into a numpy float16 array."""
    vals = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(int(line, 16))
    return np.array(vals, dtype=np.uint16).view(np.float16)


@dataclass
class CompareResult:
    passed: bool
    max_abs_diff: float = 0.0
    max_rel_diff: float = 0.0


def compare_float16(
    golden: np.ndarray,
    result: np.ndarray,
    abs_tol: float = 1e-3,
    rel_tol: float = 1e-2,
) -> CompareResult:
    """Compare two float16 arrays with abs/rel tolerance."""
    if golden.shape != result.shape:
        return CompareResult(passed=False)
    g = golden.astype(np.float64)
    r = result.astype(np.float64)
    abs_diff = np.abs(g - r)
    rel_diff = np.zeros_like(abs_diff)
    nonzero = np.abs(g) > 0
    rel_diff[nonzero] = abs_diff[nonzero] / np.abs(g[nonzero])
    ok = (abs_diff <= abs_tol) | (rel_diff <= rel_tol)
    if ok.all():
        return CompareResult(passed=True, max_abs_diff=float(np.max(abs_diff)), max_rel_diff=float(np.max(rel_diff)))
    return CompareResult(passed=False, max_abs_diff=float(np.max(abs_diff)), max_rel_diff=float(np.max(rel_diff)))


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: compare_sfu.py <test_dir> <result.hex>", file=sys.stderr)
        return 2

    test_dir = Path(sys.argv[1])
    result_path = Path(sys.argv[2])

    manifest = json.loads((test_dir / "manifest.json").read_text())
    golden_path = test_dir / manifest["files"]["golden"]

    golden = read_hex_float16(golden_path)
    result = read_hex_float16(result_path)

    # Looser absolute tolerance: RTL FP16 trig can differ by ~1 ULP (~1e-3)
    # from the float64 golden, while the relative tolerance stays tight.
    r = compare_float16(golden, result, abs_tol=2e-3, rel_tol=1e-2)

    print("INLINE_COMPARE:", "PASS" if r.passed else "FAIL")
    if not r.passed:
        print(f"max_abs_diff={r.max_abs_diff:.6e} max_rel_diff={r.max_rel_diff:.6e}")
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main())
