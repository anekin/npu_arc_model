#!/usr/bin/env python3
"""Extract Gate 1b TTFT targets from a DSE v2 result JSON.

Usage:
    uv run python sim/design_space_explorer.py --quick --batch-m 128 \
        --result-schema v2 --output /tmp/m128.json
    uv run python scripts/extract_gate1b_targets.py \
        --input /tmp/m128.json --func-model-ms 3911.05 \
        --output .omo/evidence/task-7-arc-prefill-ttft-dse-m128-extract.txt

Extracts Block 64x64 @ 1GHz LPDDR5-64b INT4 rows (wc=True primary,
wc=False reference), verifies the ratio Func_Model / Arc_TTFT falls in
[0.5, 2.0], and writes a report with the input sha256.

Exit codes: 0 = PASS, 1 = missing row or ratio out of window.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

FUNC_RATIO_LOW = 0.5
FUNC_RATIO_HIGH = 2.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_target_row(label: str) -> bool:
    if not label.startswith("bloc"):
        return False
    parts = label.split()
    if len(parts) < 4:
        return False
    dims = parts[1].split("×")
    if len(dims) != 2 or dims[0] != "64" or dims[1] != "64":
        return False
    if parts[3] != "1000MHz":
        return False
    return parts[-1] == "LPDDR5-64b"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="DSE v2 result JSON (from --result-schema v2)")
    parser.add_argument("--func-model-ms", required=True, type=float, help="Func Model reference TTFT in ms")
    parser.add_argument("--output", required=True, help="Path for the extract report")
    parser.add_argument(
        "--low", type=float, default=FUNC_RATIO_LOW, help=f"Ratio lower bound (default {FUNC_RATIO_LOW})"
    )
    parser.add_argument(
        "--high", type=float, default=FUNC_RATIO_HIGH, help=f"Ratio upper bound (default {FUNC_RATIO_HIGH})"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 1
    with open(input_path) as f:
        data = json.load(f)
    if "results" not in data:
        print(f"error: {input_path} is not a v2 DSE result (missing 'results')", file=sys.stderr)
        return 1

    func_ms = args.func_model_ms
    rows = [r for r in data["results"] if _is_target_row(r.get("config_label", ""))]
    if not rows:
        print("error: no Block 64x64 @1GHz LPDDR5-64b rows found; all block rows:", file=sys.stderr)
        for r in data["results"]:
            if (r.get("config_label") or "").startswith("bloc"):
                print(f"  {r['config_label']}", file=sys.stderr)
        return 1

    lines = [
        f"Gate 1b evidence extraction (Func Model reference = {func_ms} ms)",
        f"source: {input_path}",
        f"sha256: {_sha256(input_path)}",
        "",
    ]
    wc_true = None
    wc_false = None
    for r in sorted(rows, key=lambda r: r["config_label"]):
        label = r["config_label"]
        ttft = r["metrics"]["ttft_ms"]
        wc = "WC" in label
        ratio = func_ms / ttft if ttft > 0 else float("inf")
        lines.append(f"{label}: ttft_ms={ttft:.2f}  ratio={ratio:.3f}")
        if wc:
            wc_true = ttft
        else:
            wc_false = ttft

    if wc_true is None:
        print("error: missing wc=True Block 64x64 row", file=sys.stderr)
        return 1

    ratio_true = func_ms / wc_true
    passed = args.low <= ratio_true <= args.high
    lines.append("")
    lines.append(f"Gate 1b ratio (wc=True) = {ratio_true:.3f}")
    lines.append(f"PASS: {args.low} <= {ratio_true:.3f} <= {args.high} -> {passed}")
    if wc_false is not None:
        lines.append(f"Reference ratio (wc=False) = {func_ms / wc_false:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
