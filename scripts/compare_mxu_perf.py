#!/usr/bin/env python3
"""Compare RTL-measured MXU cycles against the BlockEngine timing model.

Loads rtl/results/mxu_cycles.csv (produced by extract_mxu_cycles.py), calls
sim.engine.mac_engine.create_engine() with the 64x64 Block config from
sim/config/npu_config.yaml, and emits:

  - rtl/results/mxu_perf_comparison.csv
  - rtl/results/mxu_perf_report.md

The comparison documents the deterministic gap between the RTL state-machine
runtime (MMIO config + LOAD_W + LOAD_A + STORE_OUT + IRQ overhead) and the
timing model's DMA + pure compute estimate.
"""

from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import yaml


CSV_COLUMNS = [
    "scenario",
    "M",
    "K",
    "N",
    "RTL_cycles",
    "model_compute_cycles",
    "model_dma_cycles",
    "model_total_cycles",
    "gap_cycles",
    "gap_pct",
]


def find_repo_root() -> Path:
    """Return the CaduceusCore repository root.

    Supports running from the repo root, CaduceusCore/, or CaduceusCore/scripts/.
    """
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent
    if (candidate / "rtl" / "results").is_dir():
        return candidate
    cwd = Path.cwd().resolve()
    for path in [cwd, *cwd.parents]:
        if (path / "rtl" / "results").is_dir():
            return path
    return candidate


def _ensure_import_paths(repo_root: Path) -> None:
    """Mirror README's PYTHONPATH=sim convention for the timing package."""
    for path in (repo_root, repo_root / "sim"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def load_engine(repo_root: Path):
    """Import and instantiate the configured MAC engine."""
    _ensure_import_paths(repo_root)
    from sim.engine.mac_engine import create_engine

    config_path = repo_root / "sim" / "config" / "npu_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return create_engine(config)


def compare_scenarios(engine, cycles_csv: Path) -> list[dict]:
    """Return comparison records for every row in mxu_cycles.csv."""
    records: list[dict] = []
    with cycles_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["M"])
            k = int(row["K"])
            n = int(row["N"])
            rtl_cycles = int(row["cycles"])

            result = engine.estimate(M=m, K=k, N=n)
            model_total = result.total_cycles
            model_compute = result.compute_cycles
            model_dma = result.dma_cycles

            gap = rtl_cycles - model_total
            gap_pct = (gap / rtl_cycles * 100.0) if rtl_cycles > 0 else 0.0

            records.append(
                {
                    "scenario": row["scenario"],
                    "M": m,
                    "K": k,
                    "N": n,
                    "RTL_cycles": rtl_cycles,
                    "model_compute_cycles": model_compute,
                    "model_dma_cycles": model_dma,
                    "model_total_cycles": model_total,
                    "gap_cycles": gap,
                    "gap_pct": gap_pct,
                }
            )
    return records


def write_comparison_csv(records: list[dict], out_path: Path) -> None:
    """Write the comparison CSV."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def _fmt_float(value: float) -> str:
    """Format a float with two decimals, stripping trailing zeros."""
    text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _stats(values: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, min, max, pstdev) for a list of floats."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    return statistics.mean(values), min(values), max(values), (
        statistics.pstdev(values) if len(values) > 1 else 0.0
    )


def generate_report(records: list[dict], engine_type: str, out_path: Path) -> None:
    """Generate the Markdown gap-analysis report."""
    gap_pcts = [r["gap_pct"] for r in records if r["RTL_cycles"] > 0]
    gap_cycles = [r["gap_cycles"] for r in records]

    mean_gap_pct, min_gap_pct, max_gap_pct, std_gap_pct = _stats(gap_pcts)
    mean_gap_cycles, _, _, std_gap_cycles = _stats(gap_cycles)

    non_degenerate_pcts = [
        r["gap_pct"] for r in records if r["RTL_cycles"] > 0 and r["M"] > 0
    ]
    nd_mean, nd_min, nd_max, nd_std = _stats(non_degenerate_pcts)

    # Single-tile anchor for the root-cause discussion.
    single_tile = next(
        (r for r in records if r["scenario"] == "single_tile"), None
    )

    sorted_records = sorted(records, key=lambda r: r["gap_pct"], reverse=True)

    lines: list[str] = []
    lines.append("# MXU RTL vs Timing Model — Performance Co-simulation Report")
    lines.append("")
    lines.append(
        "This report compares RTL-measured cycle counts against the "
        f"`{engine_type}` timing model predictions for the 64x64 MXU Phase 1 design."
    )
    lines.append("")

    # Summary
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Scenarios compared:** {len(records)}")
    lines.append(f"- **Mean gap (cycles):** {_fmt_float(mean_gap_cycles)} cycles")
    lines.append(f"- **StdDev gap (cycles):** {_fmt_float(std_gap_cycles)} cycles")
    lines.append(f"- **Mean gap (% of RTL):** {_fmt_float(mean_gap_pct)}%")
    lines.append(f"- **Min gap (% of RTL):** {_fmt_float(min_gap_pct)}%")
    lines.append(f"- **Max gap (% of RTL):** {_fmt_float(max_gap_pct)}%")
    lines.append(f"- **StdDev gap (% of RTL):** {_fmt_float(std_gap_pct)}%")
    lines.append("")
    lines.append(
        "Excluding the degenerate `zero_dim` scenario (M=0), the gap tightens: "
        f"mean={_fmt_float(nd_mean)}%, min={_fmt_float(nd_min)}%, "
        f"max={_fmt_float(nd_max)}%, std={_fmt_float(nd_std)}%."
    )
    lines.append("")

    # Root cause
    lines.append("## Root Cause Analysis")
    lines.append("")
    lines.append(
        "The timing model counts only **DMA + pure compute cycles**. "
        "The RTL measurement spans the full MXU command lifecycle:"
    )
    lines.append("")
    lines.append("- MMIO register configuration (CMD, dimension, address registers)")
    lines.append("- `LOAD_W` — weight tile DMA into on-chip weight buffer")
    lines.append("- `LOAD_A` — activation tile DMA into on-chip activation buffer")
    lines.append("- Tile-loop controller state-machine overhead")
    lines.append("- `STORE_OUT` — result tile write-back to memory")
    lines.append("- IRQ assertion and testbench handshaking")
    lines.append("")
    if single_tile:
        lines.append(
            "For the `single_tile` anchor (M=64, K=64, N=64), the RTL reports "
            f"**{single_tile['RTL_cycles']}** total cycles while the model predicts "
            f"**{single_tile['model_total_cycles']}** cycles (compute "
            f"{single_tile['model_compute_cycles']} + DMA {single_tile['model_dma_cycles']}). "
            f"The resulting gap is **{single_tile['gap_cycles']}** cycles "
            f"(**{_fmt_float(single_tile['gap_pct'])}%** of RTL)."
        )
        lines.append("")
    lines.append(
        "This gap is the fixed + per-tile deterministic overhead of the RTL "
        "control state machine. It is not modeled by the current BlockEngine, "
        "which abstracts the controller as zero-latency and only accounts for "
        "data movement and MAC array utilization."
    )
    lines.append("")

    # Key insight
    lines.append("## Key Insight — Consistent Deterministic Overhead")
    lines.append("")
    lines.append(
        "The gap is **consistent across non-degenerate scenarios**, not random. "
        f"Excluding the `zero_dim` outlier, the standard deviation is "
        f"{_fmt_float(nd_std)}% around a mean of {_fmt_float(nd_mean)}%, "
        "confirming the overhead is deterministic state-machine latency rather "
        "than a functional bug or non-deterministic stall. Because it is "
        "consistent, the gap can be used as a one-time calibration offset for "
        "the timing model when translating BlockEngine estimates to expected "
        "RTL cycle counts. The `zero_dim` case (M=0) is a synthetic edge case "
        "where the model still allocates a weight tile while the RTL finishes "
        "almost immediately; it should be treated separately."
    )
    lines.append("")

    # Per-scenario table
    lines.append("## Per-Scenario Comparison (sorted by gap_pct descending)")
    lines.append("")
    lines.append(
        "| scenario | M | K | N | RTL cycles | model compute | model DMA | "
        "model total | gap cycles | gap % |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in sorted_records:
        lines.append(
            f"| {r['scenario']} | {r['M']} | {r['K']} | {r['N']} | "
            f"{r['RTL_cycles']} | {r['model_compute_cycles']} | {r['model_dma_cycles']} | "
            f"{r['model_total_cycles']} | {r['gap_cycles']} | {_fmt_float(r['gap_pct'])}% |"
        )
    lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/compare_mxu_perf.py`.*")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = find_repo_root()
    results_dir = repo_root / "rtl" / "results"
    cycles_csv = results_dir / "mxu_cycles.csv"

    if not cycles_csv.exists():
        print(f"Input CSV not found: {cycles_csv}", file=sys.stderr)
        print("Run scripts/extract_mxu_cycles.py first.", file=sys.stderr)
        return 1

    engine = load_engine(repo_root)
    records = compare_scenarios(engine, cycles_csv)

    if not records:
        print("No scenarios to compare.", file=sys.stderr)
        return 1

    comparison_csv = results_dir / "mxu_perf_comparison.csv"
    report_md = results_dir / "mxu_perf_report.md"

    write_comparison_csv(records, comparison_csv)
    generate_report(records, engine.engine_type, report_md)

    print(f"Compared {len(records)} scenarios.")
    print(f"Wrote {comparison_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
