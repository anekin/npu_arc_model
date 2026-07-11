"""CV PPA Report — generates Markdown and JSON reports for CV simulation results."""

from dataclasses import dataclass
from typing import List
import json
import os


@dataclass
class CVPpaReport:
    """Performance, Power, Area report for a single CV engine configuration.

    Attributes:
        fps: Frames per second (computed as 1e9 / total_cycles at 1 GHz).
        area_mm2: Estimated silicon area in mm².
        power_w: Estimated power consumption in watts.
        sram_spill_mb: SRAM spillover traffic in MB.
        depthwise_util_pct: Average depthwise layer utilization (%).
        total_macs: Total multiply-accumulate operations.
        total_cycles: Total execution cycles.
        engine_type: Engine architecture label (e.g. "systolic", "block").
    """
    fps: float
    area_mm2: float
    power_w: float
    sram_spill_mb: float
    depthwise_util_pct: float
    total_macs: float
    total_cycles: int
    engine_type: str


def generate_report(results: List[CVPpaReport], output_dir: str) -> None:
    """Generate CV PPA report in Markdown and JSON formats.

    Creates ``cv_ppa_report.md`` and ``cv_ppa_report.json`` under *output_dir*.
    The directory is created if it does not exist.

    Args:
        results: CVPpaReport instances to include in the report.
        output_dir: Target directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Aggregate MACs for the ONNX validation section
    total_macs_arc = sum(r.total_macs for r in results)

    # --- Markdown Report ---
    md_lines: List[str] = []
    md_lines.append("# CV PPA Report — MobileNetV3-Small")
    md_lines.append("")
    md_lines.append("## Engine Comparison")
    md_lines.append("")
    md_lines.append("| Engine | FPS | Area (mm²) | Power (W) | SRAM Spill (MB) | DW Util (%) |")
    md_lines.append("|--------|-----|------------|-----------|-----------------|-------------|")
    for r in results:
        md_lines.append(
            f"| {r.engine_type} | {r.fps:.1f} | {r.area_mm2:.2f} | "
            f"{r.power_w:.2f} | {r.sram_spill_mb:.2f} | {r.depthwise_util_pct:.1f} |"
        )

    md_lines.append("")
    md_lines.append("## Pareto Frontier")
    md_lines.append("")
    md_lines.append("(Best per engine sorted by FPS)")
    md_lines.append("")
    md_lines.append("| Engine | FPS | Area (mm²) | Power (W) | SRAM Spill (MB) | DW Util (%) |")
    md_lines.append("|--------|-----|------------|-----------|-----------------|-------------|")

    # Pareto frontier: best FPS per engine type
    best_per_engine: dict = {}
    for r in results:
        if r.engine_type not in best_per_engine or r.fps > best_per_engine[r.engine_type].fps:
            best_per_engine[r.engine_type] = r

    sorted_best = sorted(best_per_engine.values(), key=lambda x: x.fps, reverse=True)
    for r in sorted_best:
        md_lines.append(
            f"| {r.engine_type} | {r.fps:.1f} | {r.area_mm2:.2f} | "
            f"{r.power_w:.2f} | {r.sram_spill_mb:.2f} | {r.depthwise_util_pct:.1f} |"
        )

    md_lines.append("")
    md_lines.append("## ONNX Runtime Validation")
    md_lines.append("")
    md_lines.append("| Metric | Arc Model | ONNX Runtime | Delta |")
    md_lines.append("|--------|-----------|-------------|-------|")
    md_lines.append(f"| Total MACs | {total_macs_arc:.0f} | — | — |")
    md_lines.append("")

    md_content = "\n".join(md_lines)

    # --- JSON Report ---
    json_data = {
        "report": "CV PPA Report — MobileNetV3-Small",
        "engine_comparison": [
            {
                "engine": r.engine_type,
                "fps": r.fps,
                "area_mm2": r.area_mm2,
                "power_w": r.power_w,
                "sram_spill_mb": r.sram_spill_mb,
                "depthwise_util_pct": r.depthwise_util_pct,
                "total_macs": r.total_macs,
                "total_cycles": r.total_cycles,
            }
            for r in results
        ],
        "pareto_frontier": [
            {
                "engine": r.engine_type,
                "fps": r.fps,
                "area_mm2": r.area_mm2,
                "power_w": r.power_w,
                "sram_spill_mb": r.sram_spill_mb,
                "depthwise_util_pct": r.depthwise_util_pct,
            }
            for r in sorted_best
        ],
        "onnx_validation": {
            "arc_model_macs": total_macs_arc,
            "onnx_runtime_macs": None,
            "delta_pct": None,
        },
    }

    # Write files
    md_path = os.path.join(output_dir, "cv_ppa_report.md")
    json_path = os.path.join(output_dir, "cv_ppa_report.json")

    with open(md_path, "w") as f:
        f.write(md_content)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"Report written to {md_path}")
    print(f"Report written to {json_path}")
