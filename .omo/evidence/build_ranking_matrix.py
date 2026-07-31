#!/usr/bin/env python3
"""Build 8x4x2 ranking matrix from DSE results for lpddr5_3b and onchip_7b.

Regenerates the design spaces to map design_point_ids to axis values
(process_node, engine, etc.), then merges with result metrics.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = REPO_ROOT / "sim"
EVIDENCE_DIR = REPO_ROOT / ".omo" / "evidence"

sys.path.insert(0, str(SIM_DIR))

from design_space_explorer import _resolve_scenario
from dse.space import load_design_space_from_yaml

SCENARIOS = ["lpddr5_3b", "onchip_7b"]
RESULT_FILES = {
    "lpddr5_3b": "task-2-cross-node-all-engines-dse-lpddr5.json",
    "onchip_7b": "task-3-cross-node-all-engines-dse-onchip.json",
}
NODES = [7, 12, 22, 28]
AXES_PATH = SIM_DIR / "config" / "dse_axes.yaml"

OUTPUT_JSON = EVIDENCE_DIR / "task-4-cross-node-all-engines-dse-matrix.json"
OUTPUT_MD = EVIDENCE_DIR / "task-4-cross-node-all-engines-dse-matrix.md"


def main() -> int:
    mapping: dict[str, dict[str, object]] = {}  # dpid -> axis_values

    all_results: dict[str, list[dict]] = {}

    for scenario_name in SCENARIOS:
        scenario = _resolve_scenario(scenario_name)
        ds = load_design_space_from_yaml(scenario, AXES_PATH, mode="ci_all_axes")
        gen = ds.generate_with_exclusions()
        for p in gen.points:
            mapping[p.design_point_id] = dict(p.axis_values)

        fpath = EVIDENCE_DIR / RESULT_FILES[scenario_name]
        data = json.loads(fpath.read_text(encoding="utf-8"))
        all_results[scenario_name] = data["results"]

    # Build best per (scenario, node, engine)
    best: dict[tuple[str, int, str], dict[str, float]] = {}
    for scenario_name in SCENARIOS:
        for r in all_results[scenario_name]:
            if r["status"] != "complete":
                continue
            dpid = r["design_point_id"]
            axes = mapping.get(dpid)
            if axes is None:
                continue
            node = axes.get("process_node")
            engine = axes.get("engine")
            if node is None or engine is None:
                continue
            tok = r["metrics"]["tok_per_s"]
            area = r["metrics"]["area_mm2"]
            key = (scenario_name, int(node), engine)
            if key not in best or tok > best[key]["tok_per_s"]:
                best[key] = {"tok_per_s": tok, "area_mm2": area}

    # Count engine coverage per (scenario, node)
    engines_per_node: dict[tuple[str, int], set[str]] = defaultdict(set)
    for sc, node, eng in best:
        engines_per_node[(sc, node)].add(eng)

    # Check for empty nodes (all engines filtered)
    negative_findings: list[str] = []
    for sc in SCENARIOS:
        for node in NODES:
            count = len(engines_per_node.get((sc, node), set()))
            if count == 0:
                negative_findings.append(f"FAIL: {sc} node={node}nm has 0 valid engines")
            elif count < 3:
                negative_findings.append(f"WARNING: {sc} node={node}nm has only {count} engines")

    if negative_findings:
        for nf in negative_findings:
            print(nf)
        return 1

    # Build the matrix
    all_engines = sorted(set(k[2] for k in best))
    matrix: dict[str, dict[str, dict[str, float | None]]] = {}  # engine -> node_label -> {tok, area}

    for engine in all_engines:
        matrix[engine] = {}
        for node in NODES:
            for sc in SCENARIOS:
                col = f"{sc}_{node}nm"
                key = (sc, node, engine)
                if key in best:
                    matrix[engine][col] = best[key]
                else:
                    matrix[engine][col] = None

    # Write JSON
    output_data = {
        "description": "8-engine x 4-node ranking matrix for lpddr5_3b and onchip_7b",
        "scenarios": SCENARIOS,
        "nodes": NODES,
        "engines": all_engines,
        "matrix": {
            engine: {
                col: ({"tok_per_s": v["tok_per_s"], "area_mm2": v["area_mm2"]} if v else None)
                for col, v in cols.items()
            }
            for engine, cols in matrix.items()
        },
    }
    OUTPUT_JSON.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    # Write Markdown
    md_lines: list[str] = []
    md_lines.append("# 8-Engine x 4-Node Ranking Matrix\n")
    md_lines.append(
        "**Scenarios**: `lpddr5_3b` (LPDDR5 51.2 GB/s, Qwen2.5-3B INT4) "
        "and `onchip_7b` (On-chip 3D DRAM 500 GB/s, Qwen2.5-7B INT4)\n"
    )
    md_lines.append("**Mode**: `ci-all-axes` (engine x process_node cross-product)\n")
    md_lines.append(f"**Engines**: {', '.join(all_engines)}\n")
    md_lines.append("**Cells**: tok/s (primary) | area_mm² (secondary)\n")

    col_headers = [f"{sc}_{node}nm" for sc in SCENARIOS for node in NODES]
    header = "| Engine | " + " | ".join(col_headers) + " |"
    sep = "|:---|" + "|".join(":---:|" for _ in col_headers)
    md_lines.append(header)
    md_lines.append(sep)

    for engine in all_engines:
        cells: list[str] = []
        for node in NODES:
            for sc in SCENARIOS:
                val = matrix[engine].get(f"{sc}_{node}nm")
                if val is None:
                    cells.append("constraint-filtered")
                else:
                    cells.append(f"{val['tok_per_s']:.1f} / {val['area_mm2']:.1f}")
        row = f"| {engine} | " + " | ".join(cells) + " |"
        md_lines.append(row)

    md_lines.append("")
    md_lines.append("## Observations\n")

    # Block BW-bound check (observation, not pass/fail)
    lpddr5_block_toks = [best.get(("lpddr5_3b", n, "block"), {}).get("tok_per_s", 0) for n in NODES]
    if lpddr5_block_toks and max(lpddr5_block_toks) > 0:
        max_tok = max(lpddr5_block_toks)
        min_tok = min(t for t in lpddr5_block_toks if t > 0)
        variation_pct = (max_tok - min_tok) / max_tok * 100 if max_tok > 0 else 0
        md_lines.append(
            f"1. **Block BW-bound tendency**: lpddr5_3b block tok/s range "
            f"{min_tok:.1f}–{max_tok:.1f} ({variation_pct:.0f}% variation across nodes) "
            f"— {'consistent with BW-bound behavior' if variation_pct < 5 else f'variation exceeds expected 5% BW-bound threshold'}\n"
        )

    # FSA compute-bound observation
    fsa_28 = best.get(("lpddr5_3b", 28, "fsa"), {}).get("tok_per_s", 0)
    fsa_7 = best.get(("lpddr5_3b", 7, "fsa"), {}).get("tok_per_s", 0)
    if fsa_7 > 0:
        md_lines.append(
            f"2. **FSA compute-bound**: tok/s drops from {fsa_7:.1f} (7nm) "
            f"to {fsa_28:.1f} (28nm) — frequency-bound at older nodes\n"
        )

    # GMMA/block on onchip high BW
    gmma_onchip = best.get(("onchip_7b", 7, "gmma"), {}).get("tok_per_s", 0)
    block_onchip = best.get(("onchip_7b", 7, "block"), {}).get("tok_per_s", 0)
    if gmma_onchip > 0:
        md_lines.append(
            f"3. **GMMA high-BW advantage**: onchip_7b GMMA={gmma_onchip:.1f} tok/s "
            f"vs block={block_onchip:.1f} tok/s at 7nm — GMMA+Hopper async DMA excels at high BW\n"
        )

    md_lines.append("")
    md_lines.append("## Engine Coverage per Node\n")
    for sc in SCENARIOS:
        md_lines.append(f"### {sc}\n")
        md_lines.append("| Node | Engine Count | Engines |")
        md_lines.append("|:---:|:---:|:---|")
        for node in NODES:
            eng_set = engines_per_node.get((sc, node), set())
            md_lines.append(f"| {node}nm | {len(eng_set)} | {', '.join(sorted(eng_set))} |")
        md_lines.append("")

    md_lines.append("")
    md_lines.append("---\n")
    md_lines.append("*Matrix generated by .omo/evidence/build_ranking_matrix.py*\n")

    OUTPUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    # Print summary
    for key, val in sorted(best.items()):
        sc, node, eng = key
        print(f"{sc:>12s} node={node:>2d}nm {eng:<18s} {val['tok_per_s']:>7.1f} tok/s  {val['area_mm2']:>6.1f} mm²")

    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"MD:   {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
