#!/usr/bin/env python3
"""Batch-run Arc Model DSE over the Model Zoo LLM models."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_specs import all_aliases, get_spec

SIM_DIR = Path(__file__).parent
RESULTS_DIR = (SIM_DIR / ".." / "results" / "model_zoo").resolve()
DSE = SIM_DIR / "design_space_explorer.py"


def run_dse(alias: str, batch_m: int, quick: bool, top_n: int) -> Path:
    suffix = "" if batch_m == 1 else "_m2"
    out_dir = RESULTS_DIR / alias
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pareto{suffix}.json"
    cmd = [
        sys.executable, str(DSE),
        "--model-spec", alias,
        "--batch-m", str(batch_m),
        "--top", str(top_n),
        "--output", str(out_path),
    ]
    if quick:
        cmd.append("--quick")
    subprocess.run(cmd, check=True, cwd=SIM_DIR)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use reduced DSE grid")
    parser.add_argument("--resume", action="store_true", help="Skip models with existing results")
    parser.add_argument("--top", type=int, default=30, help="Top N results per model")
    args = parser.parse_args()

    aliases = [a for a in all_aliases() if get_spec(a).model_type == "llm"]
    total = len(aliases) * 2
    completed = 0

    for alias in aliases:
        for batch_m in (1, 2):
            suffix = "" if batch_m == 1 else "_m2"
            out_path = RESULTS_DIR / alias / f"pareto{suffix}.json"
            if args.resume and out_path.exists():
                print(f"[SKIP] {alias} M={batch_m}: {out_path} exists")
                continue
            print(f"[{completed+1}/{total}] {alias} M={batch_m} ...")
            t0 = time.time()
            run_dse(alias, batch_m, args.quick, args.top)
            elapsed = time.time() - t0
            print(f"  -> {out_path} ({elapsed:.1f}s)")
            completed += 1

    print(f"\nDone. {completed} model scenarios evaluated.")


if __name__ == "__main__":
    main()
