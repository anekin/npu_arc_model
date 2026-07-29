# ruff: noqa: E402
#!/usr/bin/env python3
"""Model-integrity verifier.

Uses AST/registry introspection and focused counterexample execution to detect:
  - utilization clamp (min(utilization, 1.0))
  - unregistered calibration constants
  - duplicate engine/operator registries
  - unknown-op zero-cycle fallback
  - direct reads of legacy bandwidth_bytes_per_cycle
  - diagnostic skips

Outputs structured JSON and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

# Make sim/ importable for counterexamples.
SIM_DIR = Path(__file__).resolve().parent.parent / "sim"
sys.path.insert(0, str(SIM_DIR))

from calibration.evaluate import calibration_ids_for_design_point
from calibration.registry import CalibrationRegistry
from contracts.errors import UnsupportedOperatorError
from engine.registry import canonical_engine_ids, create_engine_by_type
from workloads.operators import DEFAULT_REGISTRY

PRODUCTION_GLOBS = [
    "engine/*.py",
    "models/*.py",
    "dse/*.py",
    "workloads/*.py",
    "scheduler/*.py",
    "cv/*.py",
]


def _files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in PRODUCTION_GLOBS:
        files.extend((project_root / "sim").glob(pattern))
    return files


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def find_utilization_clamps(files: list[Path]) -> list[str]:
    """Detect likely utilization clamp patterns in source."""
    violations: list[str] = []
    for path in files:
        source = _source(path)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "min":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and arg.value == 1.0:
                            violations.append(f"{path}:{node.lineno}: min(..., 1.0) clamp")
                        if (
                            isinstance(arg, ast.Constant)
                            and arg.value == 1
                            and any(isinstance(a, ast.Name) and "util" in a.id.lower() for a in node.args)
                        ):
                            violations.append(f"{path}:{node.lineno}: min(utilization, 1) clamp")
    return violations


def find_legacy_bandwidth_reads(files: list[Path]) -> list[str]:
    """Detect direct reads of legacy bandwidth_bytes_per_cycle field."""
    violations: list[str] = []
    for path in files:
        source = _source(path)
        if "bandwidth_bytes_per_cycle" in source:
            for lineno, line in enumerate(source.splitlines(), start=1):
                if "bandwidth_bytes_per_cycle" in line:
                    violations.append(f"{path}:{lineno}: legacy bandwidth_bytes_per_cycle read")
    return violations


def find_diagnostic_skips(files: list[Path]) -> list[str]:
    """Detect pytest.skip related to diagnostics."""
    violations: list[str] = []
    for path in files:
        if not path.name.startswith("test_"):
            continue
        source = _source(path)
        if "pytest.skip" in source:
            for lineno, line in enumerate(source.splitlines(), start=1):
                if "pytest.skip" in line and ("diagnostic" in line.lower() or "detail" in line.lower()):
                    violations.append(f"{path}:{lineno}: diagnostic skip")
    return violations


def check_duplicate_registries() -> list[str]:
    """Check that canonical registry IDs are unique."""
    violations: list[str] = []
    engine_ids = canonical_engine_ids()
    if len(engine_ids) != len(set(engine_ids)):
        violations.append("engine.registry: duplicate canonical engine IDs")

    # Operator registry should not have duplicates; __init__ already asserts this.
    ops = DEFAULT_REGISTRY
    seen: dict[str, int] = {}
    for op in ops.modeled_ops | ops.free_fused_ops | ops.unsupported_ops:
        seen[op] = seen.get(op, 0) + 1
    duplicates = [op for op, count in seen.items() if count > 1]
    if duplicates:
        violations.append(f"operator.registry: duplicate operator entries {duplicates}")
    return violations


def check_unregistered_calibration_constants() -> list[str]:
    """Every calibration ID consumed by a design point must exist in registry."""
    violations: list[str] = []
    registry = CalibrationRegistry.from_yaml()
    base_config = {
        "mac_engine": {"type": "block", "array_height": 64, "array_width": 64},
        "memory": {"type": "LPDDR5-6400", "bandwidth_gbps": 51.2},
    }
    for engine in canonical_engine_ids():
        cfg = dict(base_config)
        cfg["mac_engine"] = dict(cfg["mac_engine"])
        cfg["mac_engine"]["type"] = engine
        ids = calibration_ids_for_design_point(cfg)
        for cid in ids:
            if registry.lookup(cid) is None:
                violations.append(f"calibration: unregistered ID {cid} for engine {engine}")
    return violations


def counterexample_unknown_op_zero_cycle() -> list[str]:
    """Unknown operators must raise, not return zero cycles."""
    violations: list[str] = []
    try:
        DEFAULT_REGISTRY.lookup("definitely_unknown_op_12345")
        violations.append("operator.registry: unknown op did not raise")
    except UnsupportedOperatorError:
        pass
    except Exception as exc:  # noqa: BLE001
        violations.append(f"operator.registry: unknown op raised wrong type {type(exc).__name__}")
    return violations


def counterexample_utilization_bounds() -> list[str]:
    """Engine utilization must stay in (0, 1]."""
    violations: list[str] = []
    base_config = {
        "mac_engine": {"type": "block", "array_height": 64, "array_width": 64, "frequency_mhz": 1000},
        "memory": {"type": "LPDDR5-6400", "bandwidth_gbps": 51.2, "dram_efficiency": 0.85},
        "sram": {"l2_shared_kb": 2048},
        "optimizations": {"weight_cache": False, "dma_bw_multiplier": 1.0},
    }
    for engine in ("block", "systolic"):
        cfg = dict(base_config)
        cfg["mac_engine"] = dict(cfg["mac_engine"])
        cfg["mac_engine"]["type"] = engine
        eng = create_engine_by_type(engine, cfg)
        result = eng.estimate(1024, 1024, 1024)
        if not (0.0 < result.utilization <= 1.0):
            violations.append(f"engine.{engine}: utilization {result.utilization} out of (0,1] for M=1024")
    return violations


def verify_model_integrity(project_root: Path) -> dict[str, Any]:
    files = _files(project_root)
    clamps = find_utilization_clamps(files)
    legacy_bw = find_legacy_bandwidth_reads(files)
    skips = find_diagnostic_skips(files)
    duplicates = check_duplicate_registries()
    unregistered = check_unregistered_calibration_constants()
    unknown_op = counterexample_unknown_op_zero_cycle()
    util_bounds = counterexample_utilization_bounds()

    violations = clamps + legacy_bw + skips + duplicates + unregistered + unknown_op + util_bounds
    verdict = "PASS" if not violations else "FAIL"

    return {
        "verdict": verdict,
        "violations": violations,
        "utilization_clamps": clamps,
        "legacy_unit_reads": legacy_bw,
        "diagnostic_skips": skips,
        "duplicate_registries": duplicates,
        "unregistered_constants": unregistered,
        "fail_open_paths": unknown_op,
        "utilization_bound_violations": util_bounds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify model integrity")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = verify_model_integrity(args.project_root)

    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
