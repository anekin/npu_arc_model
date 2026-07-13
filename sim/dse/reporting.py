"""Human- and machine-readable per-engine DSE comparison reports."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from dse.evaluator import ranking_key, target_violation_score, violation_score
from dse.types import DSEPoint


ENGINE_ORDER = [
    "systolic",
    "os_systolic",
    "block",
    "tensor_core",
    "wmma",
    "gmma",
    "input_stationary",
    "fsa",
]


def _engine_names(results: Iterable[DSEPoint]) -> List[str]:
    present = {point.config.get("engine", "") for point in results}
    present.discard("")
    ordered = [name for name in ENGINE_ORDER if name in present]
    return ordered + sorted(present - set(ordered))


def build_engine_comparison(
    results: List[DSEPoint], scenario: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Select and rank one representative point for every searched engine.

    A feasible engine is represented by its best point under the scenario's
    objective order. An engine with no feasible point is represented by its
    closest point under normalized hard-constraint distance.
    """
    rows: List[Dict[str, Any]] = []
    for engine in _engine_names(results):
        candidates = [p for p in results if p.config.get("engine") == engine]
        feasible = [p for p in candidates if p.constraints_passed]
        if feasible:
            selected = min(feasible, key=lambda p: ranking_key(p, scenario))
            status = "PASS"
            selection = "scenario_objective"
            distance = 0.0
            sort_key = (0, *ranking_key(selected, scenario))
        else:
            selected = min(candidates, key=lambda p: violation_score(p, scenario))
            status = "FAIL"
            selection = "closest_to_constraints"
            distance = violation_score(selected, scenario)
            sort_key = (1, distance, -selected.decode_tps)

        rows.append({
            "engine": engine,
            "status": status,
            "selection": selection,
            "evaluated_configs": len(candidates),
            "feasible_configs": len(feasible),
            "violation_score": distance,
            "target_status": (
                "MET" if target_violation_score(selected, scenario) == 0 else "MISS"
            ),
            "target_violation_score": target_violation_score(selected, scenario),
            "config_label": selected.config_label,
            "config": selected.config,
            "metrics": {
                "decode_tps": selected.decode_tps,
                "aggregate_tps": selected.aggregate_tps,
                "prefill_tps": selected.prefill_tps,
                "ttft_ms": selected.ttft_ms,
                "itl_ms": selected.itl_ms,
                "e2e_latency_ms": selected.e2e_latency_ms,
                "area_mm2": selected.area_mm2,
                "power_w": selected.power_w,
                "tops_int8": selected.tops_int8,
                "bandwidth_util_pct": selected.bandwidth_util_pct,
                "memory_required_gb": selected.memory_required_gb,
                "memory_available_gb": selected.memory_available_gb,
            },
            "failed_reasons": selected.failed_reasons,
            "warnings": selected.warnings,
            "_sort_key": sort_key,
        })

    rows.sort(key=lambda row: row["_sort_key"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row.pop("_sort_key", None)
    return rows


def print_engine_comparison(
    rows: List[Dict[str, Any]], cv_mode: bool = False,
) -> None:
    """Print every searched engine, including closest failed candidates."""
    if not rows:
        return
    print("\n  Engine comparison (best feasible or closest failed point):")
    if cv_mode:
        print(f"  {'#':>2} {'Engine':<18} {'Status':<6} {'Config':<36} "
              f"{'FPS':>9} {'Area':>8} {'Power':>8}")
        print(f"  {'-'*92}")
        for row in rows:
            m = row["metrics"]
            print(f"  {row['rank']:>2} {row['engine']:<18} {row['status']:<6} "
                  f"{row['config_label'][:36]:<36} {m['decode_tps']:>9.2f} "
                  f"{m['area_mm2']:>7.1f} {m['power_w']:>7.1f}")
    else:
        print("  Units: TPS=tok/s, latency=ms, area=mm2, power=W")
        print(f"  {'#':>2} {'Engine':<16} {'Stat':<4} {'Tgt':<4} {'Config':<26} "
              f"{'Dec':>7} {'Agg':>7} {'Pre':>8} {'TTFT':>7} {'ITL':>7} {'Area':>6} {'Pwr':>6}")
        print(f"  {'-'*113}")
        for row in rows:
            m = row["metrics"]
            print(f"  {row['rank']:>2} {row['engine']:<16} {row['status']:<4} "
                  f"{row['target_status']:<4} "
                  f"{row['config_label'][:26]:<26} {m['decode_tps']:>7.2f} "
                  f"{m['aggregate_tps']:>7.2f} {m['prefill_tps']:>8.1f} "
                  f"{m['ttft_ms']:>7.1f} {m['itl_ms']:>7.1f} "
                  f"{m['area_mm2']:>6.1f} {m['power_w']:>6.1f}")

    failed = [row for row in rows if row["status"] == "FAIL"]
    if failed:
        print("\n  Failed engine details:")
        for row in failed:
            reasons = "; ".join(row["failed_reasons"]) or "no feasible point"
            print(f"    {row['engine']}: {reasons}")
