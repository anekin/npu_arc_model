"""Human- and machine-readable per-engine DSE comparison reports."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from dse.evaluator import ranking_key, target_violation_score, violation_score
from dse.types import DSEPoint


ENGINE_ORDER = [
    "systolic",
    "block",
    "block_fused_attention",
    "os_systolic",
    "os_systolic_fused_attention",
    "tensor_core",
    "wmma",
    "gmma",
    "input_stationary",
    "fsa",
]

WEIGHT_CACHE_ENGINES = {
    "systolic", "block", "block_fused_attention", "gmma",
}


def _engine_names(results: Iterable[DSEPoint]) -> List[str]:
    present = {point.config.get("engine", "") for point in results}
    present.discard("")
    ordered = [name for name in ENGINE_ORDER if name in present]
    return ordered + sorted(present - set(ordered))


def _weight_cache(point: DSEPoint) -> bool:
    return bool(point.config.get("weight_cache", False))


def _hardware_variant(engine: str, weight_cache: bool) -> str:
    if engine not in WEIGHT_CACHE_ENGINES:
        return "N/A"
    return "WC ON" if weight_cache else "WC OFF"


def _comparison_row(
    engine: str,
    candidates: List[DSEPoint],
    scenario: Dict[str, Any] | None,
) -> Dict[str, Any]:
    feasible = [point for point in candidates if point.constraints_passed]
    comparison_count = sum(point.comparison_eligible for point in candidates)
    product_count = sum(point.product_eligible for point in candidates)
    if feasible:
        selected = min(feasible, key=lambda point: ranking_key(point, scenario))
        status = "PASS"
        selection = "raw_scenario_objective"
        distance = 0.0
        sort_key = (0, *ranking_key(selected, scenario))
    else:
        selected = min(candidates, key=lambda point: violation_score(point, scenario))
        status = "FAIL"
        selection = "raw_closest_to_constraints"
        distance = violation_score(selected, scenario)
        sort_key = (1, distance, -selected.decode_tps)

    weight_cache = _weight_cache(selected)
    return {
        "engine": engine,
        "weight_cache": weight_cache,
        "hardware_variant": _hardware_variant(engine, weight_cache),
        "status": status,
        "selection": selection,
        "evaluated_configs": len(candidates),
        "feasible_configs": len(feasible),
        "recommendation_eligible_configs": comparison_count,
        "comparison_eligible_configs": comparison_count,
        "product_eligible_configs": product_count,
        "maturity": selected.maturity,
        "raw_exploration_eligible": selected.raw_exploration_eligible,
        "comparison_eligible": selected.comparison_eligible,
        "product_eligible": selected.product_eligible,
        "recommendation_eligible": selected.recommendation_eligible,
        "eligibility_status": (
            "PRODUCT" if selected.product_eligible
            else "COMPARE" if selected.comparison_eligible
            else "EXPLORE"
        ),
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
    }


def _rank(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows.sort(key=lambda row: row["_sort_key"])
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
        row.pop("_sort_key", None)
    return rows


def build_engine_comparison(
    results: List[DSEPoint], scenario: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Select one representative point for every searched engine."""
    rows = []
    for engine in _engine_names(results):
        candidates = [point for point in results if point.config.get("engine") == engine]
        rows.append(_comparison_row(engine, candidates, scenario))
    return _rank(rows)


def build_engine_variant_comparison(
    results: List[DSEPoint], scenario: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    """Keep WC ON and WC OFF as distinct hardware implementations."""
    rows = []
    for engine in _engine_names(results):
        engine_points = [
            point for point in results if point.config.get("engine") == engine
        ]
        for weight_cache in sorted({_weight_cache(point) for point in engine_points}):
            candidates = [
                point for point in engine_points
                if _weight_cache(point) == weight_cache
            ]
            row = _comparison_row(engine, candidates, scenario)
            row["variant"] = f"{engine} / {row['hardware_variant']}"
            rows.append(row)
    return _rank(rows)


def _print_rows(
    rows: List[Dict[str, Any]], cv_mode: bool, *, show_variant: bool,
) -> None:
    if cv_mode:
        print(f"  {'#':>2} {'Engine':<18} {'WC':<6} {'Status':<6} {'Mat':<4} "
              f"{'Config':<32} {'FPS':>9} {'Area':>8} {'Power':>8}")
        print(f"  {'-'*96}")
        for row in rows:
            metrics = row["metrics"]
            wc = row["hardware_variant"] if show_variant else "-"
            print(f"  {row['rank']:>2} {row['engine']:<18} {wc:<6} "
                  f"{row['status']:<6} {row['maturity']:<4} "
                  f"{row['config_label'][:32]:<32} {metrics['decode_tps']:>9.2f} "
                  f"{metrics['area_mm2']:>7.1f} {metrics['power_w']:>7.1f}")
        return

    print("  Units: TPS=tok/s, latency=ms, area=mm2, power=W")
    print(f"  {'#':>2} {'Engine':<16} {'WC':<6} {'Stat':<4} {'Tgt':<4} {'Mat':<4} "
          f"{'Config':<22} {'Dec':>7} {'Agg':>7} {'Pre':>8} {'TTFT':>7} "
          f"{'ITL':>7} {'Area':>6} {'Pwr':>6}")
    print(f"  {'-'*116}")
    for row in rows:
        metrics = row["metrics"]
        wc = row["hardware_variant"] if show_variant else "-"
        print(f"  {row['rank']:>2} {row['engine']:<16} {wc:<6} "
              f"{row['status']:<4} {row['target_status']:<4} "
              f"{row['maturity']:<4} {row['config_label'][:22]:<22} "
              f"{metrics['decode_tps']:>7.2f} {metrics['aggregate_tps']:>7.2f} "
              f"{metrics['prefill_tps']:>8.1f} {metrics['ttft_ms']:>7.1f} "
              f"{metrics['itl_ms']:>7.1f} {metrics['area_mm2']:>6.1f} "
              f"{metrics['power_w']:>6.1f}")


def _print_failed(rows: List[Dict[str, Any]], show_variant: bool) -> None:
    failed = [row for row in rows if row["status"] == "FAIL"]
    if not failed:
        return
    print("\n  Failed engine details:")
    for row in failed:
        reasons = "; ".join(row["failed_reasons"]) or "no feasible point"
        label = row.get("variant", row["engine"]) if show_variant else row["engine"]
        print(f"    {label}: {reasons}")


def print_engine_comparison(
    rows: List[Dict[str, Any]], cv_mode: bool = False,
) -> None:
    """Print one representative for every searched engine."""
    if not rows:
        return
    print("\n  Engine comparison (best feasible or closest failed point):")
    _print_rows(rows, cv_mode, show_variant=False)
    _print_failed(rows, show_variant=False)


def print_engine_variant_comparison(
    rows: List[Dict[str, Any]], cv_mode: bool = False,
) -> None:
    """Print separate WC ON/OFF representatives and their PPA."""
    if not rows:
        return
    print("\n  Hardware variant comparison (WC ON/OFF kept separate):")
    _print_rows(rows, cv_mode, show_variant=True)
    _print_failed(rows, show_variant=True)
