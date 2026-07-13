"""Product hard-constraint evaluation for DSE candidates."""

from typing import Any, Dict

from dse.types import ConstraintResult


def evaluate_constraints(
    metrics: Dict[str, float], scenario: Dict[str, Any] | None,
) -> ConstraintResult:
    constraints = (scenario or {}).get("constraints", {})
    failed = []
    warnings = []
    checks = (
        ("tps_min", "decode_tps", lambda actual, limit: actual >= limit,
         "decode TPS {actual:.2f} < required {limit:.2f}"),
        ("decode_tps_min", "decode_tps", lambda actual, limit: actual >= limit,
         "decode TPS {actual:.2f} < required {limit:.2f}"),
        ("aggregate_tps_min", "aggregate_tps", lambda actual, limit: actual >= limit,
         "aggregate TPS {actual:.2f} < required {limit:.2f}"),
        ("prefill_tps_min", "prefill_tps", lambda actual, limit: actual >= limit,
         "prefill TPS {actual:.2f} < required {limit:.2f}"),
        ("ttft_ms_max", "ttft_ms", lambda actual, limit: actual <= limit,
         "TTFT {actual:.2f}ms > limit {limit:.2f}ms"),
        ("itl_ms_max", "itl_ms", lambda actual, limit: actual <= limit,
         "ITL {actual:.2f}ms > limit {limit:.2f}ms"),
        ("e2e_latency_ms_max", "e2e_latency_ms", lambda actual, limit: actual <= limit,
         "E2E latency {actual:.2f}ms > limit {limit:.2f}ms"),
        ("area_mm2_max", "area_mm2", lambda actual, limit: actual <= limit,
         "area {actual:.2f}mm2 > limit {limit:.2f}mm2"),
        ("power_w_max", "power_w", lambda actual, limit: actual <= limit,
         "power {actual:.2f}W > limit {limit:.2f}W"),
    )
    for constraint_key, metric_key, predicate, message in checks:
        if constraint_key not in constraints:
            continue
        actual = float(metrics.get(metric_key, 0.0))
        limit = float(constraints[constraint_key])
        if not predicate(actual, limit):
            failed.append(message.format(actual=actual, limit=limit))

    required = float(metrics.get("memory_required_gb", 0.0))
    available = float(metrics.get("memory_available_gb", 0.0))
    capacity_specified = bool(metrics.get("memory_capacity_specified", False))
    if capacity_specified and required > available:
        failed.append(
            f"memory capacity {required:.3f}GB required > {available:.3f}GB usable"
        )
    if not capacity_specified:
        warnings.append("memory capacity is unspecified; capacity feasibility was not checked")
    if float(metrics.get("concurrent_requests", 1)) > 1:
        warnings.append("prefill queueing is not included in TTFT yet")
    if scenario and "power_w_max" not in constraints:
        warnings.append("power_w_max is not specified by the scenario")

    return ConstraintResult(
        passed=not failed,
        failed_reasons=failed,
        warnings=warnings,
    )
