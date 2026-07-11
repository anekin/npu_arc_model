"""Product hard-constraint evaluation for DSE candidates."""

from typing import Any, Dict

from dse.types import ConstraintResult


def evaluate_constraints(
    metrics: Dict[str, float], scenario: Dict[str, Any] | None,
) -> ConstraintResult:
    if not scenario:
        return ConstraintResult()

    constraints = scenario.get("constraints", {})
    failed = []
    warnings = []
    checks = (
        ("tps_min", "tok_s", lambda actual, limit: actual >= limit,
         "TPS {actual:.2f} < required {limit:.2f}"),
        ("ttft_ms_max", "ttft_ms", lambda actual, limit: actual <= limit,
         "TTFT {actual:.2f}ms > limit {limit:.2f}ms"),
        ("area_mm2_max", "area_mm2", lambda actual, limit: actual <= limit,
         "area {actual:.2f}mm2 > limit {limit:.2f}mm2"),
        ("power_w_max", "power_w", lambda actual, limit: actual <= limit,
         "power {actual:.2f}W > limit {limit:.2f}W"),
    )
    for constraint_key, metric_key, predicate, message in checks:
        if constraint_key not in constraints:
            if constraint_key == "power_w_max":
                warnings.append("power_w_max is not specified by the scenario")
            continue
        actual = float(metrics.get(metric_key, 0.0))
        limit = float(constraints[constraint_key])
        if not predicate(actual, limit):
            failed.append(message.format(actual=actual, limit=limit))

    return ConstraintResult(
        passed=not failed,
        failed_reasons=failed,
        warnings=warnings,
    )
