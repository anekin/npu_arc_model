"""Scenario-driven design-space generator.

Reads orthogonal axes from ``sim/config/dse_axes.yaml`` and produces
``DesignPoint`` candidates bound to a ``Scenario``.  Supports ``full``
(all combinations within constraints) and ``ci_all_axes`` (small coverage
set touching every axis value) modes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts.errors import ConfigError
from contracts.identity import digest_sha256
from dse.config_loader import build_axes, build_constraints, load_axes_config, load_base_config
from dse.hardware_builder import build_hardware_config
from dse.manifest import CoverageManifest, ExclusionRecord
from dse.models import Constraint, DesignPoint
from scenarios.schema import Scenario

SIM_DIR = Path(__file__).resolve().parent.parent
AXES_PATH = SIM_DIR / "config" / "dse_axes.yaml"

_TEMPORAL_AXES = {
    "queue_policy",
    "nonpreemptible_quantum_ms",
    "partition",
    "request_batch",
    "active_sequences",
    "token_block",
    "image_count",
    "action_horizon",
    "flow_steps",
    "resident_models",
    "inflight_jobs",
}


@dataclass(frozen=True)
class GenerationResult:
    """Output of ``DesignSpace.generate_with_exclusions``."""

    points: tuple[DesignPoint, ...]
    exclusions: tuple[ExclusionRecord, ...]


class DesignSpace:
    """Generate design points from a scenario and declarative axes."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        axes_config: dict[str, Any] | None = None,
        mode: str = "full",
        base_config: dict[str, Any] | None = None,
    ) -> None:
        if mode not in ("full", "ci_all_axes"):
            raise ConfigError(f"unknown DSE mode {mode!r}", field_path="mode")

        if axes_config is None:
            axes_config = load_axes_config(AXES_PATH)
        if base_config is None:
            base_path = SIM_DIR / axes_config.get("base_config_source", "config/design_space.yaml")
            base_config = load_base_config(base_path)

        self.scenario = scenario
        self.axes_config = axes_config
        self.base_config = base_config
        self.mode = mode
        self.axes = dict(sorted(build_axes(axes_config).items()))
        self.constraints = build_constraints(axes_config)
        self.defaults = dict(sorted(axes_config.get("defaults", {}).items()))
        self.reason_codes = axes_config.get("reason_codes", {})

        # Bind external bandwidth axis to the scenario's declared bandwidth, if any.
        scenario_bw = self._scenario_bandwidth_gbps()
        if scenario_bw is not None and "bandwidth_gbps" in self.axes:
            self.constraints.append(
                Constraint(
                    name="scenario_bandwidth_match",
                    when={},
                    require={"bandwidth_gbps": (scenario_bw,)},
                    reason="scenario_bandwidth_match",
                ),
            )
            self.reason_codes["scenario_bandwidth_match"] = (
                f"External bandwidth axis pinned to scenario value {scenario_bw} GB/s"
            )

    def generate(self) -> list[DesignPoint]:
        """Generate design points according to ``self.mode``."""
        return list(self.generate_with_exclusions().points)

    def generate_with_exclusions(self) -> GenerationResult:
        """Generate design points and record structured exclusions."""
        if self.mode == "full":
            combos, exclusions = self._full_combinations()
        elif self.mode == "ci_all_axes":
            combos, exclusions = self._ci_all_axes_combinations()
        else:
            raise ConfigError(f"unknown DSE mode {self.mode!r}", field_path="mode")

        points = tuple(self._build_point(combo) for combo in combos)
        return GenerationResult(points=points, exclusions=tuple(exclusions))

    def _scenario_bandwidth_gbps(self) -> float | None:
        """Return the scenario's declared external memory bandwidth, if any."""
        if self.scenario.metadata is None:
            return None
        mem = self.scenario.metadata.get("memory") or {}
        bw = mem.get("bandwidth_gbps")
        return float(bw) if bw is not None else None

    def build_manifest(self, result: GenerationResult | None = None) -> CoverageManifest:
        """Build a ``CoverageManifest`` from a generation result."""
        if result is None:
            result = self.generate_with_exclusions()
        return CoverageManifest(self.axes, result.points, result.exclusions)

    def exclusion_reason(self, reason_code: str) -> str:
        """Return human-readable text for a reason code."""
        return self.reason_codes.get(reason_code, reason_code)

    def _full_combinations(self) -> tuple[list[dict[str, Any]], list[ExclusionRecord]]:
        axis_names = list(self.axes.keys())
        value_lists = [list(self.axes[name].values) for name in axis_names]
        valid: list[dict[str, Any]] = []
        exclusions: list[ExclusionRecord] = []

        for raw_values in itertools.product(*value_lists):
            combo = dict(zip(axis_names, raw_values, strict=False))
            ok, exclusion = self._check(combo)
            if ok:
                valid.append(combo)
            elif exclusion is not None:
                exclusions.append(exclusion)
        return valid, exclusions

    def _ci_all_axes_combinations(self) -> tuple[list[dict[str, Any]], list[ExclusionRecord]]:
        seen: dict[tuple[Any, ...], dict[str, Any]] = {}
        axis_names = list(self.axes.keys())
        exclusions: list[ExclusionRecord] = []

        def add(combo: dict[str, Any]) -> None:
            key = tuple(combo.get(a) for a in axis_names)
            if key not in seen:
                seen[key] = combo

        def record_repair_exclusion(
            axis_name: str, value: Any, changes: list[tuple[str, Any, Any, Constraint]]
        ) -> None:
            for changed_axis, original, _, constraint in changes:
                if changed_axis == axis_name and original == value:
                    exclusions.append(
                        ExclusionRecord(
                            axis=axis_name,
                            value=value,
                            reason=constraint.reason,
                            constraint_name=constraint.name,
                        )
                    )
                    break

        defaults = dict(self.defaults)
        repaired_defaults, _ = self._repair(defaults)
        if self._is_valid(repaired_defaults):
            add(repaired_defaults)

        for axis_name, spec in self.axes.items():
            for value in spec.values:
                combo = dict(self.defaults)
                combo[axis_name] = value
                combo, changes = self._repair(combo)
                if self._is_valid(combo):
                    add(combo)
                    if combo.get(axis_name) != value:
                        record_repair_exclusion(axis_name, value, changes)
                else:
                    _, exclusion = self._check(combo)
                    if exclusion is not None:
                        exclusions.append(exclusion)

        return list(seen.values()), exclusions

    def _repair(self, combo: dict[str, Any]) -> tuple[dict[str, Any], list[tuple[str, Any, Any, Constraint]]]:
        combo = dict(combo)
        changes: list[tuple[str, Any, Any, Constraint]] = []
        for _ in range(10):
            changed = False
            for constraint in self.constraints:
                if not constraint.matches(combo):
                    continue
                for axis in constraint.require_violations(combo):
                    allowed = constraint.require[axis]
                    if allowed:
                        new_value = allowed[0]
                        if combo[axis] != new_value:
                            changes.append((axis, combo[axis], new_value, constraint))
                            combo[axis] = new_value
                            changed = True
            if not changed:
                break
        return combo, changes

    def _is_valid(self, combo: dict[str, Any]) -> bool:
        for constraint in self.constraints:
            if not constraint.matches(combo):
                continue
            if constraint.require_violations(combo):
                return False
            if constraint.excludes(combo):
                return False
        return True

    def _check(self, combo: dict[str, Any]) -> tuple[bool, ExclusionRecord | None]:
        for constraint in self.constraints:
            if not constraint.matches(combo):
                continue
            violations = constraint.require_violations(combo)
            if violations:
                axis = violations[0]
                return False, ExclusionRecord(
                    axis=axis,
                    value=combo.get(axis),
                    reason=constraint.reason,
                    constraint_name=constraint.name,
                )
            if constraint.excludes(combo):
                axis = next(iter(constraint.when.keys())) if constraint.when else ""
                return False, ExclusionRecord(
                    axis=axis,
                    value=combo.get(axis),
                    reason=constraint.reason,
                    constraint_name=constraint.name,
                )
        return True, None

    def _build_point(self, combo: dict[str, Any]) -> DesignPoint:
        hw = build_hardware_config(self.base_config, combo)
        temporal = {axis: combo[axis] for axis in self.axes if axis in _TEMPORAL_AXES}
        id_source = {
            "hardware": hw,
            "scenario_ref": self.scenario.name,
            "workload_ref": self.scenario.workload_ref,
            "temporal": temporal,
        }
        dp_id = digest_sha256(id_source)
        return DesignPoint(
            design_point_id=dp_id,
            hardware_config=hw,
            scenario_ref=self.scenario.name,
            workload_ref=self.scenario.workload_ref,
            axis_values=dict(combo),
        )


def load_design_space_from_yaml(
    scenario: Scenario,
    yaml_path: Path | str,
    *,
    mode: str = "full",
) -> DesignSpace:
    """Convenience constructor loading axes from a YAML file."""
    axes_config = load_axes_config(Path(yaml_path))
    return DesignSpace(scenario, axes_config=axes_config, mode=mode)


__all__ = [
    "AXES_PATH",
    "DesignPoint",
    "DesignSpace",
    "GenerationResult",
    "load_design_space_from_yaml",
]
