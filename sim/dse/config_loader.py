"""YAML/config loading helpers for the DSE axis specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from contracts.errors import ConfigError
from dse.models import AxisSpec, Constraint


def load_axes_config(path: Path) -> Dict[str, Any]:
    """Load the declarative axes YAML."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"axes config {path.name} must be a mapping", field_path="")
    return raw


def load_base_config(path: Path) -> Dict[str, Any]:
    """Load the base hardware config YAML."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"base config {path.name} must be a mapping", field_path="")
    return raw


def build_axes(axes_config: Dict[str, Any]) -> Dict[str, AxisSpec]:
    """Build ordered AxisSpec mapping from config."""
    axes: Dict[str, AxisSpec] = {}
    for name, spec in axes_config.get("axes", {}).items():
        values = tuple(spec.get("values", []))
        if not values:
            raise ConfigError(f"axis {name!r} has no values", field_path=f"axes.{name}")
        axes[name] = AxisSpec(
            name=name,
            values=values,
            description=spec.get("description", ""),
            provenance=spec.get("provenance", ""),
        )
    return axes


def build_constraints(axes_config: Dict[str, Any]) -> List[Constraint]:
    """Build Constraint objects from config."""
    constraints: List[Constraint] = []
    for idx, raw in enumerate(axes_config.get("constraints", [])):
        when = _normalize_rule(raw.get("when", {}))
        require = _normalize_rule(raw.get("require", {}))
        exclude_raw = raw.get("exclude_combinations", [])
        exclude_combinations = tuple(_normalize_rule(e) for e in exclude_raw)
        constraints.append(
            Constraint(
                name=raw.get("name", f"constraint_{idx}"),
                when=when,
                require=require,
                exclude_combinations=exclude_combinations,
                reason=raw.get("reason", ""),
            )
        )
    return constraints


def _normalize_rule(rule: Dict[str, Any]) -> Dict[str, Tuple[Any, ...]]:
    """Normalize a constraint rule dict to tuple-of-values form."""
    return {
        axis: tuple(values if isinstance(values, (list, tuple)) else [values])
        for axis, values in rule.items()
    }


__all__ = [
    "build_axes",
    "build_constraints",
    "load_axes_config",
    "load_base_config",
]
