"""Scenario-driven design-space enumeration and coverage manifest.

Public API:
* ``DesignSpace`` — reads orthogonal axes from ``sim/config/dse_axes.yaml`` and
  generates ``DesignPoint`` candidates bound to a ``Scenario``.
* ``DesignPoint`` — normalized hardware config + scenario/workload refs +
  stable SHA-256 identity.
* ``CoverageManifest`` — tracks requested/generated/evaluated/successful/
  pruned/failed/missing per axis and enforces counting invariants.
* ``ScenarioDseRunner`` — evaluates design points via ``ScenarioRunner`` and
  aggregates ``ScenarioMetrics`` into schema-v2 results.
* ``MultiObjectivePareto`` — multi-objective Pareto frontier with hard gates.
* ``write_replay_bundle`` / ``read_replay_bundle`` — reproducible replay bundles.
"""

from __future__ import annotations

from dse.legacy_adapter import evaluate_config, find_pareto, generate_configs
from dse.manifest import (
    AxisCoverage,
    CoverageManifest,
    ExclusionRecord,
)
from dse.models import AxisSpec, Constraint, DesignPoint
from dse.pareto import (
    DEFAULT_OBJECTIVES,
    GateResult,
    HardGateCode,
    MultiObjectivePareto,
    Objective,
    ObjectiveDirection,
    ParetoPoint,
    objectives_from_scenario,
)
from dse.runner import DseRunConfig, EvaluatedPoint, ScenarioDseRunner, run_scenario_dse
from dse.serialization import (
    ReplayBundlePaths,
    read_replay_bundle,
    replay_bundle_canonical_digest,
    write_replay_bundle,
)
from dse.space import DesignSpace, GenerationResult

__all__ = [
    "AxisCoverage",
    "AxisSpec",
    "Constraint",
    "CoverageManifest",
    "DesignPoint",
    "DesignSpace",
    "DseRunConfig",
    "EvaluatedPoint",
    "ExclusionRecord",
    "GateResult",
    "GenerationResult",
    "HardGateCode",
    "MultiObjectivePareto",
    "Objective",
    "ObjectiveDirection",
    "ParetoPoint",
    "ReplayBundlePaths",
    "ScenarioDseRunner",
    "DEFAULT_OBJECTIVES",
    "evaluate_config",
    "find_pareto",
    "generate_configs",
    "objectives_from_scenario",
    "read_replay_bundle",
    "replay_bundle_canonical_digest",
    "run_scenario_dse",
    "write_replay_bundle",
]
