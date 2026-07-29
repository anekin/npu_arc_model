"""Scenario-driven design-space enumeration and coverage manifest.

Public API:
* ``DesignSpace`` — reads orthogonal axes from ``sim/config/dse_axes.yaml`` and
  generates ``DesignPoint`` candidates bound to a ``Scenario``.
* ``DesignPoint`` — normalized hardware config + scenario/workload refs +
  stable SHA-256 identity.
* ``CoverageManifest`` — tracks requested/generated/evaluated/successful/
  pruned/failed/missing per axis and enforces counting invariants.
"""

from __future__ import annotations

from dse.manifest import (
    AxisCoverage,
    CoverageManifest,
    ExclusionRecord,
)
from dse.models import AxisSpec, Constraint, DesignPoint
from dse.space import DesignSpace, GenerationResult

__all__ = [
    "AxisCoverage",
    "AxisSpec",
    "Constraint",
    "CoverageManifest",
    "DesignPoint",
    "DesignSpace",
    "ExclusionRecord",
    "GenerationResult",
]
