"""Full acceptance matrix for scenario-driven Arc Model DSE.

Enumerates the Cartesian space required by Todo 18:
  - 8 engines
  - M boundary shapes
  - 800/1000/1200 MHz
  - LPDDR5/LPDDR5X/HBM2e/HBM3
  - on-chip 3D full/partial/spill
  - image/action/flow dimensions
  - resident/inflight
  - 50/90/95/110% offered load
  - invalid schema/op/hash/calibration paths

The matrix is lazy: ``build_matrix`` returns an ``AcceptanceMatrix`` whose
``entries()`` generator yields one ``MatrixEntry`` at a time.  Tests and
coverage tools can sample or walk the full space without materialising it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from engine.registry import canonical_engine_ids

SIM_DIR = Path(__file__).resolve().parent.parent


class MatrixCategory(str, Enum):
    """Category of a matrix entry."""

    VALID = "valid"
    INVALID_SCHEMA = "invalid_schema"
    INVALID_OP = "invalid_op"
    INVALID_HASH = "invalid_hash"
    INVALID_CALIBRATION = "invalid_calibration"


class OnChipState(str, Enum):
    """Residency state for on-chip 3D DRAM."""

    NONE = "none"
    FULL = "full"
    PARTIAL = "partial"
    SPILL = "spill"


@dataclass(frozen=True)
class MatrixEntry:
    """One cell of the acceptance matrix."""

    engine: str
    m: int
    frequency_mhz: int
    memory_tier: str
    onchip_state: str
    image_count: int
    action_horizon: int
    flow_steps: int
    resident_models: int
    inflight_jobs: int
    load_pct: int
    category: MatrixCategory = MatrixCategory.VALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "m": self.m,
            "frequency_mhz": self.frequency_mhz,
            "memory_tier": self.memory_tier,
            "onchip_state": self.onchip_state,
            "image_count": self.image_count,
            "action_horizon": self.action_horizon,
            "flow_steps": self.flow_steps,
            "resident_models": self.resident_models,
            "inflight_jobs": self.inflight_jobs,
            "load_pct": self.load_pct,
            "category": self.category.value,
        }


@dataclass(frozen=True)
class AcceptanceMatrix:
    """Lazy full acceptance matrix."""

    engines: tuple[str, ...]
    m_boundaries: tuple[int, ...]
    frequencies_mhz: tuple[int, ...]
    memory_tiers: tuple[str, ...]
    onchip_states: tuple[str, ...]
    image_counts: tuple[int, ...]
    action_horizons: tuple[int, ...]
    flow_steps: tuple[int, ...]
    resident_models: tuple[int, ...]
    inflight_jobs: tuple[int, ...]
    load_pcts: tuple[int, ...]
    invalid_counts: dict[MatrixCategory, int]

    def entries(self) -> Iterator[MatrixEntry]:
        """Yield every matrix cell (valid + invalid injections)."""
        yield from _valid_entries(self)
        yield from _invalid_entries(self)

    def valid_entries(self) -> Iterator[MatrixEntry]:
        """Yield only the valid cells."""
        yield from _valid_entries(self)

    def __len__(self) -> int:
        valid = (
            len(self.engines)
            * len(self.m_boundaries)
            * len(self.frequencies_mhz)
            * len(self.memory_tiers)
            * len(self.onchip_states)
            * len(self.image_counts)
            * len(self.action_horizons)
            * len(self.flow_steps)
            * len(self.resident_models)
            * len(self.inflight_jobs)
            * len(self.load_pcts)
        )
        return valid + sum(self.invalid_counts.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": {
                "engines": list(self.engines),
                "m_boundaries": list(self.m_boundaries),
                "frequencies_mhz": list(self.frequencies_mhz),
                "memory_tiers": list(self.memory_tiers),
                "onchip_states": list(self.onchip_states),
                "image_counts": list(self.image_counts),
                "action_horizons": list(self.action_horizons),
                "flow_steps": list(self.flow_steps),
                "resident_models": list(self.resident_models),
                "inflight_jobs": list(self.inflight_jobs),
                "load_pcts": list(self.load_pcts),
            },
            "invalid_counts": {k.value: v for k, v in self.invalid_counts.items()},
            "total_cells": len(self),
        }


# Default axis values from the plan.
DEFAULT_ENGINES = canonical_engine_ids()
DEFAULT_M_BOUNDARIES = (1, 2, 3, 15, 16, 17, 63, 64, 65, 128, 1024)
DEFAULT_FREQUENCIES_MHZ = (800, 1000, 1200)
DEFAULT_MEMORY_TIERS = ("LPDDR5", "LPDDR5X", "HBM2e", "HBM3")
DEFAULT_ONCHIP_STATES = ("none", "full", "partial", "spill")
DEFAULT_IMAGE_COUNTS = (1, 2, 3, 4)
DEFAULT_ACTION_HORIZONS = (8, 10, 25, 50)
DEFAULT_FLOW_STEPS = (4, 8, 10)
DEFAULT_RESIDENT_MODELS = (4, 8)
DEFAULT_INFLIGHT_JOBS = (4, 8, 16)
DEFAULT_LOAD_PCTS = (50, 90, 95, 110)


def build_matrix(
    engines: Iterable[str] | None = None,
    m_boundaries: Iterable[int] | None = None,
    frequencies_mhz: Iterable[int] | None = None,
    memory_tiers: Iterable[str] | None = None,
    onchip_states: Iterable[str] | None = None,
    image_counts: Iterable[int] | None = None,
    action_horizons: Iterable[int] | None = None,
    flow_steps: Iterable[int] | None = None,
    resident_models: Iterable[int] | None = None,
    inflight_jobs: Iterable[int] | None = None,
    load_pcts: Iterable[int] | None = None,
    invalid_counts: dict[MatrixCategory, int] | None = None,
) -> AcceptanceMatrix:
    """Build the full lazy acceptance matrix."""
    return AcceptanceMatrix(
        engines=tuple(engines or DEFAULT_ENGINES),
        m_boundaries=tuple(m_boundaries or DEFAULT_M_BOUNDARIES),
        frequencies_mhz=tuple(frequencies_mhz or DEFAULT_FREQUENCIES_MHZ),
        memory_tiers=tuple(memory_tiers or DEFAULT_MEMORY_TIERS),
        onchip_states=tuple(onchip_states or DEFAULT_ONCHIP_STATES),
        image_counts=tuple(image_counts or DEFAULT_IMAGE_COUNTS),
        action_horizons=tuple(action_horizons or DEFAULT_ACTION_HORIZONS),
        flow_steps=tuple(flow_steps or DEFAULT_FLOW_STEPS),
        resident_models=tuple(resident_models or DEFAULT_RESIDENT_MODELS),
        inflight_jobs=tuple(inflight_jobs or DEFAULT_INFLIGHT_JOBS),
        load_pcts=tuple(load_pcts or DEFAULT_LOAD_PCTS),
        invalid_counts=dict(invalid_counts or _default_invalid_counts()),
    )


def _default_invalid_counts() -> dict[MatrixCategory, int]:
    return {
        MatrixCategory.INVALID_SCHEMA: 1,
        MatrixCategory.INVALID_OP: 1,
        MatrixCategory.INVALID_HASH: 1,
        MatrixCategory.INVALID_CALIBRATION: 1,
    }


def _valid_entries(matrix: AcceptanceMatrix) -> Iterator[MatrixEntry]:
    for engine in matrix.engines:
        for m in matrix.m_boundaries:
            for freq in matrix.frequencies_mhz:
                for tier in matrix.memory_tiers:
                    for onchip in matrix.onchip_states:
                        for image in matrix.image_counts:
                            for action in matrix.action_horizons:
                                for flow in matrix.flow_steps:
                                    for resident in matrix.resident_models:
                                        for inflight in matrix.inflight_jobs:
                                            for load in matrix.load_pcts:
                                                yield MatrixEntry(
                                                    engine=engine,
                                                    m=m,
                                                    frequency_mhz=freq,
                                                    memory_tier=tier,
                                                    onchip_state=onchip,
                                                    image_count=image,
                                                    action_horizon=action,
                                                    flow_steps=flow,
                                                    resident_models=resident,
                                                    inflight_jobs=inflight,
                                                    load_pct=load,
                                                    category=MatrixCategory.VALID,
                                                )


def _invalid_entries(matrix: AcceptanceMatrix) -> Iterator[MatrixEntry]:
    base = MatrixEntry(
        engine=matrix.engines[0] if matrix.engines else "block",
        m=matrix.m_boundaries[0] if matrix.m_boundaries else 1,
        frequency_mhz=matrix.frequencies_mhz[0] if matrix.frequencies_mhz else 1000,
        memory_tier=matrix.memory_tiers[0] if matrix.memory_tiers else "LPDDR5",
        onchip_state="none",
        image_count=matrix.image_counts[0] if matrix.image_counts else 1,
        action_horizon=matrix.action_horizons[0] if matrix.action_horizons else 8,
        flow_steps=matrix.flow_steps[0] if matrix.flow_steps else 4,
        resident_models=matrix.resident_models[0] if matrix.resident_models else 4,
        inflight_jobs=matrix.inflight_jobs[0] if matrix.inflight_jobs else 4,
        load_pct=matrix.load_pcts[0] if matrix.load_pcts else 50,
    )
    categories = [
        MatrixCategory.INVALID_SCHEMA,
        MatrixCategory.INVALID_OP,
        MatrixCategory.INVALID_HASH,
        MatrixCategory.INVALID_CALIBRATION,
    ]
    for category in categories:
        for idx in range(matrix.invalid_counts.get(category, 0)):
            yield MatrixEntry(
                engine=base.engine,
                m=base.m + idx,
                frequency_mhz=base.frequency_mhz,
                memory_tier=base.memory_tier,
                onchip_state=base.onchip_state,
                image_count=base.image_count,
                action_horizon=base.action_horizon,
                flow_steps=base.flow_steps,
                resident_models=base.resident_models,
                inflight_jobs=base.inflight_jobs,
                load_pct=base.load_pct,
                category=category,
            )


def load_axes_defaults(yaml_path: Path | None = None) -> dict[str, Any]:
    """Load the DSE axes defaults so the matrix can map to real design points."""
    path = yaml_path or (SIM_DIR / "config" / "dse_axes.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("defaults", {})


__all__ = [
    "AcceptanceMatrix",
    "MatrixCategory",
    "MatrixEntry",
    "OnChipState",
    "build_matrix",
    "load_axes_defaults",
]
