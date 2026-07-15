"""Validated evidence manifests for DSE engine candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


MATURITY_RANK = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4}
EVIDENCE_AXES = (
    "performance",
    "ppa",
    "functional_scope",
    "system_integration",
)
REQUIRED_UNCERTAINTY = ("performance_pct", "area_pct", "power_pct")


@dataclass(frozen=True)
class EngineManifest:
    name: str
    display_name: str
    module: str
    class_name: str
    role: str
    maturity: str
    evidence: Dict[str, Dict[str, Any]]
    scope: Dict[str, Any]
    fallbacks: Dict[str, str]
    uncertainty: Dict[str, float]
    calibration_dataset: str | None
    known_gaps: tuple[str, ...]
    sources: tuple[str, ...]
    component_evidence: Dict[str, Dict[str, Any]]

    @property
    def maturity_rank(self) -> int:
        return MATURITY_RANK[self.maturity]

    @property
    def raw_exploration_eligible(self) -> bool:
        return self.maturity_rank >= MATURITY_RANK["M1"]

    @property
    def comparison_ready(self) -> bool:
        return self.maturity_rank >= MATURITY_RANK["M2"]

    @property
    def product_qualified(self) -> bool:
        return self.maturity_rank >= MATURITY_RANK["M3"]

    @property
    def calibration_tier(self) -> str:
        return str(self.evidence["performance"]["kind"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _validate_evidence(name: str, evidence: Mapping[str, Any]) -> None:
    for axis in EVIDENCE_AXES:
        item = _require_mapping(evidence.get(axis), f"{name}.evidence.{axis}")
        level = item.get("level")
        kind = item.get("kind")
        if not isinstance(level, int) or not 0 <= level <= 4:
            raise ValueError(f"{name}.evidence.{axis}.level must be 0..4")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"{name}.evidence.{axis}.kind must be non-empty")


def _validate_maturity(name: str, maturity: str, evidence: Mapping[str, Any]) -> None:
    if maturity not in MATURITY_RANK:
        raise ValueError(f"{name}.maturity must be one of {tuple(MATURITY_RANK)}")
    levels = {axis: int(evidence[axis]["level"]) for axis in EVIDENCE_AXES}
    if MATURITY_RANK[maturity] >= 1 and min(levels.values()) < 1:
        raise ValueError(f"{name}: M1 requires all evidence axes >= 1")
    if MATURITY_RANK[maturity] >= 2:
        if levels["performance"] < 2:
            raise ValueError(f"{name}: M2 requires performance evidence >= 2")
        if levels["functional_scope"] < 2 or levels["system_integration"] < 2:
            raise ValueError(f"{name}: M2 requires full workload/system evidence >= 2")
    if MATURITY_RANK[maturity] >= 3:
        if levels["performance"] < 3 or levels["ppa"] < 2:
            raise ValueError(f"{name}: M3 requires calibrated performance and PPA evidence")


def _validate_uncertainty(name: str, uncertainty: Mapping[str, Any]) -> None:
    for field in REQUIRED_UNCERTAINTY:
        value = uncertainty.get(field)
        if not isinstance(value, (int, float)) or not 0 < float(value) <= 100:
            raise ValueError(f"{name}.uncertainty.{field} must be in (0, 100]")


def _manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "engine_manifests.yaml"


@lru_cache(maxsize=4)
def load_engine_manifests(path: str | Path | None = None) -> Dict[str, EngineManifest]:
    manifest_path = Path(path) if path is not None else _manifest_path()
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    root = _require_mapping(payload, "manifest root")
    if root.get("schema_version") != 1:
        raise ValueError("engine manifest schema_version must be 1")
    raw_engines = _require_mapping(root.get("engines"), "engines")
    manifests: Dict[str, EngineManifest] = {}
    for name, raw in raw_engines.items():
        if not isinstance(name, str) or not name:
            raise ValueError("engine name must be a non-empty string")
        item = _require_mapping(raw, name)
        evidence = dict(_require_mapping(item.get("evidence"), f"{name}.evidence"))
        uncertainty = dict(_require_mapping(item.get("uncertainty"), f"{name}.uncertainty"))
        _validate_evidence(name, evidence)
        maturity = str(item.get("maturity", ""))
        _validate_maturity(name, maturity, evidence)
        _validate_uncertainty(name, uncertainty)
        required_strings = ("display_name", "module", "class_name", "role")
        for field in required_strings:
            if not isinstance(item.get(field), str) or not str(item[field]).strip():
                raise ValueError(f"{name}.{field} must be non-empty")
        scope = dict(_require_mapping(item.get("scope"), f"{name}.scope"))
        if not scope.get("dataflow") or not scope.get("workload"):
            raise ValueError(f"{name}.scope must define dataflow and workload")
        gaps = item.get("known_gaps")
        sources = item.get("sources")
        if not isinstance(gaps, list) or not gaps:
            raise ValueError(f"{name}.known_gaps must be a non-empty list")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{name}.sources must be a non-empty list")
        manifests[name] = EngineManifest(
            name=name,
            display_name=str(item["display_name"]),
            module=str(item["module"]),
            class_name=str(item["class_name"]),
            role=str(item["role"]),
            maturity=maturity,
            evidence=evidence,
            scope=scope,
            fallbacks=dict(_require_mapping(item.get("fallbacks", {}), f"{name}.fallbacks")),
            uncertainty={key: float(value) for key, value in uncertainty.items()},
            calibration_dataset=item.get("calibration_dataset"),
            known_gaps=tuple(str(value) for value in gaps),
            sources=tuple(str(value) for value in sources),
            component_evidence=dict(item.get("component_evidence", {})),
        )
    if not manifests:
        raise ValueError("engine manifest must contain at least one engine")
    return manifests


def get_engine_manifest(name: str) -> EngineManifest:
    try:
        return load_engine_manifests()[name]
    except KeyError as exc:
        raise ValueError(f"engine {name!r} has no validated manifest") from exc


def engine_names(minimum_maturity: str = "M1") -> tuple[str, ...]:
    if minimum_maturity not in MATURITY_RANK:
        raise ValueError(f"unknown maturity: {minimum_maturity}")
    threshold = MATURITY_RANK[minimum_maturity]
    return tuple(
        name for name, manifest in load_engine_manifests().items()
        if manifest.maturity_rank >= threshold
    )


def validate_manifest_set(expected_engines: Iterable[str]) -> None:
    expected = set(expected_engines)
    actual = set(load_engine_manifests())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"engine manifest mismatch: missing={missing}, extra={extra}")
