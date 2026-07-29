"""Compile a ``Scenario`` (+ optional workload fixture) into a runnable form.

The compiler turns arrival patterns into a deterministic job release list,
instantiates scheduler resources, builds queues, and configures admission
control.  It does not run the simulation; that is the responsibility of
``scenario_runner.ScenarioRunner``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from contracts.errors import ConfigError
from models.memory_hierarchy import build_hierarchy_from_config
from models.residency import build_memory_access_plan
from scheduler.admission import AdmissionController
from scheduler.metrics import ms_to_ps
from scheduler.queues import BoundedFIFO, MailboxLatest
from scheduler.resources import CapacityResource
from scenarios.schema import ArrivalMode, QueuePolicy, Scenario, WorkloadClass
from workloads.catalog import WorkloadFixture, load_all_fixtures


@dataclass(frozen=True)
class JobRelease:
    """A single compiled job release event."""

    job_id: str
    class_id: str
    arrival_ps: int
    work_ps: int
    deadline_ps: int
    timeout_ps: int
    is_warmup: bool
    is_measurement: bool
    sequence: int


@dataclass
class CompiledScenario:
    """Runnable output of ``compile_scenario``."""

    scenario: Scenario
    fixture: Optional[WorkloadFixture]
    releases: List[JobRelease] = field(default_factory=list)
    resources: Dict[str, CapacityResource] = field(default_factory=dict)
    queues: Dict[str, Any] = field(default_factory=dict)
    admission: AdmissionController = field(default_factory=lambda: AdmissionController(0, 1))
    window_start_ps: int = 0
    window_end_ps: int = 0
    memory_plan: Optional[Any] = None


def _load_scenario_from_yaml(yaml_path: str | Path) -> Scenario:
    """Load a ``Scenario`` from a YAML file."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"scenario {yaml_path.name} must be a mapping", field_path="")
    # Support both top-level scenario and nested "scenario" key.
    data = raw.get("scenario", raw)
    return Scenario.model_validate(data)


def _build_resources(scenario: Scenario) -> Dict[str, CapacityResource]:
    """Create capacity resources from the scenario configuration."""
    resources: Dict[str, CapacityResource] = {
        "compute": CapacityResource(name="compute", capacity=scenario.compute_capacity),
    }
    for name, capacity in scenario.resources.items():
        if capacity <= 0:
            raise ConfigError(
                f"resource {name!r} capacity must be positive, got {capacity}",
                field_path=f"resources.{name}",
            )
        resources[name] = CapacityResource(name=name, capacity=capacity)
    return resources


def _build_queues(scenario: Scenario) -> Dict[str, Any]:
    """Create queue objects for FIFO and mailbox_latest classes."""
    queues: Dict[str, Any] = {}
    for cls in scenario.classes:
        if cls.queue_policy == QueuePolicy.FIFO:
            queues[f"fifo:{cls.id}"] = BoundedFIFO(capacity=cls.queue_capacity, name=f"fifo:{cls.id}")
        elif cls.queue_policy == QueuePolicy.MAILBOX_LATEST:
            stream = cls.stream_id or cls.id
            key = f"mailbox:{stream}"
            if key not in queues:
                queues[key] = MailboxLatest(name=key)
    return queues


def _build_admission(scenario: Scenario, fixture: Optional[WorkloadFixture]) -> AdmissionController:
    """Build admission controller using scenario + optional fixture footprint."""
    memory_available = scenario.memory_available_bytes
    if fixture is not None:
        # Build a minimal memory hierarchy from the fixture's scenario config if
        # present; otherwise use the scenario's memory_available_bytes.
        hierarchy = build_hierarchy_from_config(fixture.scenario.get("memory_config", {}))
        if not hierarchy.tiers:
            hierarchy = build_hierarchy_from_config({"memory": {"capacity_gb": scenario.memory_available_bytes / 1e9}})
        try:
            plan = build_memory_access_plan(
                fixture.graph,
                hierarchy,
                resident_models=fixture.bindings.resident_models or 1,
                allow_spill=True,
            )
            memory_available = max(memory_available, plan.hierarchy.total_usable_bytes())
        except Exception:  # noqa: BLE001
            pass
    return AdmissionController(
        memory_available_bytes=memory_available,
        max_inflight_jobs=scenario.max_inflight_jobs,
        max_bandwidth_fraction=scenario.max_bandwidth_fraction,
    )


def compile_scenario(
    scenario: Scenario,
    *,
    fixture: Optional[WorkloadFixture] = None,
) -> CompiledScenario:
    """Compile a scenario into a runnable ``CompiledScenario``.

    Args:
        scenario: The scenario to compile.
        fixture: Optional workload fixture for memory/resource planning.

    Returns:
        A ``CompiledScenario`` with release list, resources, queues, admission,
        and measurement window bounds.
    """
    releases: List[JobRelease] = []
    seq = 0
    class_window_ends: List[int] = []

    for cls in scenario.classes:
        release_times = cls.arrival.release_times_ps()
        total = len(release_times)
        warmup = min(scenario.warmup_count, total)
        measure = min(scenario.measurement_count, total - warmup)

        for idx, arrival_ps in enumerate(release_times):
            is_warmup = idx < warmup
            is_measurement = warmup <= idx < warmup + measure
            rel_deadline_ps = cls.relative_deadline_ps
            deadline_ps = arrival_ps + rel_deadline_ps
            timeout_ps = arrival_ps + cls.timeout_ps
            job_id = f"{cls.id}-{idx:06d}"
            releases.append(
                JobRelease(
                    job_id=job_id,
                    class_id=cls.id,
                    arrival_ps=arrival_ps,
                    work_ps=cls.work_ps,
                    deadline_ps=deadline_ps,
                    timeout_ps=timeout_ps,
                    is_warmup=is_warmup,
                    is_measurement=is_measurement,
                    sequence=seq,
                )
            )
            seq += 1
            if is_measurement:
                class_window_ends.append(arrival_ps)

    releases.sort(key=lambda r: (r.arrival_ps, r.sequence))

    # Measurement window: from first non-warmup arrival to the expected end of
    # the measurement interval (last measurement arrival + one period) so that
    # utilization denominators match the offered load over a full period grid.
    measurement_releases = [r for r in releases if r.is_measurement]
    window_start_ps = min((r.arrival_ps for r in measurement_releases), default=0)
    window_end_ps = max((r.arrival_ps for r in measurement_releases), default=window_start_ps)
    for cls in scenario.classes:
        if cls.arrival.mode != ArrivalMode.PERIODIC:
            continue
        cls_measurements = [r for r in measurement_releases if r.class_id == cls.id]
        if not cls_measurements:
            continue
        first_arrival = cls_measurements[0].arrival_ps
        expected_end = first_arrival + len(cls_measurements) * cls.arrival.period_ps
        if expected_end > window_end_ps:
            window_end_ps = expected_end

    compiled = CompiledScenario(
        scenario=scenario,
        fixture=fixture,
        releases=releases,
        resources=_build_resources(scenario),
        queues=_build_queues(scenario),
        admission=_build_admission(scenario, fixture),
        window_start_ps=window_start_ps,
        window_end_ps=window_end_ps,
    )
    return compiled


def compile_scenario_from_yaml(
    yaml_path: str | Path,
    *,
    fixtures: Optional[Dict[str, WorkloadFixture]] = None,
) -> CompiledScenario:
    """Load and compile a scenario from a YAML file.

    Args:
        yaml_path: Path to the scenario YAML file.
        fixtures: Optional mapping of workload fixtures.  If omitted, fixtures
            are discovered from the default catalog directory.
    """
    scenario = _load_scenario_from_yaml(yaml_path)
    fixture: Optional[WorkloadFixture] = None
    if scenario.workload_ref is not None:
        fixtures = fixtures or load_all_fixtures()
        fixture = fixtures.get(scenario.workload_ref)
        if fixture is None:
            raise ConfigError(
                f"scenario references unknown workload fixture {scenario.workload_ref!r}",
                field_path="workload_ref",
            )
    return compile_scenario(scenario, fixture=fixture)


__all__ = [
    "JobRelease",
    "CompiledScenario",
    "compile_scenario",
    "compile_scenario_from_yaml",
]
