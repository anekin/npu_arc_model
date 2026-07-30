"""Scenario-driven DSE runner.

``ScenarioDseRunner`` takes a ``Scenario``, a ``DesignSpace``, and evaluates each
``DesignPoint`` through ``ScenarioRunner``.  Results are assembled into schema-v2
``DesignSpaceResultV2`` with stable IDs, full config, and temporal metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from calibration.evaluate import (
    TrustGate,
    calibration_digest,
    calibration_ids_for_design_point,
)
from calibration.registry import CalibrationRegistry
from contracts.errors import ConfigError
from contracts.identity import digest_sha256
from contracts.result import (
    CalibrationRef,
    DesignPointResult,
    DesignSpaceResultV2,
    EngineMetrics,
    ErrorRecord,
    ResultSummary,
    RunStatus,
    RunTrustLevel,
    release_recommendation,
)
from dse.manifest import CoverageManifest
from dse.models import DesignPoint
from dse.pareto import MultiObjectivePareto, ParetoPoint
from dse.space import DesignSpace, GenerationResult
from engine.ppa_model import AreaModel, _node_scale_factor
from scenario_runner import run_scenario
from scenarios.compiler import compile_scenario
from scenarios.schema import (
    ArrivalMode,
    ArrivalPattern,
    QueuePolicy,
    Scenario,
    WorkloadClass,
)
from scheduler.metrics import ScenarioMetrics
from workloads.catalog import WorkloadFixture, load_all_fixtures

SIM_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DseRunConfig:
    """Configuration controlling a scenario-driven DSE run."""

    scenario: Scenario
    design_space: DesignSpace
    seed: int = 0
    allow_partial: bool = False
    thermal_limit_w: float = 150.0
    quality_gate_required: bool = False
    # How many releases per class are measured.  Smaller = faster CI.
    measurement_count: int = 50
    warmup_count: int = 5
    trust_mode: str = "exploratory"


@dataclass
class EvaluatedPoint:
    """A design point together with its evaluation outcome."""

    point: DesignPoint
    result: DesignPointResult | None = None
    metrics: ScenarioMetrics | None = None
    ppa: Any | None = None
    error: ErrorRecord | None = None
    calibration_violations: list[dict[str, Any]] = field(default_factory=list)


class ScenarioDseRunner:
    """Run scenario-driven design-space exploration."""

    def __init__(self, run_config: DseRunConfig) -> None:
        self.run_config = run_config
        self.scenario = run_config.scenario
        self.design_space = run_config.design_space
        self.pareto = MultiObjectivePareto.from_scenario(run_config.scenario)
        if run_config.thermal_limit_w is not None:
            self.pareto.thermal_limit_w = run_config.thermal_limit_w
        self.pareto.quality_gate_required = run_config.quality_gate_required
        self._fixtures: dict[str, WorkloadFixture] | None = None
        self._registry = CalibrationRegistry.from_yaml()
        self._trust_gate = TrustGate(self._registry)

    def _fixtures_map(self) -> dict[str, WorkloadFixture]:
        if self._fixtures is None:
            self._fixtures = load_all_fixtures()
        return self._fixtures

    def _fixture_for_scenario(self) -> WorkloadFixture | None:
        ref = self.scenario.workload_ref
        if ref is None:
            return None
        return self._fixtures_map().get(ref)

    @staticmethod
    def _scenario_for_point(
        base_scenario: Scenario,
        point: DesignPoint,
        work_ms: float,
    ) -> Scenario:
        """Build a point-specific scenario with workload dimensions applied."""
        av = point.axis_values
        queue_policy = QueuePolicy(av.get("queue_policy", "fifo"))
        quantum_ms = float(av.get("nonpreemptible_quantum_ms", 0.0))
        partition = av.get("partition", "none")
        resident_models = int(av.get("resident_models", 1))
        inflight_jobs = int(av.get("inflight_jobs", 4))

        period_ms = max(1.0, work_ms * 2.0)
        deadline_ms = max(1.0, work_ms * 3.0)

        classes: list[WorkloadClass] = []
        for idx, base_cls in enumerate(base_scenario.classes):
            cls_work_ms = work_ms
            # Scale service time by relevant workload dimensions.
            action_horizon = int(av.get("action_horizon", 1))
            flow_steps = int(av.get("flow_steps", 1))
            token_block = int(av.get("token_block", 1))
            active_sequences = int(av.get("active_sequences", 1))
            image_count = int(av.get("image_count", 1))

            # VLA/Physical-AI: one job processes action_horizon * flow_steps actions.
            if (
                base_scenario.workload_ref
                and "vla" not in base_scenario.workload_ref
                and "physical" not in base_scenario.workload_ref
            ):
                cls_work_ms = work_ms * max(token_block / max(active_sequences, 1), 1.0)
            else:
                cls_work_ms = work_ms * action_horizon * flow_steps * max(image_count, 1)

            cls_work_ms = max(0.1, cls_work_ms)
            cls_period = max(cls_work_ms * 1.5, period_ms)
            cls_deadline = max(cls_work_ms * 2.0, deadline_ms)

            classes.append(
                WorkloadClass(
                    id=base_cls.id or f"class_{idx}",
                    arrival=ArrivalPattern(
                        mode=ArrivalMode.PERIODIC,
                        period_ms=cls_period,
                        count=max(base_scenario.warmup_count + base_scenario.measurement_count + 10, 20),
                        offset_ms=0.0,
                    ),
                    work_ms=cls_work_ms,
                    relative_deadline_ms=cls_deadline,
                    timeout_ms=cls_deadline * 2.0,
                    priority=base_cls.priority,
                    queue_policy=queue_policy,
                    queue_capacity=base_cls.queue_capacity,
                    stream_id=base_cls.stream_id,
                    resource_requirements=dict(base_cls.resource_requirements),
                    memory_bytes=base_cls.memory_bytes,
                    bandwidth_fraction=base_cls.bandwidth_fraction,
                    admission_excluded=base_cls.admission_excluded,
                    metadata={
                        **base_cls.metadata,
                        "resident_models": resident_models,
                        "inflight_jobs": inflight_jobs,
                        "partition": partition,
                        "nonpreemptible_quantum_ms": quantum_ms,
                    },
                )
            )

        return Scenario(
            name=base_scenario.name,
            description=base_scenario.description,
            seed=base_scenario.seed,
            warmup_count=base_scenario.warmup_count,
            measurement_count=base_scenario.measurement_count,
            drain=base_scenario.drain,
            max_simulation_time_ms=base_scenario.max_simulation_time_ms,
            recovery_phase_start_ms=base_scenario.recovery_phase_start_ms,
            workload_ref=base_scenario.workload_ref,
            classes=classes,
            compute_capacity=max(base_scenario.compute_capacity, 1),
            resources=dict(base_scenario.resources),
            memory_available_bytes=max(base_scenario.memory_available_bytes, 1),
            max_inflight_jobs=max(inflight_jobs, base_scenario.max_inflight_jobs),
            max_bandwidth_fraction=base_scenario.max_bandwidth_fraction,
            preemption_enabled=base_scenario.preemption_enabled,
            metadata={**base_scenario.metadata, "point_axis_values": dict(av)},
        )

    def _evaluate_ppa(self, point: DesignPoint) -> tuple[Any, str | None]:
        """Evaluate PPA for a design point using legacy AreaModel/PowerModel."""
        try:
            import design_space_explorer as dse_module
            from engine.ppa_model import AreaModel, PowerModel

            base_path = SIM_DIR / "config" / "design_space.yaml"
            base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))

            # Merge the design point's process_node so cross-node DSE uses the
            # correct area model parameters instead of always defaulting to 7nm.
            point_node = point.hardware_config.get("area_model", {}).get("process_node")
            if point_node is not None:
                base_cfg.setdefault("area_model", {})["process_node"] = point_node

            area_model = AreaModel(base_cfg)
            power_model = PowerModel(base_cfg)

            # Ensure legacy globals point at a consistent proxy trace so PPA is
            # deterministic and independent of whatever earlier CLI set them to.
            dse_module._CV_MODEL = ""
            dse_module._LLM_TRACE = dse_module.generate_trace_from_spec("qwen2.5-3b", batch_m=1)
            dse_module._NUM_LAYERS = 28

            ppa = dse_module.evaluate_config(point.hardware_config, area_model, power_model)
            return ppa, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    def _work_ms_from_ppa(self, ppa: Any, point: DesignPoint) -> float:
        """Derive scenario service time from PPA throughput."""
        tok_s = getattr(ppa, "tok_s", 0.0) or 1.0
        return max(0.01, 1000.0 / tok_s)

    def evaluate_point(self, point: DesignPoint) -> EvaluatedPoint:
        """Evaluate a single design point."""
        ppa, ppa_error = self._evaluate_ppa(point)
        if ppa is None:
            error = ErrorRecord(
                design_point_id=point.design_point_id,
                code="PPAError",
                message=ppa_error or "unknown PPA error",
            )
            result = DesignPointResult(
                design_point_id=point.design_point_id,
                status=RunStatus.failed,
                scenario_ref=point.scenario_ref,
                workload_ref=point.workload_ref or "",
                engine_type=point.hardware_config.get("mac_engine", {}).get("type", "unknown"),
                trust_level=RunTrustLevel.non_authoritative,
                error=error,
            )
            return EvaluatedPoint(point=point, result=result, error=error)

        work_ms = self._work_ms_from_ppa(ppa, point)
        point_scenario = self._scenario_for_point(self.scenario, point, work_ms)

        try:
            fixture = self._fixture_for_scenario()
            compiled = compile_scenario(point_scenario, fixture=fixture)
            metrics = run_scenario(compiled)
        except Exception as exc:  # noqa: BLE001
            error = ErrorRecord(
                design_point_id=point.design_point_id,
                code="ScenarioError",
                message=str(exc),
            )
            result = DesignPointResult(
                design_point_id=point.design_point_id,
                status=RunStatus.failed,
                scenario_ref=point.scenario_ref,
                workload_ref=point.workload_ref or "",
                engine_type=point.hardware_config.get("mac_engine", {}).get("type", "unknown"),
                trust_level=RunTrustLevel.non_authoritative,
                error=error,
            )
            return EvaluatedPoint(point=point, result=result, ppa=ppa, error=error)

        # Build EngineMetrics combining PPA and scenario temporal metrics.
        compute_util = 0.0
        for ru in metrics.resource_utilization:
            if ru.resource_name == "compute":
                compute_util = ru.utilization
                break

        # Power proxy: if scenario energy is available and window time is positive.
        energy = (
            metrics.energy_joules
            if metrics.energy_joules > 0
            else ppa.power_w * (metrics.window_time_ps / 1_000_000_000_000.0)
        )

        engine_metrics = EngineMetrics(
            tok_per_s=ppa.tok_s,
            area_mm2=ppa.area_mm2,
            power_w=ppa.power_w,
            efficiency_tok_per_watt=ppa.efficiency_tok_per_watt,
            efficiency_tok_per_mm2=ppa.efficiency_tok_per_mm2,
            completed_throughput_hz=metrics.completed_throughput_hz,
            mac_count=None,
            op_count=None,
            total_cycles=None,
            utilization=compute_util,
            avg_latency_s=metrics.latency_p50_ms / 1000.0 if metrics.latency_p50_ms is not None else None,
            p50_latency_s=metrics.latency_p50_ms / 1000.0 if metrics.latency_p50_ms is not None else None,
            p99_latency_s=metrics.latency_p99_ms / 1000.0 if metrics.latency_p99_ms is not None else None,
            max_latency_s=metrics.latency_max_ms / 1000.0 if metrics.latency_max_ms is not None else None,
            deadline_miss_count=metrics.deadline_miss_count,
            drop_count=metrics.dropped_count + metrics.replaced_count,
            memory_footprint_gib=None,
            spill_bytes=None,
            energy_joules=energy,
        )

        calibration_ids = calibration_ids_for_design_point(point.hardware_config)
        gate_ok, gate_trust, violations = self._trust_gate.check(
            calibration_ids,
            hw_config=point.hardware_config,
        )

        if self.run_config.trust_mode == "decision_grade" and (not gate_ok or gate_trust.value in {"T0", "T1"}):
            point_trust = RunTrustLevel.non_authoritative
        elif gate_trust.value == "T0" or gate_trust.value == "T1":
            point_trust = RunTrustLevel.exploratory
        elif gate_trust.value == "T2":
            point_trust = RunTrustLevel.calibrated_estimate
        else:
            point_trust = RunTrustLevel.authoritative

        area_model = AreaModel(point.hardware_config)
        node_scale = _node_scale_factor(area_model.process_node_nm)

        result = DesignPointResult(
            design_point_id=point.design_point_id,
            status=RunStatus.complete,
            hardware_digest=digest_sha256(point.hardware_config),
            scenario_ref=point.scenario_ref,
            workload_ref=point.workload_ref or "",
            calibration=CalibrationRef(
                process_node_nm=area_model.process_node_nm,
                node_scale=node_scale,
                dram_efficiency=point.hardware_config.get("memory", {}).get("dram_efficiency", 0.85),
                pe_area_ratio_block_systolic=2.0,
                trust_level=point_trust,
            ),
            config_label=ppa.config_label,
            engine_type=point.hardware_config.get("mac_engine", {}).get("type", "unknown"),
            trust_level=point_trust,
            metrics=engine_metrics,
        )
        ep = EvaluatedPoint(point=point, result=result, metrics=metrics, ppa=ppa)
        ep.calibration_violations = violations
        return ep

    def run(
        self,
        generation_result: GenerationResult | None = None,
    ) -> tuple[DesignSpaceResultV2, CoverageManifest, list[ParetoPoint]]:
        """Evaluate all design points and return v2 result, manifest, and Pareto frontier."""
        if generation_result is None:
            generation_result = self.design_space.generate_with_exclusions()

        points = list(generation_result.points)
        manifest = CoverageManifest(self.design_space.axes, points, generation_result.exclusions)

        evaluated: list[EvaluatedPoint] = []
        v2_results: list[DesignPointResult] = []
        errors: list[ErrorRecord] = []

        for point in points:
            manifest.record_evaluated(point)
            ep = self.evaluate_point(point)
            evaluated.append(ep)
            if ep.result is not None:
                v2_results.append(ep.result)
                if ep.error is not None:
                    errors.append(ep.error)
                    manifest.record_failed(point, ep.error.message or "error")
                else:
                    manifest.record_success(point)
            else:
                # Should not happen, but defensively create an error result.
                err = ErrorRecord(
                    design_point_id=point.design_point_id,
                    code="UnknownError",
                    message="evaluation returned no result",
                )
                errors.append(err)
                v2_results.append(
                    DesignPointResult(
                        design_point_id=point.design_point_id,
                        status=RunStatus.failed,
                        scenario_ref=point.scenario_ref,
                        workload_ref=point.workload_ref or "",
                        trust_level=RunTrustLevel.non_authoritative,
                        error=err,
                    )
                )
                manifest.record_failed(point, err.message)

        has_errors = any(ep.error is not None for ep in evaluated)
        is_partial = has_errors and self.run_config.allow_partial

        # Adjust trust levels for partial runs.
        set_trust = RunTrustLevel.non_authoritative if is_partial else RunTrustLevel.exploratory
        if is_partial:
            for r in v2_results:
                r.trust_level = RunTrustLevel.non_authoritative

        summary = ResultSummary(
            generated=len(points),
            evaluated=sum(1 for ep in evaluated if ep.error is None or ep.result is not None),
            pruned=0,
            failed=sum(1 for ep in evaluated if ep.error is not None),
            filtered=0,
            complete=sum(
                1
                for ep in evaluated
                if ep.error is None and ep.result is not None and ep.result.status == RunStatus.complete
            ),
            partial=sum(1 for ep in evaluated if ep.result is not None and ep.result.status == RunStatus.partial),
        )

        # Input / workload / calibration digests for reproducibility.
        input_source = {
            "scenario": self.scenario.model_dump(mode="json"),
            "axes_config_path": str(
                self.design_space.axes_config.get("base_config_source", "config/design_space.yaml")
            ),
            "seed": self.run_config.seed,
        }
        input_digest = digest_sha256(input_source)

        workload_digest = ""
        fixture = self._fixture_for_scenario()
        if fixture is not None:
            workload_digest = fixture.footprint_digest

        result_cal_digest = calibration_digest(self._registry)

        result_set = DesignSpaceResultV2(
            trust_level=set_trust,
            summary=summary,
            results=v2_results,
            errors=errors,
            input_digest=input_digest,
            workload_digest=workload_digest,
            calibration_digest=result_cal_digest,
        )

        frontier = self.pareto.compute_frontier(v2_results)
        result_set.frontier_design_point_ids = [p.result.design_point_id for p in frontier]

        if self.run_config.trust_mode == "decision_grade":
            violating_ids = sorted({v["calibration_id"] for ep in evaluated for v in ep.calibration_violations})
            if violating_ids:
                raise ConfigError(
                    f"decision-grade trust gate failed for calibration IDs: {', '.join(violating_ids)}",
                    field_path="trust_mode",
                    value=violating_ids,
                )

        return result_set, manifest, frontier

    def recommendation(self, result_set: DesignSpaceResultV2) -> list[DesignPointResult]:
        """Return release recommendation or raise NonAuthoritativeRunError."""
        return release_recommendation(result_set)


def run_scenario_dse(
    scenario: Scenario,
    design_space: DesignSpace,
    *,
    seed: int = 0,
    allow_partial: bool = False,
    thermal_limit_w: float = 150.0,
    trust_mode: str = "exploratory",
) -> tuple[DesignSpaceResultV2, CoverageManifest, list[ParetoPoint]]:
    """Convenience wrapper to run scenario-driven DSE."""
    config = DseRunConfig(
        scenario=scenario,
        design_space=design_space,
        seed=seed,
        allow_partial=allow_partial,
        thermal_limit_w=thermal_limit_w,
        trust_mode=trust_mode,
    )
    return ScenarioDseRunner(config).run()


__all__ = [
    "DseRunConfig",
    "EvaluatedPoint",
    "ScenarioDseRunner",
    "run_scenario_dse",
]
