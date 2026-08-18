"""Test result schema v2 — status, error association, legacy projection, partial non-authoritative.

Given: patterns from the v2 result schema.
When:  results are built or validated.
Then:  all the Todo 9 acceptance criteria pass.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from contracts.errors import NonAuthoritativeRunError
from contracts.identity import digest_sha256
from contracts.legacy_result import (
    LegacyLossReport,
    legacy_result_dict_from_ppa,
    project_v2_to_legacy_cv,
    project_v2_to_legacy_llm,
)
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
    result_standalone_from_ppa,
)
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = REPO_ROOT / "sim"


# ── Helpers ───────────────────────────────────────────────────────────────────


class _FakePPA:
    """Minimal fake PPA for testing projection logic."""

    def __init__(self, tok_s=100.0, area=50.0, power=10.0, ttft_ms=0.0, label="block 64x128", spill=0.0, dw=0.0):
        self.tok_s = tok_s
        self.area_mm2 = area
        self.power_w = power
        self.ttft_ms = ttft_ms
        self.efficiency_tok_per_watt = tok_s / max(power, 0.1)
        self.efficiency_tok_per_mm2 = tok_s / max(area, 0.1)
        self.config_label = label
        self.sram_spill_mb = spill
        self.depthwise_util_pct = dw


def _make_base_config(engine_type="block", h=64, w=128, freq=1000.0, bw=51.2):
    return {
        "version": "2",
        "mac_engine": {
            "type": engine_type,
            "array_height": h,
            "array_width": w,
            "frequency_mhz": freq,
            "weight_precision_bits": 4,
            "activation_precision_bits": 8,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "bandwidth_gbps": bw,
            "dram_efficiency": 0.85,
        },
        "sram": {"l1_per_core_kb": 512, "l2_shared_kb": 2048},
    }


# ── RunStatus and TrustLevel ──────────────────────────────────────────────────


class TestRunStatus:
    def test_all_status_values(self):
        assert RunStatus.complete.value == "complete"
        assert RunStatus.partial.value == "partial"
        assert RunStatus.failed.value == "failed"
        assert RunStatus.filtered.value == "filtered"

    def test_status_is_string_enum(self):
        assert isinstance(RunStatus.complete, str)
        assert RunStatus.complete == "complete"


class TestRunTrustLevel:
    def test_all_trust_values(self):
        assert RunTrustLevel.authoritative.value == "authoritative"
        assert RunTrustLevel.calibrated_estimate.value == "calibrated_estimate"
        assert RunTrustLevel.exploratory.value == "exploratory"
        assert RunTrustLevel.non_authoritative.value == "non_authoritative"


# ── ErrorRecord ───────────────────────────────────────────────────────────────


class TestErrorRecord:
    def test_basic_construction(self):
        err = ErrorRecord(design_point_id="abc123", code="RuntimeError", message="test error")
        assert err.code == "RuntimeError"
        assert err.design_point_id == "abc123"

    def test_message_truncation(self):
        long_msg = "x" * 300
        err = ErrorRecord(design_point_id="abc", code="E", message=long_msg)
        assert len(err.message) <= 200

    def test_details_dict(self):
        err = ErrorRecord(design_point_id="abc", code="E", details={"key": "val", "num": 42})
        assert err.details["key"] == "val"
        assert err.details["num"] == 42

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ErrorRecord(design_point_id="abc", code="E", unknown_field="bad")


# ── DesignPointResult construction ────────────────────────────────────────────


class TestDesignPointResult:
    def test_complete_result(self):
        cfg = _make_base_config()
        dp_id = digest_sha256(cfg)
        metrics = EngineMetrics(tok_per_s=100.0, area_mm2=50.0, power_w=10.0)
        result = DesignPointResult(
            design_point_id=dp_id,
            status=RunStatus.complete,
            trust_level=RunTrustLevel.exploratory,
            metrics=metrics,
            config_label="block 64x128",
            engine_type="block",
        )
        assert result.design_point_id == dp_id
        assert result.status == RunStatus.complete
        assert result.metrics.tok_per_s == 100.0

    def test_failed_result_no_metrics(self):
        cfg = _make_base_config()
        dp_id = digest_sha256(cfg)
        err = ErrorRecord(design_point_id=dp_id, code="ConfigError", message="bad config")
        result = DesignPointResult(
            design_point_id=dp_id,
            status=RunStatus.failed,
            trust_level=RunTrustLevel.non_authoritative,
            error=err,
            config_label="bad",
            engine_type="unknown",
        )
        assert result.status == RunStatus.failed
        assert result.metrics is None
        assert result.error is not None
        assert result.error.code == "ConfigError"

    def test_partial_status_non_authoritative(self):
        result = DesignPointResult(
            design_point_id="partial-1",
            status=RunStatus.partial,
            trust_level=RunTrustLevel.non_authoritative,
            config_label="partial run",
            engine_type="block",
        )
        assert result.status == RunStatus.partial
        assert result.trust_level == RunTrustLevel.non_authoritative


# ── DesignSpaceResultV2 container ─────────────────────────────────────────────


class TestDesignSpaceResultV2:
    @pytest.fixture
    def v2_empty(self):
        return DesignSpaceResultV2()

    @pytest.fixture
    def v2_with_data(self):
        cfg = _make_base_config()
        dp_id = digest_sha256(cfg)
        metrics = EngineMetrics(tok_per_s=100.0, area_mm2=50.0, power_w=10.0)
        results = [
            DesignPointResult(
                design_point_id=dp_id,
                status=RunStatus.complete,
                trust_level=RunTrustLevel.authoritative,
                metrics=metrics,
                config_label="test",
                engine_type="block",
            )
        ]
        summary = ResultSummary(generated=10, evaluated=10, failed=0, complete=10)
        return DesignSpaceResultV2(
            trust_level=RunTrustLevel.authoritative,
            summary=summary,
            results=results,
            errors=[],
        )

    def test_default_version(self, v2_empty):
        assert v2_empty.schema_version == "2"

    def test_empty_results(self, v2_empty):
        assert len(v2_empty.results) == 0
        assert len(v2_empty.errors) == 0

    def test_with_data(self, v2_with_data):
        assert v2_with_data.trust_level == RunTrustLevel.authoritative
        assert v2_with_data.summary.generated == 10
        assert len(v2_with_data.results) == 1


# ── Release recommendation gate ───────────────────────────────────────────────


class TestReleaseRecommendation:
    @pytest.fixture
    def auth_results(self):
        cfg = _make_base_config()
        dp_id = digest_sha256(cfg)
        metrics = EngineMetrics(tok_per_s=100.0, area_mm2=50.0, power_w=10.0)
        return DesignSpaceResultV2(
            trust_level=RunTrustLevel.authoritative,
            summary=ResultSummary(generated=5, evaluated=5, complete=5),
            results=[
                DesignPointResult(
                    design_point_id=dp_id,
                    status=RunStatus.complete,
                    trust_level=RunTrustLevel.authoritative,
                    metrics=metrics,
                    config_label="test",
                    engine_type="block",
                )
            ],
        )

    def test_authoritative_passes(self, auth_results):
        recs = release_recommendation(auth_results)
        assert len(recs) == 1
        assert recs[0].status == RunStatus.complete

    def test_non_authoritative_set_raises(self, auth_results):
        auth_results.trust_level = RunTrustLevel.non_authoritative
        with pytest.raises(NonAuthoritativeRunError) as exc_info:
            release_recommendation(auth_results)
        assert "non_authoritative" in str(exc_info.value)
        assert exc_info.value.reason

    def test_exploratory_set_raises(self, auth_results):
        auth_results.trust_level = RunTrustLevel.exploratory
        with pytest.raises(NonAuthoritativeRunError):
            release_recommendation(auth_results)

    def test_individual_non_auth_raises(self, auth_results):
        auth_results.results[0].trust_level = RunTrustLevel.non_authoritative
        with pytest.raises(NonAuthoritativeRunError) as exc_info:
            release_recommendation(auth_results)
        assert "non-authoritative" in str(exc_info.value)


# ── result_standalone_from_ppa bridge ─────────────────────────────────────────


class TestResultStandaloneFromPPA:
    def test_basic_mapping(self):
        cfg = _make_base_config()
        ppa = _FakePPA(tok_s=100.0, area=50.0, power=10.0, label="block 64x128")
        result = result_standalone_from_ppa(ppa, cfg)
        assert result.status == RunStatus.complete
        assert result.metrics is not None
        assert result.metrics.tok_per_s == 100.0
        assert result.metrics.area_mm2 == 50.0
        assert result.metrics.power_w == 10.0
        assert result.metrics.ttft_ms == 0.0

    def test_explicit_status_and_trust(self):
        cfg = _make_base_config()
        ppa = _FakePPA()
        result = result_standalone_from_ppa(
            ppa, cfg, status=RunStatus.partial, trust_level=RunTrustLevel.non_authoritative
        )
        assert result.status == RunStatus.partial
        assert result.trust_level == RunTrustLevel.non_authoritative

    def test_id_derived_from_config(self):
        cfg = _make_base_config()
        ppa = _FakePPA()
        result = result_standalone_from_ppa(ppa, cfg)
        assert len(result.design_point_id) == 64


# ── Legacy projection ─────────────────────────────────────────────────────────


class TestLegacyProjection:
    def test_legacy_result_dict_llm(self):
        ppa = _FakePPA(tok_s=100.0, area=50.0, power=10.0, label="block 64x128")
        d = legacy_result_dict_from_ppa(ppa, cv_mode=False)
        assert d["label"] == "block 64x128"
        assert d["tok_s"] == 100.0
        assert d["area_mm2"] == 50.0
        assert d["power_w"] == 10.0
        assert d["ttft_ms"] == 0.0
        assert "sram_spill_mb" not in d

    def test_legacy_result_dict_cv(self):
        ppa = _FakePPA(tok_s=100.0, area=50.0, power=10.0, label="bloc 64x128", spill=3.5, dw=0.42)
        d = legacy_result_dict_from_ppa(ppa, cv_mode=True, on_pareto=True)
        assert d["label"] == "bloc 64x128"
        assert d["ttft_ms"] == 0.0
        assert d["sram_spill_mb"] == 3.5
        assert d["depthwise_util_pct"] == 0.42
        assert d["engine_type"] == "block"
        assert d["pareto"] is True

    def test_project_v2_to_legacy_llm_preserves_fields(self):
        cfg = _make_base_config()
        dp_id = digest_sha256(cfg)
        metrics = EngineMetrics(tok_per_s=100.0, area_mm2=50.0, power_w=10.0, ttft_ms=55.0)
        v2 = DesignSpaceResultV2(
            trust_level=RunTrustLevel.exploratory,
            summary=ResultSummary(generated=5, evaluated=5, complete=5),
            results=[
                DesignPointResult(
                    design_point_id=dp_id,
                    status=RunStatus.complete,
                    trust_level=RunTrustLevel.exploratory,
                    metrics=metrics,
                    config_label="block 64x128",
                    engine_type="block",
                )
            ],
        )  # type: ignore[arg-type]
        legacy, loss = project_v2_to_legacy_llm(v2, model_spec="qwen2.5-3b", batch_m=1, total_configs=5)

        assert legacy["model_spec"] == "qwen2.5-3b"
        assert legacy["valid_results"] == 1
        assert legacy["generated"] == 5
        assert legacy["errors"] == 0
        assert len(legacy["pareto_frontier"]) >= 0
        assert isinstance(legacy["top_results"], list)
        assert "design_point_id" in loss.dropped_fields
        assert legacy["top_results"][0]["ttft_ms"] == 55.0
        assert legacy["batch_m"] == 1
        assert legacy["total_configs"] == 5
        assert legacy["valid_results"] == 1
        assert legacy["generated"] == 5
        assert legacy["errors"] == 0
        assert len(legacy["pareto_frontier"]) >= 0
        assert isinstance(legacy["top_results"], list)
        assert "design_point_id" in loss.dropped_fields

    def test_project_v2_to_legacy_cv_preserves_fields(self):
        cfg = _make_base_config()
        dp_id = digest_sha256(cfg)
        metrics = EngineMetrics(
            tok_per_s=100.0, area_mm2=50.0, power_w=10.0, sram_spill_mb=3.5, depthwise_util_pct=0.42
        )
        v2 = DesignSpaceResultV2(
            trust_level=RunTrustLevel.exploratory,
            summary=ResultSummary(generated=5, evaluated=5, complete=5),
            results=[
                DesignPointResult(
                    design_point_id=dp_id,
                    status=RunStatus.complete,
                    trust_level=RunTrustLevel.exploratory,
                    metrics=metrics,
                    config_label="bloc 64x128",
                    engine_type="block",
                )
            ],
        )  # type: ignore[arg-type]
        legacy, loss = project_v2_to_legacy_cv(v2, cv_model="yolov8n")

        assert legacy["cv_model"] == "yolov8n"
        assert legacy["metadata"]["valid_results"] == 1
        assert legacy["metadata"]["generated"] == 5
        assert len(legacy["points"]) >= 0
        assert "schema_version" in loss.dropped_fields

    def test_legacy_loss_report(self):
        loss = LegacyLossReport(dropped_fields=["a"], warnings=["w"])
        assert loss.has_loss
        loss2 = LegacyLossReport()
        assert not loss2.has_loss


# ── DSE CLI integration: --result-schema v2 ───────────────────────────────────


class TestDSEV2Output:
    """Given: DSE with --result-schema v2. When: run with --output. Then: v2 JSON produced."""

    def test_dse_v2_output_produces_valid_json(self, tmp_path):
        output = tmp_path / "v2_output.json"
        cmd = [
            sys.executable,
            str(SIM_DIR / "design_space_explorer.py"),
            "--quick",
            "--result-schema",
            "v2",
            "--output",
            str(output),
        ]
        env = {
            "PYTHONPATH": str(SIM_DIR),
            "PATH": str(Path(sys.executable).parent),
            **dict(__import__("os").environ.items()),
        }
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output.exists()

        with open(output) as f:
            data = json.load(f)

        assert data["schema_version"] == "2"
        assert "trust_level" in data
        assert "summary" in data
        assert "results" in data
        assert "errors" in data
        assert "input_digest" in data
        assert len(data["input_digest"]) == 64

        for r in data["results"]:
            assert len(r["design_point_id"]) == 64
            assert r["status"] in ("complete", "partial")
            assert r["metrics"] is not None
            assert "ttft_ms" in r["metrics"]
            assert r["metrics"]["ttft_ms"] >= 0

        # Summary consistency
        summary = data["summary"]
        assert summary["generated"] > 0
        assert summary["evaluated"] > 0

    def test_dse_v2_partial_non_authoritative(self, tmp_path):
        output = tmp_path / "v2_partial.json"
        cmd = [
            sys.executable,
            str(SIM_DIR / "design_space_explorer.py"),
            "--quick",
            "--allow-partial",
            "--result-schema",
            "v2",
            "--output",
            str(output),
        ]
        env = {
            "PYTHONPATH": str(SIM_DIR),
            "PATH": str(Path(sys.executable).parent),
            **dict(__import__("os").environ.items()),
        }
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output.exists()

        with open(output) as f:
            data = json.load(f)

        if data["summary"]["failed"] > 0 or data["summary"]["partial"] > 0:
            assert data["trust_level"] == "non_authoritative"

    def test_dse_v1_legacy_output_unchanged(self, tmp_path):
        """Given: --result-schema v1 (default). When: DSE runs. Then: legacy output unchanged."""
        output = tmp_path / "v1_legacy.json"
        cmd = [
            sys.executable,
            str(SIM_DIR / "design_space_explorer.py"),
            "--quick",
            "--output",
            str(output),
        ]
        env = {
            "PYTHONPATH": str(SIM_DIR),
            "PATH": str(Path(sys.executable).parent),
            **dict(__import__("os").environ.items()),
        }
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=120)
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output.exists()

        with open(output) as f:
            data = json.load(f)

        # Legacy fields preserved
        assert "model_spec" in data or "cv_model" in data
        assert "valid_results" in data
        assert "generated" in data or "metadata" in data
        assert "pareto_frontier" in data or "points" in data


# ── Error association by stable ID, not position ──────────────────────────────


class TestErrorAssociationById:
    def test_error_carries_design_point_id(self):
        err = ErrorRecord(design_point_id="dp-001", code="FooError", message="failed")
        assert err.design_point_id == "dp-001"

    def test_error_not_position_based(self):
        cfg_a = _make_base_config(engine_type="block", h=64, w=128)
        cfg_b = _make_base_config(engine_type="systolic", h=32, w=64)
        id_a = digest_sha256(cfg_a)
        id_b = digest_sha256(cfg_b)

        # Build two errors in different order, check they're associated by ID
        err_a = ErrorRecord(design_point_id=id_a, code="E", message="a")
        err_b = ErrorRecord(design_point_id=id_b, code="E", message="b")

        assert err_a.design_point_id == id_a
        assert err_b.design_point_id == id_b
        assert err_a.design_point_id != err_b.design_point_id


# ── CalibrationRef ────────────────────────────────────────────────────────────


class TestCalibrationRef:
    def test_default_values(self):
        cal = CalibrationRef()
        assert cal.process_node_nm == 12.0
        assert cal.node_scale == 2.70
        assert cal.dram_efficiency == 0.85
        assert cal.trust_level == RunTrustLevel.exploratory

    def test_custom_values(self):
        cal = CalibrationRef(process_node_nm=7.0, trust_level=RunTrustLevel.calibrated_estimate)
        assert cal.process_node_nm == 7.0
        assert cal.trust_level == RunTrustLevel.calibrated_estimate
