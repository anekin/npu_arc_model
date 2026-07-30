"""Tests for legacy DSE import surface and default CLI behavior."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from dse.legacy_adapter import evaluate_config, find_pareto, generate_configs
from engine.ppa_model import AreaModel, PowerModel


@pytest.fixture
def legacy_models():
    base_path = Path(__file__).resolve().parents[1] / "config" / "design_space.yaml"
    base_cfg = json.loads(json.dumps(__import__("yaml").safe_load(base_path.read_text(encoding="utf-8"))))
    return AreaModel(base_cfg), PowerModel(base_cfg)


def test_legacy_adapter_exports_generate_configs(legacy_models):
    area_model, power_model = legacy_models
    configs = list(generate_configs(quick=True))
    assert len(configs) > 0
    assert all(isinstance(c, dict) for c in configs)
    assert all("mac_engine" in c for c in configs)


def test_legacy_adapter_exports_evaluate_config(legacy_models):
    area_model, power_model = legacy_models
    configs = list(generate_configs(quick=True))
    ppa = evaluate_config(configs[0], area_model, power_model)
    assert hasattr(ppa, "tok_s")
    assert hasattr(ppa, "area_mm2")
    assert hasattr(ppa, "power_w")


def test_legacy_adapter_exports_find_pareto(legacy_models):
    area_model, power_model = legacy_models
    configs = list(generate_configs(quick=True))
    ppas = [evaluate_config(c, area_model, power_model) for c in configs]
    frontier = find_pareto(ppas)
    assert isinstance(frontier, list)
    assert len(frontier) <= len(ppas)


def test_legacy_quick_cli_runs_with_v1_schema_by_default(tmp_path: Path):
    output = tmp_path / "legacy.json"
    result = subprocess.run(
        [sys.executable, "sim/design_space_explorer.py", "--quick", "--output", str(output)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "schema_version" not in data or data.get("schema_version") in ("1", 1)
    assert "pareto_frontier" in data
    assert "top_results" in data


def test_scenario_cli_produces_v2_schema(tmp_path: Path):
    output = tmp_path / "v2bundle"
    result = subprocess.run(
        [
            sys.executable,
            "sim/design_space_explorer.py",
            "--scenario",
            "embodied_compact_vla",
            "--space",
            "ci-all-axes",
            "--seed",
            "1",
            "--result-schema",
            "v2",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "result.json").exists()
    data = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "2"
    assert "frontier_design_point_ids" in data


def test_legacy_and_scenario_flags_are_mutually_exclusive(tmp_path: Path):
    output = tmp_path / "conflict.json"
    result = subprocess.run(
        [
            sys.executable,
            "sim/design_space_explorer.py",
            "--quick",
            "--scenario",
            "embodied_compact_vla",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "mutually exclusive" in (result.stdout + result.stderr).lower()
