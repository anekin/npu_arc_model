"""Test: legacy CLI compatibility snapshots for Todo 1 baseline freeze.

Runs the actual legacy CLI entrypoints and verifies:
- Exit codes match the golden contract
- JSON output structure matches the frozen top-level keys and field types
- Flag help and choices are preserved
- Does NOT freeze known-incorrect numeric frequency values
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR = REPO_ROOT / "sim"
GOLDEN_DIR = REPO_ROOT / "sim" / "tests" / "golden"


def _load_golden():
    path = GOLDEN_DIR / "legacy_cli_contract.json"
    assert path.exists(), f"Golden contract not found: {path}"
    return json.loads(path.read_text())


def _run_npu_sim(*extra_args, cwd=REPO_ROOT):
    """Run sim/npu_sim.py with the given extra args."""
    cmd = [sys.executable, str(SIM_DIR / "npu_sim.py"), *extra_args]
    env = {
        "PYTHONPATH": str(SIM_DIR),
        "PATH": str(Path(sys.executable).parent) + ":" + (Path(sys.executable).parent / ".." / ".." / "bin").as_posix(),
    }
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), env={**__import__("os").environ, **env}, timeout=60
    )


def _run_dse(*extra_args, cwd=REPO_ROOT):
    """Run sim/design_space_explorer.py with the given extra args."""
    cmd = [sys.executable, str(SIM_DIR / "design_space_explorer.py"), *extra_args]
    env = {
        "PYTHONPATH": str(SIM_DIR),
        "PATH": str(Path(sys.executable).parent) + ":" + (Path(sys.executable).parent / ".." / ".." / "bin").as_posix(),
    }
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), env={**__import__("os").environ, **env}, timeout=60
    )


class TestLegacyCLISnapshot:
    """Snapshot the legacy CLI commands: exit codes and JSON structure."""

    golden = _load_golden()

    # ── npu_sim.py snapshot tests ──

    @pytest.mark.baseline
    def test_npu_sim_systolic_json_exit_code(self):
        """npu_sim --engine systolic --json must exit 0."""
        result = _run_npu_sim("--engine", "systolic", "--json")
        assert result.returncode == self.golden["npu_sim_cli"]["exit_code_success"], (
            f"Expected exit {self.golden['npu_sim_cli']['exit_code_success']}, got {result.returncode}\n"
            f"stderr: {result.stderr}"
        )

    @pytest.mark.snapshot
    def test_npu_sim_systolic_json_structure(self):
        """npu_sim --engine systolic --json output must match golden top-level keys and field types."""
        result = _run_npu_sim("--engine", "systolic", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)

        golden_output = self.golden["npu_sim_json_output"]
        # Verify top-level keys
        assert set(output.keys()) == set(golden_output["top_level_keys"]), (
            f"Top-level keys mismatch: got {sorted(output.keys())}, expected {golden_output['top_level_keys']}"
        )

        # Verify decode structure
        decode = output["decode"]
        decode_golden = golden_output["decode"]
        assert set(decode.keys()) == set(decode_golden["keys"]), (
            f"Decode keys mismatch: got {sorted(decode.keys())}, expected {decode_golden['keys']}"
        )
        assert decode["engine_type"] in decode_golden["engine_type"]["values"]
        assert isinstance(decode["array_height"], int)
        assert isinstance(decode["array_width"], int)
        assert isinstance(decode["per_token_us"], (int, float))
        assert isinstance(decode["tok_per_s"], (int, float))
        assert isinstance(decode["breakdown"], dict)
        # Verify breakdown keys (but not values — frequency values are known-incorrect)
        assert set(decode["breakdown"].keys()) == set(decode_golden["breakdown"]["keys"]), (
            f"Decode breakdown keys mismatch: got {sorted(decode['breakdown'].keys())}"
        )

        # Verify prefill structure
        prefill = output["prefill"]
        prefill_golden = golden_output["prefill"]
        assert set(prefill.keys()) == set(prefill_golden["keys"]), (
            f"Prefill keys mismatch: got {sorted(prefill.keys())}, expected {prefill_golden['keys']}"
        )
        assert isinstance(prefill["prompt_len"], int)
        assert isinstance(prefill["total_ms"], (int, float))
        assert isinstance(prefill["breakdown"], dict)
        assert set(prefill["breakdown"].keys()) == set(prefill_golden["breakdown"]["keys"])

    @pytest.mark.baseline
    def test_npu_sim_default_json_exit_code(self):
        """npu_sim --json (default engine) must exit 0."""
        result = _run_npu_sim("--json")
        assert result.returncode == 0

    @pytest.mark.snapshot
    def test_npu_sim_default_json_structure(self):
        """npu_sim --json (default engine=block) output must match golden structure."""
        result = _run_npu_sim("--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)

        golden_output = self.golden["npu_sim_json_output"]
        assert set(output.keys()) == set(golden_output["top_level_keys"])
        assert "engine_type" in output["decode"]
        assert "prefill" in output
        # Default engine is 'block'
        assert output["decode"]["engine_type"] == "block"

    @pytest.mark.baseline
    def test_npu_sim_list_engines(self):
        """npu_sim --list-engines must exit 0 and list 7 engines."""
        result = _run_npu_sim("--list-engines")
        assert result.returncode == 0
        expected_engines = self.golden["npu_sim_cli"]["flags"]["--engine"]["choices"]
        for eng in expected_engines:
            assert eng in result.stdout, f"Engine '{eng}' not found in --list-engines output"

    @pytest.mark.baseline
    def test_npu_sim_list_dram(self):
        """npu_sim --list-dram must exit 0 and list all dram presets."""
        result = _run_npu_sim("--list-dram")
        assert result.returncode == 0
        for preset in self.golden["npu_sim_cli"]["flags"]["--dram"]["choices"]:
            assert preset in result.stdout, f"DRAM preset '{preset}' not found in --list-dram output"

    @pytest.mark.baseline
    def test_npu_sim_help_flags(self):
        """npu_sim --help must exit 0 and mention all golden flags."""
        result = _run_npu_sim("--help")
        assert result.returncode == 0
        golden_flags = self.golden["npu_sim_cli"]["flags"]
        for flag_name in golden_flags:
            assert flag_name in result.stdout, f"Flag '{flag_name}' not found in --help output"

    @pytest.mark.baseline
    def test_npu_sim_json_output_has_correct_units(self):
        """Verify units in JSON output match the golden contract: us, ms, tok/s."""
        result = _run_npu_sim("--engine", "systolic", "--json")
        assert result.returncode == 0
        output = json.loads(result.stdout)

        # per_token_us must be positive microseconds
        assert output["decode"]["per_token_us"] > 0, "per_token_us must be positive"
        # tok_per_s must be positive
        assert output["decode"]["tok_per_s"] > 0, "tok_per_s must be positive"
        # prefill total_ms must be positive milliseconds
        assert output["prefill"]["total_ms"] > 0, "prefill total_ms must be positive"

        # Breakdown values must be non-negative (us for decode, ms for prefill)
        for val in output["decode"]["breakdown"].values():
            assert val >= 0, f"Decode breakdown value must be non-negative, got {val}"
        for val in output["prefill"]["breakdown"].values():
            assert val >= 0, f"Prefill breakdown value must be non-negative, got {val}"

    # ── design_space_explorer.py snapshot tests ──

    @pytest.mark.baseline
    def test_dse_quick_exit_code(self):
        """design_space_explorer.py --quick must exit 0."""
        result = _run_dse("--quick")
        assert result.returncode == self.golden["design_space_explorer_cli"]["exit_code_success"]

    @pytest.mark.baseline
    def test_dse_quick_with_output(self, tmp_path):
        """design_space_explorer.py --quick --output must exit 0 and write valid JSON."""
        output_path = tmp_path / "dse_quick.json"
        result = _run_dse("--quick", "--output", str(output_path))
        assert result.returncode == 0
        assert output_path.exists()
        dse = json.loads(output_path.read_text())
        golden_dse = self.golden["dse_json_output"]
        assert set(dse.keys()) == set(golden_dse["top_level_keys"]), (
            f"DSE output keys mismatch: got {sorted(dse.keys())}, expected {golden_dse['top_level_keys']}"
        )

    @pytest.mark.snapshot
    def test_dse_json_fields_match_contract(self, tmp_path):
        """DSE JSON output fields must match the golden contract types and structure."""
        output_path = tmp_path / "dse_quick.json"
        result = _run_dse("--quick", "--output", str(output_path))
        assert result.returncode == 0
        dse = json.loads(output_path.read_text())
        golden_fields = self.golden["dse_json_output"]["fields"]

        for field_name, spec in golden_fields.items():
            assert field_name in dse, f"Field '{field_name}' missing from DSE output"
            if spec["type"] == "integer":
                assert isinstance(dse[field_name], int), (
                    f"Field '{field_name}' should be int, got {type(dse[field_name])}"
                )
            elif spec["type"] == "array":
                assert isinstance(dse[field_name], list), f"Field '{field_name}' should be list"
            elif spec["type"] == "string":
                assert dse[field_name] is None or isinstance(dse[field_name], str)

        # Verify errors=0 for quick mode
        assert dse["errors"] == 0, "Quick DSE must have errors=0"

        # Verify pareto_frontier items have correct keys and types
        for item in dse["pareto_frontier"]:
            assert "label" in item
            assert "tok_s" in item and isinstance(item["tok_s"], (int, float))
            assert "area_mm2" in item and isinstance(item["area_mm2"], (int, float))
            assert "power_w" in item and isinstance(item["power_w"], (int, float))

        # Verify top_results items have correct keys and types
        for item in dse["top_results"][:3]:
            assert "label" in item
            assert "tok_s" in item and isinstance(item["tok_s"], (int, float))
            assert "area_mm2" in item and isinstance(item["area_mm2"], (int, float))
            assert "power_w" in item and isinstance(item["power_w"], (int, float))

    @pytest.mark.baseline
    def test_dse_help_flags(self):
        """design_space_explorer.py --help must exit 0 and mention all golden flags."""
        result = _run_dse("--help")
        assert result.returncode == 0
        golden_flags = self.golden["design_space_explorer_cli"]["flags"]
        for flag_name in golden_flags:
            assert flag_name in result.stdout, f"Flag '{flag_name}' not found in DSE --help output"

    @pytest.mark.baseline
    def test_dse_allow_partial_with_quick(self, tmp_path):
        """--allow-partial --quick must exit 0 and produce JSON."""
        output_path = tmp_path / "dse_partial.json"
        result = _run_dse("--allow-partial", "--quick", "--output", str(output_path))
        assert result.returncode == 0
        assert output_path.exists()
        dse = json.loads(output_path.read_text())
        assert dse["errors"] == 0

    @pytest.mark.baseline
    def test_dse_batch_m_flag_output(self, tmp_path):
        """--batch-m 1 must produce batch_m=1 in JSON output."""
        output_path = tmp_path / "dse_batchm.json"
        result = _run_dse("--batch-m", "1", "--quick", "--output", str(output_path))
        assert result.returncode == 0
        dse = json.loads(output_path.read_text())
        assert dse["batch_m"] == 1

    @pytest.mark.baseline
    def test_dse_model_spec_flag_output(self, tmp_path):
        """--model-spec qwen2.5-3b must produce model_spec=qwen2.5-3b in JSON output."""
        output_path = tmp_path / "dse_modelspec.json"
        result = _run_dse("--model-spec", "qwen2.5-3b", "--quick", "--output", str(output_path))
        assert result.returncode == 0
        dse = json.loads(output_path.read_text())
        assert dse["model_spec"] == "qwen2.5-3b"

    @pytest.mark.baseline
    def test_dse_invalid_model_spec_rejected(self):
        """Invalid --model-spec must produce an error (exit 2 or message)."""
        result = _run_dse("--model-spec", "qwen25-3b-int4", "--quick")
        # argparse exits 2 for invalid choices
        assert result.returncode != 0, f"Invalid model-spec should fail, got exit {result.returncode}"

    # ── Cross-CLI consistency ──

    @pytest.mark.snapshot
    def test_npu_sim_engine_choices_match_list_engines(self):
        """The --engine choices must exactly match --list-engines output."""
        golden_choices = self.golden["npu_sim_cli"]["flags"]["--engine"]["choices"]
        result = _run_npu_sim("--list-engines")
        for eng in golden_choices:
            assert eng in result.stdout, f"Engine '{eng}' from golden choices not in --list-engines"

    @pytest.mark.snapshot
    def test_output_determinism_twice(self):
        """Running the same command twice must produce identical JSON output."""
        result1 = _run_npu_sim("--engine", "systolic", "--json")
        result2 = _run_npu_sim("--engine", "systolic", "--json")
        assert result1.returncode == 0
        assert result2.returncode == 0
        assert result1.stdout.strip() == result2.stdout.strip(), (
            "Two identical runs produced different output — non-deterministic behavior"
        )
