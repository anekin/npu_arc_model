import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "sim"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SIM)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_dse_cli_exposes_scenario_and_requirements():
    result = _run("sim/design_space_explorer.py", "--help")
    assert result.returncode == 0, result.stderr
    assert "--scenario" in result.stdout
    assert "--requirements" in result.stdout


def test_legacy_arc_help_does_not_require_gguf_adapter():
    result = _run("sim/arc_model.py", "--help")
    assert result.returncode == 0, result.stderr
    assert "evaluate" in result.stdout


def test_dse_core_has_no_product_runtime_imports():
    files = [SIM / "design_space_explorer.py", *sorted((SIM / "dse").glob("*.py"))]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    for forbidden in ("golden_executor", "q4_dequant", "npu_sim", "sim.regmap"):
        assert forbidden not in source
