# Learnings — arc-model-scenario-driven-dse-development

## Todo 1: Baseline Freeze (2026-07-29)

### What was done
- Created `pyproject.toml` with CPython >=3.10,<3.13; dependencies: numpy, pyyaml, pydantic v2, onnx, pytest, ruff, basedpyright
- Generated `uv.lock` (22 packages resolved) with digest `ce33a246`
- Created `sim/tests/golden/legacy_cli_contract.json` — frozen golden contract capturing legacy CLI shape
- Created `sim/tests/test_legacy_compatibility.py` — 18 tests validating legacy CLI commands, flags, exit codes, JSON structure, and determinism
- Created `sim/tests/test_environment_repro.py` — 13 tests for env reproducibility (7 positive, 6 negative-path)
- Updated `README.md` Section 8 with scope disclaimers and baseline provenance table
- Updated `pytest.ini` with custom marker registration

### Key findings
- The `--freq` flag does NOT propagate in the current legacy code — output at 800/1000/1200 MHz is identical
- `uv sync --frozen` fails due to slow PyPI downloads on this server, but `uv lock` completes successfully and the lock file is valid
- The project's PYTHONPATH=sim pattern works consistently with both system Python and uv venv
- All 94 tests pass (63 original + 31 new), confirming backwards compatibility

### Technical decisions
- Used `hatchling` as build backend with `packages = ["sim"]`
- Golden contract does NOT freeze numeric frequency values (per plan)
- Historical baseline records `node_scale=2.94x` (to be corrected to 2.70x in Todo 11)
- Pytest markers in both `pytest.ini` and `pyproject.toml` for resilience
