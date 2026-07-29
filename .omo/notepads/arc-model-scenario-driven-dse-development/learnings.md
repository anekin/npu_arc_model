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

## Todo 3: Physical Invariant Red Matrix (2026-07-29)

### What was done
- Created `sim/tests/oracles/physics.py` — independent closed-form oracle computing MAC/byte/cycle conservation bounds from first principles (AST-verified: zero engine imports)
- Created `sim/tests/test_engine_physical_invariants.py` — 760 parametrized tests across 8 engines × M={1..1024} × (64,64)/(110,72) shapes × LPDDR5/HBM3/HIGH_BW bandwidth tiers
- Created `sim/tests/test_engine_invalid_inputs.py` — 424 tests covering all `estimate` and `estimate_weight_cache_pair` paths with 0/negative/float/bool/string shapes, invalid array/precision/bandwidth
- Created `sim/tests/__init__.py` and `sim/tests/oracles/__init__.py` for proper package imports
- Evidence: `.omo/evidence/task-3-physical-collect.txt` (exit 0, 1184 tests collected), `.omo/evidence/task-3-physical-red.json` (exit 1, 45 failed/1139 passed)

### Key findings — Red manifest failures mapped to Todos 4/5/6

| Failure Category | Affected Engines | Mapped Todo | Root Cause |
|:---|:---|:---|:---|
| **Systolic M=2→3 latency decrease** | systolic | Todo 5 | Decode (M≤2) vs prefill (M>2) branch discontinuity creates latency drop at M=3 boundary |
| **OS M scaling** | os_systolic | Todo 5 | Per-tile compute does not scale with M; M=1 and M=1024 return same compute cycles |
| **GMMA pipeline undercuts MAC floor** | gmma | Todo 5 | `pipeline_scale=0.05` with small tiles produces total_cycles below `ceil(mac_count / peak)` |
| **InputStationary M monotonicity** | input_stationary | Todo 5 | Reuse factor `min(M,H)/H` creates non-monotonic transition around M=H boundary |
| **FSA mac_count inflated** | fsa | Todo 4/5 | Stores `M×K×N×ops_per_mac` in legacy `ops` field instead of MAC count M×K×N |
| **FSA weight_bytes=0** | fsa | Todo 5 | SRAM caching logic zeroes weight_bytes when weights fit in buffer; invariant requires >0 |
| **WMMA missing diagnostics** | wmma | Todo 4 | Cache-pair path omits `per_fragment_dma`; direct path details incomplete |
| **Bandwidth saturation (all engines)** | all 8 | Todo 6 | None of the engines correctly saturate at compute floor as BW increases; GB/s↔bytes/cycle conversion needs Todo 6 unit repair |
| **GMMA/OS DMA floor** | gmma, os_systolic | Todo 5/6 | DMA cycles computed below raw byte-transfer floor when bandwidth is high; ceil usage inconsistent |

### Technical decisions
- Oracle uses `bandwidth_gbps × 1000 / frequency_mhz` for bytes/cycle conversion (plan formula)
- `raw_transfer_bytes = weight_bytes + activation_bytes` with NO caching efficiency — represents absolute physical floor
- Diagnostics cache-pair test relaxed to subset check (cache-pair paths have different key sets)
- Invalid-input tests use non-crash criterion (engines may not raise for all invalid inputs, but must not crash)
- `sim/tests/__init__.py` added to enable proper package imports for oracle module
- FSA engine uses `ops_per_mac=2` and stores op_count not mac_count — need Todo 4 contract to enforce semantics
