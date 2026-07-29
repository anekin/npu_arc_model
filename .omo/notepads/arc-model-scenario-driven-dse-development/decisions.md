# Decisions — arc-model-scenario-driven-dse-development

## Todo 1: Baseline Freeze (2026-07-29)

### Decision 1: Package structure
**What**: Project uses `hatchling` + `packages = ["sim"]` for wheel build, with `sim/` as the root package.
**Why**: Existing codebase uses `sim/` as the import root (PYTHONPATH=sim). Hatchling needs explicit package config.
**Alternatives**: Could have used `setuptools` with `find:` but hatchling is simpler and modern.

### Decision 2: Pytest config dual-location
**What**: Markers registered in both `pytest.ini` and `pyproject.toml [tool.pytest.ini_options]`.
**Why**: `pytest.ini` takes precedence and was pre-existing; `pyproject.toml` provides the canonical config. Both ensure marker warnings are suppressed regardless of which config pytest reads.

### Decision 3: Golden contract scope
**What**: Freeze only CLI flags, exit codes, JSON field names, and units — NOT frequency-dependent numeric values.
**Why**: The current code has a known bug where `--freq` does not propagate. Freezing the incorrect values would create a false baseline. The plan explicitly requires not freezing known-incorrect frequency behavior.

### Decision 4: Negative-path test design
**What**: Negative tests use `pytest.raises(AssertionError)` to verify that missing/broken lock fixtures are properly detected.
**Why**: Tests that just assert false conditions would fail in CI. The `pytest.raises` pattern validates the detection logic itself, not just the condition.

## Todo 2: Schema v2 and Contracts (2026-07-29)

### Decision 5: Auto-default version to "2"
**What**: `HardwareConfigV2.model_validator(mode="before")` defaults `version` to `"2"` when missing rather than raising an error.
**Why**: Makes programmatic construction ergonomic — `HardwareConfigV2(mac_engine=..., memory=...)` works without `version="2"` boilerplate. Wrong versions (e.g., "1") still fail.
**Alternatives**: Could have required explicit `version="2"` everywhere, but that adds noise to every test and constructor call.

### Decision 6: Bool rejection via `mode="before"` field validators
**What**: Bool-as-int detection uses `@field_validator(mode="before")` rather than `AfterValidator` in the type annotation.
**Why**: Pydantic v2 coerces `bool` to `int`/`float` BEFORE running `AfterValidator`, so `isinstance(v, bool)` would never see the original bool. `mode="before"` runs first and sees the raw input.
**Alternatives**: Could have used `StrictInt`/`StrictFloat` but that would break YAML parsing (which produces Python ints from YAML integer literals).

### Decision 7: Partial v2 coverage of real YAML files
**What**: Real YAML files (`npu_config.yaml`, `design_space.yaml`) have sections (optimizations, sfu, vector, interconnect, riscv, etc.) not yet in the v2 schema. Tests strip these before validation.
**Why**: These sections will be added to the schema in future todos (Todo 7 for workload, Todo 11 for PPA). Forcing them all into v2 now would be premature.
**Alternatives**: Could have created a "loose" validation mode that allows extra fields, but that defeats the purpose of fail-closed schema enforcement.
