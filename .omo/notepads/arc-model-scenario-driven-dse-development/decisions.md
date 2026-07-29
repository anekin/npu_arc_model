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

## Todo 4: Engine Registry and Result Contracts (2026-07-30)

### Decision 8: Engine registry as single source of truth
**What**: All engine lists (factory, DSE, CLI choices, tests) derive from `sim/engine/registry.py`.
**Why**: The codebase had 3 divergent engine lists: factory (8 engines in create_engine), DSE (8 engines in generate_configs full, 3 in quick), CLI (7 engines missing FSA). The registry eliminates divergence.
**Alternatives**: Could have used dynamic reflection on engine modules, but explicit registration is clearer and enables future configuration-driven engine discovery.

### Decision 9: Import path convention
**What**: Changed `from sim.contracts.X` → `from contracts.X` in all `sim/` files. `from sim.engine.Y` → `from engine.Y` for existing patterns.
**Why**: Standalone scripts (`python sim/npu_sim.py`) do `sys.path.insert(0, 'sim/')`, making `sim/` the import root. `sim.contracts` paths break; `contracts` works. This matches existing patterns like `from engine.mac_engine import ...`.
**Alternatives**: Could have changed scripts to add repo-root to sys.path, but that changes established behavior.

### Decision 10: `ops` deprecated property, not removed
**What**: `EngineResult.ops` is now a `@property` returning `mac_count` with `DeprecationWarning`.
**Why**: The legacy field is consumed by tests and downstream consumers. Making it a property allows gradual migration without breaking all callers at once.
**Alternatives**: Could have removed `ops` entirely with `field(init=False)` to force migration, but that would break existing physical invariant tests (Todo 3 scope).

### Decision 11: `__post_init__` validates finiteness only
**What**: `EngineResult.__post_init__` rejects NaN/Inf, negative cycles, and non-positive mac_count/op_count. It does NOT validate diagnostics or utilization bounds.
**Why**: Diagnostic validation requires engine-type context (different engines have different required keys). Utilization validation is a test/oracle responsibility, not a construction-time check.
**Alternatives**: Could have added `engine_type` field to EngineResult for self-validation, but that couples result to creation context.
