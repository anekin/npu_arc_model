# Final F4: Scope Fidelity — VERDICT: APPROVE

**Date**: 2026-07-28
**Reviewer**: Sisyphus-Junior (F4)
**Baseline**: `844cadf` (pre-fix)
**Target**: `HEAD` (post-fix)

---

## 1. Diff vs Scope Allowlist

`git diff --stat 844cadf..HEAD` → 22 files, 1955 insertions(+), 600 deletions(-)

| # | File | Plan Trace | Verdict |
|---|------|-----------|---------|
| 1 | `.omo/plans/dse-engine-model-bug-fix.md` | Plan tracking (checkbox updates) | ✓ Expected |
| 2 | `README.md` | Todo 9 (all-engine smoke test step) | ✓ Expected |
| 3 | `docs/NPU_Engines_Architecture_Guide.md` | Todo 11 (ranking table, formulas, annotations) | ✓ Expected |
| 4 | `docs/NPU硬件详细架构设计v0.1.md` | Todo 11 (PPA table, OS PE, version history) | ✓ Expected |
| 5 | `pytest.ini` | Must have #8 (testpaths, pythonpath) | ✓ Required |
| 6 | `reports/dse-engine-model-bugs-postfix-2026-07-27.md` | Todo 11 (post-fix evidence) | ✓ Expected |
| 7 | `sim/config/__init__.py` | Todo 1 (package init) | ✓ Expected |
| 8 | `sim/config/design_space.yaml` | Must have #9 (gmma.pipeline_scale) | ✓ Required |
| 9 | `sim/config/npu_config.py` | Todo 1 (load_config() loader) | ✓ Expected |
| 10 | `sim/config/npu_config.yaml` | Must have #9 (gmma.pipeline_scale) | ✓ Required |
| 11 | `sim/design_space_explorer.py` | Must have #6 (fail-closed errors) | ✓ Required |
| 12 | `sim/engine/gmma_engine.py` | Must have #5/8 (pipeline scale + raw-DMA floor) | ✓ Required |
| 13 | `sim/engine/os_systolic_engine.py` | Must have #3/6 (K-reduction + DMA) | ✓ Required |
| 14 | `sim/engine/systolic_engine.py` | Must have #1/2 (decode + prefill) | ✓ Required |
| 15 | `sim/engine/tensor_core_engine.py` | Must have #4/7 (descriptor overhead) | ✓ Required |
| 16 | `sim/tests/conftest.py` | Todo 1 (engine_config fixture) | ✓ Expected |
| 17 | `sim/tests/test_calibration_config.py` | Todo 2 (GMMA/TensorCore config validation) | ✓ Expected |
| 18 | `sim/tests/test_dse_coverage.py` | Todo 9 (engine list + quick mode + pytest.ini) | ✓ Expected |
| 19 | `sim/tests/test_dse_strict.py` | Todo 5 (fail-closed DSE error handling) | ✓ Expected |
| 20 | `sim/tests/test_engine_result_contract.py` | Todo 4 (cross-engine result shape) | ✓ Expected |
| 21 | `sim/tests/test_engines.py` | Must have #7 (CLI baselines + engine tests) | ✓ Required |
| 22 | `sim/tests/test_standalone_assets.py` | Todo 9 (asset completeness) | ✓ Expected |

**Result**: All 22 files trace to a documented todo or a "Must have" scope item. Zero unexplained modifications.

---

## 2. Original Report SHA256

```
Expected: 61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3
Actual:   61fe73e163f4dc61c1c746ea3a115b176c4d745bf387c7b2a4350a195d88ccd3
Result:   MATCH ✓
```

File `reports/dse-engine-model-bugs-2026-07-27.md` was NOT in the diff range (`git diff --name-only 844cadf..HEAD` returned empty for this path).

---

## 3. `.omo/ultraresearch/` Boundary

`git diff --name-only 844cadf..HEAD -- '.omo/ultraresearch/'` → **no output**.

No files under `.omo/ultraresearch/` were modified in the commit range. Existing research artifacts remain untouched.

---

## 4. CaduceusCore / External Repo Boundary

`git diff --name-only 844cadf..HEAD -- ../CaduceusCore` → `fatal: outside repository`.

CaduceusCore is outside the current repository; no files from that repo can be modified here. The repo boundary is enforced by git itself.

Additionally verified: `sim/golden_executor.py`, `sim/models/mxu.py`, and `sim/npu_sim.py` (Golden Executor / Func Model entry points) were NOT in the diff range.

---

## 5. Claimed Files Existence

All 12 new files verified present on disk:
- `pytest.ini` ✓
- `sim/config/__init__.py` ✓
- `sim/config/npu_config.py` ✓
- `sim/tests/conftest.py` ✓
- `sim/tests/test_calibration_config.py` ✓
- `sim/tests/test_dse_coverage.py` ✓
- `sim/tests/test_dse_strict.py` ✓
- `sim/tests/test_engine_result_contract.py` ✓
- `sim/tests/test_standalone_assets.py` ✓
- `reports/dse-engine-model-bugs-postfix-2026-07-27.md` ✓

All pre-existing modified files (engines, configs, DSE) also confirmed on disk.

---

## Verdict

**APPROVE** — All scope boundaries verified:

- ✓ 22 files in diff = 9 must-have entries + 11 todo deliverables + 1 plan tracker + 1 postfix report
- ✓ Original report SHA256 unchanged (`61fe73e...`)
- ✓ `.omo/ultraresearch/` untouched
- ✓ No CaduceusCore / external repo files modified
- ✓ Golden Executor, MXUModel, npu_sim not touched
- ✓ All claimed test/config/report files exist and are tracked/created
- ✓ Zero unexplained modifications
