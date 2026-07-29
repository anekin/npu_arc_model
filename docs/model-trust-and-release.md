# Model Trust Levels and Release Profiles

This document defines the trust framework used to qualify Arc Model results and
the release profiles under which architecture recommendations may be published.

## Trust Levels for Individual Parameters

Every decision-driving physical parameter carries a trust level from the
calibration registry (`references/calibration/parameters.yaml`):

| Level | Meaning | Permitted Use |
|-------|---------|---------------|
| **T0** | Engineering assumption, no direct evidence | Exploratory sensitivity only |
| **T1** | Published proxy or analytic bound | Feasibility / bound arguments |
| **T2** | Reproduced from verified source + held-out validation | Relative ranking inside calibrated range |
| **T3** | Signed-off reference RTL or silicon | Numeric prediction with residual interval |

A design point's effective trust level is the **minimum** trust level among all
parameters it consumes.  One T0 assumption is enough to keep the whole point
exploratory.

## Run Trust Levels

`DesignSpaceResultV2.trust_level` classifies an entire run:

| Level | Conditions |
|-------|------------|
| `authoritative` | Complete coverage, no failures, all ranking parameters T2+ and in range |
| `calibrated_estimate` | Calibrated model but missing secondary coverage axes |
| `exploratory` | Contains T0/T1 parameters or extrapolated values |
| `non_authoritative` | Partial run, failures, or coverage gaps on required axes |

## Release Profiles

### `experimental`

- Allows T0/T1 parameters.
- Requires all exploratory points to be explicitly tagged (`trust_level=exploratory`).
- Requires complete coverage manifest (`missing_axes={}`).
- Requires valid content-addressed artifact hashes.
- Does **not** publish promoted rankings.

Pass command:

```bash
uv run python scripts/release_gate.py --profile experimental
```

### `decision-grade`

- Every Pareto-driving parameter must be T2+.
- Every design point must be inside its calibration range.
- No extrapolated winner may enter the recommendation set.
- Coverage must be complete and the worktree must be clean.
- Generates a content-addressed release bundle under
  `artifacts/releases/<run-id>/`.

Because the current calibration registry keeps several ranking drivers at T0/T1
(e.g. `gmma_pipeline_scale`, `tensor_core_descriptor_overhead`),
`decision-grade` is expected to fail until additional measured evidence is
provided.  This is intentional and prevents uncalibrated rankings from being
promoted as authoritative.

```bash
# Expected to fail until T2+ evidence is added for T0/T1 parameters.
uv run python scripts/release_gate.py --profile decision-grade
```

## Release Artifacts

A successful gate writes:

```
artifacts/releases/<run-id>/
├── manifest.json    # profile, scenario, commit, bundle digest
└── SHA256SUMS       # checksums over canonical payload files
```

Existing artifact directories are never overwritten.  The canonical payload
(inputs + result + coverage + manifest) must reproduce byte-identically on a
clean checkout.

## Evidence Requirements

Each todo in the development plan must have matching evidence under
`.omo/evidence/`.  The evidence ledger verifier (`scripts/verify_evidence_ledger.py`)
checks that every todo and final verification wave (F1-F4) has a file, that
recorded commands exit 0, and that artifact digests are present.

## Scope Rules

The scope verifier (`scripts/verify_scope.py`) enforces:

- No PyTorch, ROS, Ramulator, or DRAMSim dependencies in phase one.
- Dated historical reports (`reports/dse-engine-model-bugs-2026-07-27.md` and
  `reports/dse-engine-model-bugs-postfix-2026-07-27.md`) are not modified.
- `.omo/ultraresearch/20260723-vla-models/sources/` is not staged.
- Every current recommendation in `docs/publication-manifest.yaml` binds to a
  run manifest.

## Mutation Tests

The acceptance suite includes monkey-patch mutations that prove the following
regressions are caught:

- Frequency forced to 1000 MHz.
- GMMA ideal MAC floor removed.
- TensorCore descriptor overhead ignored.
- Memory spill forced to zero.
- Unknown CV op returning zero cycles.
- Design-point positional association restored.
- Partial/non-authoritative point entering Pareto frontier.

Each mutation test restores state in teardown and asserts that a gate or oracle
fails.
