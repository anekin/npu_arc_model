# Calibration Raw Fixtures

This directory contains tiny, deterministic raw calibration fixtures used by
`scripts/calibrate_mxu_model.py`.  They are intentionally minimal so that
checksum-bound tests can run offline without real RTL.

## Files

- `mxu_train.csv` — training cases for MXUModel vs RTL cycle fitting.
- `mxu_heldout.csv` — held-out cases; must never participate in fitting.
- `SHA256SUMS` — SHA-256 hashes of the two CSV files.

## Format

Each CSV has a header and one row per case:

```
case_id,M,N,K,measured_cycles
```

- `case_id` — stable, deterministic identifier.
- `M`, `N`, `K` — GEMM dimensions.
- `measured_cycles` — synthetic RTL-equivalent cycle count for this fixture.

## Trust semantics

These fixtures are **synthetic** and serve only to exercise the calibration
pipeline (checksum validation, train/held-out separation, digest stability,
and fail-closed behavior).  They are not a substitute for real measured RTL
and therefore remain **T0** provenance for any real ranking decision.
