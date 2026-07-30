# Todo 1 — SRAM bitcell area lookup & data provenance

**Date:** 2026-07-30

## What was done

- Created `sim/contracts/bitcell.py` with `BitcellTable` (TSMC HD bitcell data for 7/12/22/28nm) and `sram_area_mm2()` convenience function.
- Created `scripts/p0_c1_sram_calibration_gate.py` — cross-checks bitcell-derived area against TPUv1 (28nm) and RK1828 (22nm) external references; halts only if disagreement > ±30%.
- Created `sim/tests/test_bitcell_table.py` — 28 tests (positive + negative) covering known nodes, error paths, overhead bounds.
- Updated `references/area_sources.md` §4 (bitcell reference) and §7 (limitations).

## Key findings

1. **Old model (`l1_per_kb` × `node_scale`) grossly overestimates SRAM area** — the geometric scaling `(node/7)²` applied to a fixed mm²/KB constant produces ~10× larger area than the true bitcell-based calculation. The old model was computing area for a hypothetical SRAM that's physically impossible.
2. **TSMC HD bitcell data is self-consistent** — the four known nodes (7/12/22/28nm) follow a smooth sub-quadratic scaling trend consistent with published foundry roadmaps.
3. **Peripheral overhead is the dominant unknown** — the 1.5× (L1) / 1.3× (L2) defaults are placeholders; actual overhead depends on macro size, bank count, and ECC inclusion. Calibration against real chip die shots is needed.
4. **External refs pass** — TPUv1 (28nm, 28 MiB UB) and RK1828 (22nm, ~8 MiB SRAM) both produce bitcell-derived SRAM areas within ±30% of die-shot estimates.

## Open questions

- Should we add interpolated nodes (e.g. 16nm = 16FFC) via (node/7)² scaling from the 7nm baseline?
- The peripheral overhead should eventually become a function of capacity (not a fixed multiplier).
- Samsung/Intel bitcell data would be needed for multi-foundry scenarios.
