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

# Todo 2 — Refactor AreaModel: SRAM via bitcell table, logic via node_scale

**Date:** 2026-07-30

## What was done

- Modified `sim/engine/ppa_model.py` `AreaModel.__init__`:
  - Stored `self.process_node_nm` (float) from config.
  - Kept `self.l1_per_kb` / `self.l2_per_kb` as legacy fallback values (no longer multiplied by `node_scale`).
  - Added `self._bitcell_table: BitcellTable` and config-driven `l1_overhead` (default 1.5) / `l2_overhead` (default 1.3).
  - Preserved all logic-area baselines (PE variants, SFU, DMA, PCIe, DRAM PHY, crossbar, RISC-V) multiplied by `node_scale`.
- Modified `AreaModel.estimate()` to compute L1/L2 SRAM area via `sram_area_mm2()` from `sim.contracts.bitcell`.
- Modified `PowerModel.estimate()` to compute SRAM area via `sram_area_mm2()` instead of the legacy `area_model.l1_per_kb`.
- Added deprecation comment for `l1_per_kb` / `l2_per_kb` to document the legacy fallback path.

## Verification

- `uv run pytest sim/tests/test_engine_physical_invariants.py sim/tests/test_engine_result_contract.py -q` — passed.
- `uv run ruff check sim/engine/ppa_model.py` — All checks passed.
- `uv run basedpyright sim/engine/ppa_model.py` — 0 errors, 0 warnings, 0 notes.
- `uv run pytest sim/tests/test_bitcell_table.py -q` — passed.
- Manual comparison captured in `.omo/evidence/task-2-engine-selection-p0-areamodel.json`.
- Negative/legacy path captured in `.omo/evidence/task-2-engine-selection-p0-areamodel-negative.txt`.

## Key findings

1. **SRAM area drops sharply** — 12nm 512KB L1 is now ~0.776 mm² vs the old geometrically-scaled model ~2.76 mm² (-72%). This matches Todo 1's finding that the old model overestimated SRAM area by ~10× at 7nm and ~3× at 28nm.
2. **Total area impact is modest** — for a 12nm block 128×128 + 512KB L1 + 2MB L2 + LPDDR5 config, total area changes from ~126.5 mm² to ~119.1 mm² (-5.9%). The task's "<5%" expectation appears to refer to total configuration area, not SRAM area alone; the exact number depends on array size and memory subsystem selection.
3. **Cross-node monotonicity holds** — 512KB L1 area is strictly ordered 7nm (0.283) < 12nm (0.776) < 22nm (0.965) < 28nm (1.332), as required.
4. **PowerModel stays consistent with AreaModel** — both now use the same bitcell-derived L1/L2 areas; derived SRAM mm² in power equals L1+L2 area.
5. **Legacy configs still parse** — `process_node`, `l1_sram_per_kb_mm2`, and `l2_sram_per_kb_mm2` are retained as backward-compatible fallback keys. Unknown nodes (e.g., 14nm) raise `ConfigError` because they are absent from the bitcell table.

## Open questions

- The exact "<5% total area difference" target is configuration-dependent. For smaller logic footprints (e.g., block 64×128), the SRAM refactor impact on total area will be proportionally larger.
- Should we expose `l1_overhead` / `l2_overhead` in `sim/config/design_space.yaml` for sensitivity sweeps?
- When interpolated nodes (e.g., 16nm) are added, `AreaModel` will need a fallback strategy or a broader bitcell table.
