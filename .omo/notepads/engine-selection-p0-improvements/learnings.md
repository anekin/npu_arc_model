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

# Todo 3 — Fix hardcoded 12nm in MemoryTopology and CalibrationRef

**Date:** 2026-07-30

## What was done

- Modified `sim/engine/ppa_model.py`:
  - `AreaModel._memory_area_estimate()` now passes `process_node_nm=self.process_node_nm` to `MemoryTopology`.
  - `PowerModel._memory_power_estimate()` now passes `process_node_nm=area_model.process_node_nm` to `MemoryTopology`.
- Modified `sim/dse/runner.py`:
  - Imported `AreaModel` and `_node_scale_factor` from `engine.ppa_model`.
  - `evaluate_point()` now derives `CalibrationRef.process_node_nm` from `AreaModel(point.hardware_config).process_node_nm` and `node_scale` from `_node_scale_factor()`.
- Modified `sim/contracts/result.py`:
  - Kept `CalibrationRef` Pydantic defaults (`process_node_nm=12.0`, `node_scale=2.70`) unchanged for backward compatibility.
  - Updated `result_standalone_from_ppa()` to explicitly compute and pass `CalibrationRef` values from config.
- Modified tests:
  - `sim/tests/test_memory_ppa.py`: `_request()` accepts `process_node_nm`; added `test_topology_process_node_parameterized` for 7/12/22/28nm; parametrized monotonic and validation tests across nodes; kept oracle test anchored at 12nm.
  - `sim/tests/test_memory_backend.py`: `_onchip_request()` accepts `process_node_nm`; added `test_topology_process_node_parameterized`; parametrized response and PHY-rejection tests across nodes.

## Verification

- `uv run pytest sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py sim/tests/test_result_schema.py -q` — passed.
- `uv run ruff check sim/engine/ppa_model.py sim/dse/runner.py sim/contracts/result.py sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py` — All checks passed.
- `uv run basedpyright sim/engine/ppa_model.py sim/dse/runner.py sim/contracts/result.py sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py` — 0 errors, 0 warnings, 0 notes.
- Happy-path evidence: `.omo/evidence/task-3-engine-selection-p0-hardcode.json`.
- Negative-path evidence: `.omo/evidence/task-3-engine-selection-p0-hardcode-negative.txt`.

## Key findings

1. **MemoryTopology now propagates the configured node** — for 7/12/22/28nm inputs, `topology.process_node_nm` equals the input value.
2. **CalibrationRef.node_scale is now dynamic** — 7nm=1.0, 12nm=2.70 (density ratio), 22nm≈9.88, 28nm=16.0 (geometric `(node/7)²`).
3. **Backward compatibility preserved** — `CalibrationRef` schema defaults remain 12.0/2.70, so existing callers and serialized results remain valid.
4. **Backend scaling is still pending** — `Parametric3DMemoryBackend.estimate()` does not yet scale area/power by `topology.process_node_nm`; the oracle already does. This is out of scope for Todo 3 and is a prerequisite for true cross-node memory PPA.
5. **Tests now cover four nodes structurally** — node-agnostic assertions (monotonicity, validation, field propagation) run across 7/12/22/28nm; exact-oracle tests remain anchored at 12nm until backend scaling lands.

## Open questions

- Should `Parametric3DMemoryBackend` scale macro parameters by `topology.process_node_nm` in the next todo, or should scaling happen at the `AreaModel` layer?
- Do we need a calibration gate that asserts `CalibrationRef.node_scale` matches `_node_scale_factor(process_node_nm)` for every result?

# Todo 4 — Clean up legacy_result anti-pattern + expand calibration parameters

**Date:** 2026-07-30

## What was done

- Removed the non-12nm loss guards in `sim/contracts/legacy_result.py` (lines ~136 and ~209).
- Added `_calibration_from_results()` helper that propagates `CalibrationRef` fields into the legacy LLM and CV projections as a top-level `calibration` dict.
- Extended `references/calibration/parameters.yaml` with 12 new per-node PE area entries:
  - `systolic_pe_area_{7,12,22,28}nm` (T2, primary anchor)
  - `block_pe_area_{7,12,22,28}nm` (T1, architectural ratio)
  - `fsa_pe_area_{7,12,22,28}nm` (T1, architectural ratio)
  - All entries include `source_uri: references/area_sources.md` and calibrated ranges derived from `_node_scale_factor`.
- Extended `sim/calibration/evaluate.py`:
  - `calibration_ids_for_design_point()` now reads `area_model.process_node_nm` and emits per-node PE area IDs.
  - `_actual_value()` supports `systolic_pe_area_*nm`, `block_pe_area_*nm`, and `fsa_pe_area_*nm` by deriving scaled values from `AreaModel`.
- Updated `sim/tests/test_calibration_registry.py` `EXPECTED_IDS` to include the new entries.
- Added focused tests in `sim/tests/test_calibration_evaluate.py` for per-node ID selection and `_actual_value` scaling.
- Extended `references/area_sources.md` §1 with the 28nm→22nm→12nm→7nm scaling table and a reference to `contracts/bitcell.py` for SRAM area.

## Verification

- `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` — 29 passed.
- `uv run ruff check sim/contracts/legacy_result.py sim/calibration/evaluate.py` — All checks passed.
- `uv run basedpyright sim/contracts/legacy_result.py sim/calibration/evaluate.py` — 0 errors, 0 warnings, 0 notes.
- Happy-path evidence: `.omo/evidence/task-4-engine-selection-p0-legacy.json` shows `calibration.process_node_nm == 28.0` and `calibration_marked_as_loss == false`.
- Negative-path evidence: `.omo/evidence/task-4-engine-selection-p0-legacy-negative.txt` shows unknown `systolic_pe_area_13nm` is fail-closed as `unknown_calibration_id`.

## Key findings

1. **The old 12nm-only loss guard was an anti-pattern** — `CalibrationRef` now carries the real process node from `AreaModel`; projecting any node to legacy should not be treated as data loss.
2. **Per-node calibration entries make the registry node-aware** — `evaluate.py` no longer hardcodes `systolic_pe_area_7nm`; it selects the ID matching the configured node and derives the actual scaled value from `AreaModel`.
3. **12nm uses the density ratio 2.70×, not geometric (12/7)²** — the new `systolic_pe_area_12nm = 5.4 mm²` reflects the TSMC 12FFC density correction already present in `_node_scale_factor`.
4. **Registry cardinality changed** — the canonical registry grew from 10 to 21 entries; the test fixture `EXPECTED_IDS` was updated to match.

## Open questions

- Should `block_systolic_pe_ratio` be retired now that per-node `block_pe_area_*nm` entries exist, or kept as an independent cross-check?
- Should `os_pe_area_*nm`, `is_pe_area_*nm`, `tc_pe_area_*nm`, `wmma_pe_area_*nm`, and `gmma_pe_area_*nm` be added to the registry for full engine coverage, or is the current systolic/block/fsa set sufficient for P0?

# Todo 5 — Cross-node area regression tests + oracle node-scale alignment

**Date:** 2026-07-30

## What was done

- Created `sim/tests/test_area_cross_node.py` with 28 parameterized tests covering all 8 engine types × 4 process nodes (7/12/22/28nm):
  - `TestTotalAreaMonotonic`: total area strictly decreases as node shrinks (28 > 22 > 12 > 7).
  - `TestSramShare`: SRAM/total share increases across geometric nodes (28 < 22 < 7), with 12nm treated as a known high outlier due to the TSMC 12FFC density-ratio correction (2.70×).
  - `TestRelativeRatioDirection`: absolute SRAM-heavy area disadvantage (8MB L2 vs 512KB L2) grows monotonically from 7nm → 12nm → 22nm → 28nm.
  - `TestNodeScaleReference`: oracle `_node_scale` matches `engine.ppa_model._node_scale_factor` at all four nodes.
- Rewrote `sim/tests/oracles/ppa.py`:
  - `_node_scale()` now anchors at 7nm = 1.0, returns 2.70 at 12nm, and uses `(node / 7.0) ** 2` elsewhere.
  - Re-anchored `_MEMORY_DIE_AREA_PER_GB_MM2`, `_TSV_AREA_PER_GBPS_MM2`, `_PHY_AREA_FIXED_MM2`, and `_PACKAGE_AREA_FIXED_MM2` from 12nm to 7nm by dividing each by 2.70.
  - Verified 12nm oracle output is unchanged and 28nm old output × 2.70 ≈ new output.

## Verification

- `uv run pytest sim/tests/test_area_cross_node.py -q` — 28 passed.
- `uv run pytest sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py -q` — 110 passed (existing 12nm oracle-anchored tests still pass).
- `uv run ruff check sim/tests/test_area_cross_node.py sim/tests/oracles/ppa.py` — All checks passed.
- `uv run basedpyright sim/tests/test_area_cross_node.py sim/tests/oracles/ppa.py` — 0 errors, 0 warnings, 0 notes.
- Happy-path evidence: `.omo/evidence/task-5-engine-selection-p0-cross-node.json`.
- Negative-path / old-vs-new compatibility evidence: `.omo/evidence/task-5-engine-selection-p0-cross-node-negative.txt`.

## Key findings

1. **Oracle is now aligned with `AreaModel`** — both use 7nm baseline, 12nm density-ratio 2.70×, and geometric scaling for other nodes. This removes the ~7.4× inconsistency at 7nm and the ~0.17× inconsistency at 28nm that existed when the oracle anchored at 12nm.
2. **12nm is a non-monotonic outlier for SRAM share** — because `_node_scale_factor(12) = 2.70` is below pure geometric `(12/7)² = 2.94`, logic area at 12nm is smaller than geometric prediction, raising SRAM share above the 7nm value. Tests account for this by checking the geometric-node trend (28 < 22 < 7) and asserting 12nm > 7nm.
3. **Absolute SRAM-heavy disadvantage grows monotonically** — the extra area of an 8MB L2 vs 512KB L2 block engine grows from 3.9 mm² (7nm) → 10.7 mm² (12nm) → 13.3 mm² (22nm) → 18.4 mm² (28nm). This confirms that SRAM-heavy designs pay a larger absolute area penalty at older nodes.
4. **Existing 12nm oracle tests are preserved** — re-anchoring constants by dividing by 2.70 keeps all 12nm values identical, so `test_memory_ppa.py` oracle-anchored assertions still pass.

## Open questions

- Should the cross-node test matrix also include on-chip 3D DRAM memory configs to cover memory-die area scaling?
- Should the 12nm density-correction outlier be documented in `references/area_sources.md` as a known non-monotonicity?

# Todo 6 — Add access_type field to MemoryAccessPattern

**Date:** 2026-07-30

## What was done

- Added `AccessType(str, enum.Enum)` to `sim/models/memory_backend.py` with `SEQUENTIAL="sequential"` and `RANDOM="random"`.
- Added `access_type: AccessType` field to `MemoryAccessPattern` with default `AccessType.SEQUENTIAL`. Validation is enforced by the Pydantic enum field — any value other than `"sequential"` or `"random"` raises `ValidationError` with the message `"Input should be 'sequential' or 'random'"`.
- Updated both `MemoryAccessPattern` creation sites in `sim/engine/ppa_model.py` (`_memory_area_estimate()` and `_memory_power_estimate()`) to pass `access_type=AccessType.SEQUENTIAL`.
- Updated test helpers in `sim/tests/test_memory_ppa.py` and `sim/tests/test_memory_backend.py` to accept an `access_type` parameter (defaulting to `AccessType.SEQUENTIAL`).
- Added documentation comments in `sim/engine/mac_engine.py` (weight DMA = sequential, KV cache = random), `sim/engine/compiler.py` (weight DMA/activation = sequential), and `sim/models/kv_cache.py` (KV cache = random).
- Created `sim/tests/test_memory_access_pattern.py` with 19 tests covering:
  - Schema validation (4 valid values accepted, 3 invalid values rejected)
  - Default value behaviour (defaults to SEQUENTIAL)
  - Serialization roundtrip (model_dump → model_validate, JSON roundtrip)
  - Integration with engine creation sites (ppa_model patterns default to SEQUENTIAL)
  - Frozen instance immutability

## Verification

- `uv run pytest sim/tests/test_memory_access_pattern.py -q` — 19 passed.
- `uv run pytest sim/tests/test_memory_backend.py -q` — 41 passed.
- `uv run ruff check sim/models/memory_backend.py sim/engine/mac_engine.py sim/engine/compiler.py sim/models/kv_cache.py sim/tests/test_memory_access_pattern.py sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py` — All checks passed.
- `uv run basedpyright sim/models/memory_backend.py sim/engine/mac_engine.py sim/engine/compiler.py sim/models/kv_cache.py sim/tests/test_memory_access_pattern.py sim/tests/test_memory_ppa.py sim/tests/test_memory_backend.py` — 0 errors, 0 warnings, 0 notes.
- Happy-path evidence: `.omo/evidence/task-6-engine-selection-p0-pattern.json`.
- Negative-path evidence: `.omo/evidence/task-6-engine-selection-p0-pattern-negative.txt`.

## Key findings

1. **Pydantic str enum provides automatic validation** — using `AccessType(str, enum.Enum)` as the field type gives us model_dump serialization to string, model_validate deserialization from string, and automatic rejection of invalid values with a clear error message. No need for a separate `pattern` regex validator.
2. **MemoryAccessPattern has exactly 4 creation sites** (2 × ppa_model, 2 × test helpers). The `mac_engine.py`, `compiler.py`, and `kv_cache.py` files do NOT directly create `MemoryAccessPattern` objects; they model DMA/KVCache at a higher abstraction level using `dram_efficiency` and `MemoryAccessPlan`.
3. **Default access_type is SEQUENTIAL** — all existing callers that omit the new field continue to work unchanged. Only sites that need RANDOM (future KV cache DRAM modeling) need to pass `access_type=AccessType.RANDOM` explicitly.
4. **The test file validates at the schema level, not the integration level** — creation-site tests verify that the ppa_model pattern defaults to SEQUENTIAL. Full integration with pattern-based DRAM efficiency (mac_engine/compiler/kv_cache pathways) belongs in Todo 7.

# Todo 7 — Pattern-based DRAM efficiency audit + implementation

**Date:** 2026-07-30

## Audit of existing DRAM efficiency helpers in `sim/engine/mac_engine.py`

### `_dram_eff_for_bytes(transfer_bytes: int) -> float` (lines ~190–207)

**Purpose:** Compute a DRAM utilization factor for *sequential* weight loads.  The helper assumes the transfer is a contiguous bulk read (cf. `AccessType.SEQUENTIAL`) and models how much of the weight working set fits in the weight portion of L2 SRAM (`self.wbuf_kb = l2_shared_kb * 0.6`).

**Formula:**
- `transfer_bytes <= 0` → `1.0` (no-op)
- `weight_mb <= wbuf_mb` → `0.0` (fully cached; caller should skip DRAM)
- otherwise `0.55 + 0.40 * (wbuf_mb / weight_mb) / (0.3 + wbuf_mb / weight_mb)`

**Value range:**
- `{0.0}` when the weight tile fits in the weight buffer
- `(0.55, 0.92]` for partially cached / large sequential transfers (asymptote ≈ 0.95 when weight_mb ≪ wbuf_mb)

**Relationship to new model:** This helper stays intact.  After Todo 7 it is used *only* inside the sequential weight path if an engine wants a per-transfer cache scaling factor.  The new sequential baseline efficiency is `dram_efficiency` (default 0.90), applied before this helper, so the final effective bytes/cycle for a large weight transfer is `bw_raw * dram_efficiency * _dram_eff_for_bytes(...)`.

### `_kv_dram_efficiency(kv_bytes: int) -> float` (lines ~209–216)

**Purpose:** Compute a DRAM utilization factor for *random* KV cache reads.  The helper models the SRAM-resident KV fraction using the KV portion of L2 SRAM (`self.kvbuf_kb = l2_shared_kb * 0.4`).

**Formula:**
- `kv_bytes <= 0` → `1.0` (no-op)
- `ratio = kvbuf_mb / max(kv_mb, 0.001)`
- `0.55 + 0.40 * ratio / (0.3 + ratio)`

**Value range:**
- `(0.55, 0.92]` for typical KV working sets (asymptote ≈ 0.95 when kv_mb ≪ kvbuf_mb)
- Returns `1.0` for zero-byte corner cases

**Relationship to new model:** This helper stays intact and is multiplied into the *random* KV bandwidth baseline.  The new random bandwidth efficiency is `dram_efficiency_random_bw` (default 0.50), so the final effective bytes/cycle for KV reads is `bw_raw * dram_efficiency_random_bw * _kv_dram_efficiency(kv_bytes) * bw_multiplier`.  The additional `random_latency_penalty_cycles` (default 40) is added only on KV misses, independent of bandwidth.

## Design decisions for Todo 7

1. **No silent duplication.** `_dram_eff_for_bytes()` and `_kv_dram_efficiency()` are kept unchanged.  The new pattern-based layer introduces `dram_efficiency_random_bw` and `random_latency_penalty_cycles` config parameters; the old `dram_efficiency` is re-documented as the *sequential* baseline.
2. **Two effective bandwidths in `mac_engine.py`.**
   - `eff_bw_weight = bw_raw * dram_efficiency * bw_multiplier` — used for weight and activation reads (`AccessType.SEQUENTIAL`).
   - `eff_bw_kv_base = bw_raw * dram_efficiency_random_bw * bw_multiplier` — used for KV reads (`AccessType.RANDOM`), multiplied at call time by `_kv_dram_efficiency(kv_bytes)`.
3. **Per-call DMA helper.** A new `MACEngine._dma_cycles()` helper takes `AccessType` and, for random accesses, the KV byte count and hit flag.  This is the single place where random latency penalty is applied, ensuring it is added only on KV miss.
4. **`sim/models/dma.py` also understands `AccessType`.** `DMAModel.estimate_transfer()` accepts an optional `access_type` argument (default `SEQUENTIAL`) and applies the same sequential/random efficiency split, so callers outside the engine family (golden executor, NPU sim, param sweep) can opt into pattern-aware DMA.
5. **Backward compatibility.** `dram_efficiency` remains a valid config key.  `self.eff_bw` is retained as an alias to `self.eff_bw_weight` so that any code path that has not yet been migrated continues to see the same sequential value.

## Open questions

- Should `mac_engine.py`, `compiler.py`, and `kv_cache.py` be refactored to create `MemoryAccessPattern` objects directly (for Todo 7 integration), or should the pattern-based DRAM efficiency layer accept `AccessType` from the caller independently?
- The current `AccessType` has only two values. Do we need a third for mixed/streaming patterns (e.g., gather + compute)?
