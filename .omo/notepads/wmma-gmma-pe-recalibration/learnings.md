# Todo 9 — Decision-grade impact assessment + documentation update

**Date:** 2026-07-31

## What was done

- **README.md §1.4 "跨节点验证结论" cell** — the decision-grade FAIL clause was updated from `FAIL（WMMA/GMMA PE 比仍 T0，频率-节点绑定为探索性结论）` to `FAIL（频率-节点绑定为探索性结论，多节点覆盖不完全）`. Added a sentence reflecting that WMMA/GMMA PE calibration is now T1 (PE 面积 4.5/5.5 mm² @7nm、WMMA 片段序列化 120 cycles、GMMA pipeline，源自 H100 SM die/Volta 架构分析) and that WMMA full-model tok/s is still low (~0.5) but improved ~10× (per-FFN_down-GEMM 6.9→67.6). **§1.3 "关键技术决策" table has no decision-grade row** — the plan's line reference (README.md:59-60) pointed at the pre-edit layout; the FAIL clause actually lives in §1.4, which is where it was edited.
- **docs/model-trust-and-release.md "Decision-Grade State" section** — replaced the "WMMA/GMMA PE 比率仍为 T0" bullet with an updated bullet: the 7 WMMA/GMMA entries (`gmma_pipeline_scale`, `tma_overlap`, `wmma_fragment_serialization_cycles`, `wmma_pe_ratio`, `gmma_pe_ratio`, `wmma_pe_area_7nm`, `gmma_pe_area_7nm`) are now T1 and no longer a FAIL reason, but `tensor_core_descriptor_overhead` remains T0 so T0 parameters are not fully eliminated. Second bullet rewritten to state the remaining FAIL reasons explicitly: 频率-节点绑定为探索性结论 + 多节点覆盖不完全 (also dropped the stale "os_systolic 的 PE 面积参数为 T0" claim — os_systolic uses `os_pe_area_mm2` default 4.0 = block PE baseline, which is T1 in the registry). Updated the gate-prediction comment date/reason.
- **docs/model-trust-and-release.md §跨节点引擎选择发现 trust note** — "WMMA/GMMA PE 比等参数仍为 T0" → notes the 2026-07-31 T1 upgrade while clarifying the cross-node tables still reflect the pre-calibration WMMA cycle model (ser=1600).
- **docs/NPU_Engines_Architecture_Guide.md** — replaced all WMMA `6.9 tok/s` values with calibrated `67.6 tok/s` (per-FFN_down-GEMM metric, which is the guide's stated metric at line 4: M=1, K=11008, N=2048): overview diagram, area list, §2.5 performance row (6.9→67.6, 比 Block 慢 370×→~38×), 一句话 summary, quick-reference table, and the §五 mechanism diagram (10c DMA wait → 120c serialization + 48c compute; 100万 cycles → 1,060 万 cycles). Added a calibration note block in §2.5 clarifying the metric split (per-GEMM 67.6 vs full-model ~0.5) and provenance. Root-cause row updated to the calibrated serialization model (120c + 32c sync + 16c MAC × 88,064 fragments ≈ 1,480 万 cycles). Trust note at top of guide no longer lists `gmma_pipeline_scale` as T0.

## Key findings

1. **The plan's "README.md §1.3" reference for the decision-grade clause was stale.** §1.3 is the 关键技术决策 table (no decision-grade row); the FAIL clause lives in §1.4's 跨节点验证结论 cell. Edited the real location. Plan's README.md:59-60 line refs predate subsequent doc edits.
2. **Empirically re-derived the calibrated numbers before writing them into docs** (per plan QA rule "不得用 grep 命中、worker 自述或历史 JSON 代替实际执行"): WMMA per-FFN_down-GEMM @ser=120 = **67.59 tok/s** (pre-cal ser=1600 → 6.89), Block = 2540.43 tok/s, WMMA/block ratio = 0.0266, Block/WMMA = 37.6×, improvement = 9.8×. Full-model `npu_sim --engine wmma --json` → **0.476 tok/s** ≈ 0.5. All doc numbers trace to these runs.
3. **Metric split matters in the guide**: the guide's header (line 4) defines its numbers as per-FFN_down-GEMM, so 6.9→67.6 is the right replacement there; the full-model ~0.5 figure is stated separately to avoid conflating the two (the pre-edit "370× slower" was per-GEMM vs per-GEMM and became ~38×).
4. **"os_systolic 的 PE 面积参数为 T0" in the old Decision-Grade bullet was stale** — `AreaModel` computes `os_pe_baseline` from `os_pe_area_mm2` default 4.0 (comment: "output stationary ≈ block"), i.e. block PE (registry T1). Dropped the claim; kept the real remaining reasons.
5. **decision-grade stays FAIL with two remaining reasons**: 频率-节点绑定为探索性结论 (frequency bounds from architecture reasoning, not silicon) and 多节点覆盖不完全 (cross-node coverage now fuller after Plan B but still exploratory; e.g. 3D DRAM axis not in search space). `tensor_core_descriptor_overhead` T0 is an additional contributing factor, not the primary one.

## Verification

- Values: per-GEMM 67.59 tok/s (ser=120) / 6.89 (ser=1600), full-model 0.476 tok/s, WMMA/block ratio 0.0266, Block/WMMA 37.6×.
- `grep "WMMA/GMMA PE 比仍 T0"` across README/docs → no hits; remaining "6.9" hits are before→after calibration notes.
- `uv run ruff check .` → All checks passed.
- `uv run pytest sim/tests/test_engines.py sim/tests/test_calibration_registry.py -q` → see evidence file.
- Evidence: `.omo/evidence/task-9-wmma-gmma-pe-recalibration-docs.json` + `-negative.txt`.

---

# Todo 8 — Calibration parameter registry update (WMMA/GMMA cycle + area params)

**Date:** 2026-07-31

## What was done

- Added 2 new entries to `references/calibration/parameters.yaml` (both T1, non-empty source_uri, numeric range_min/range_max):
  - `wmma_pe_area_7nm`: value=4.5, unit=mm2, calibration_range 3.5–5.0 mm2, source_uri = NVIDIA H100/Hopper whitepaper URL (same verified URI used by `wmma_pe_ratio`/`gmma_pe_ratio`/`gmma_pipeline_scale`).
  - `gmma_pe_area_7nm`: value=5.5, unit=mm2, calibration_range 5.0–6.5 mm2, same source_uri.
- Verified the 3 entries from prior todos are already registered with T1 + non-empty source_uri: `wmma_fragment_serialization_cycles` (Todo 2, Volta Tuning Guide, [50, 200]), `gmma_pipeline_scale` (Todo 3), `wmma_pe_ratio`/`gmma_pe_ratio` (Todos 5/6).
- Updated `sim/calibration/evaluate.py`:
  - `calibration_ids_for_design_point()` — WMMA engine set now adds `wmma_fragment_serialization_cycles`.
  - `_actual_value()` — added extraction for `wmma_fragment_serialization_cycles` (`hw_config["wmma"]["fragment_serialization_cycles"]`, default 120), `wmma_pe_area_7nm` (`area_model["wmma_pe_area_mm2"]`, default 4.5), `gmma_pe_area_7nm` (`area_model["gmma_pe_area_mm2"]`, default 5.5). Area entries read the YAML config value directly (7nm baseline).
- Updated `EXPECTED_IDS` in `sim/tests/test_calibration_registry.py` — added `wmma_pe_area_7nm` and `gmma_pe_area_7nm` (`wmma_fragment_serialization_cycles` was already present from Todo 2).

## Key findings

1. **The two new area entries are registry-only anchors; they are NOT added to the engine ID set.** The plan (Todo 8 §3) explicitly wires only `wmma_fragment_serialization_cycles` into the WMMA ID set. `wmma_pe_area_7nm`/`gmma_pe_area_7nm` are 7nm-specific baselines and would be misleading for 12/22/28nm design points (the actual config value is node-scaled), so leaving them out of `calibration_ids_for_design_point` avoids a false out-of-range cap at non-7nm nodes. `_actual_value()` still extracts them so any explicit trust-gate check can use them.
2. **Unit spelling follows file convention `mm2` (ASCII), not `mm²`** — all 16 existing area entries use `mm2`; the schema (`unit: str`) accepts either, and `test_parameters_yaml_has_required_fields` only asserts truthiness. The task text's "mm²" is descriptive shorthand.
3. **`_actual_value()` fallbacks stay at the pre-calibration defaults** (wmma 6.0, gmma 7.0, block 4.0) exactly like the existing ratio/area extractors — `evaluate.py` is a reader of whatever `hw_config` carries, so the test configs (which still inject 6.0/7.0) and real YAML (4.5/5.5) both behave consistently.
4. **The exact-ID-set test is the contract for registry growth** — Todo 8 grew the registry 23→25 entries; every new ID must land in `EXPECTED_IDS` or `test_valid_registry_has_all_parameters` fails immediately. This is the third time this surface has been hit (Todo 2, Todo 3, Todo 8).
5. **Running `evaluate.py` directly requires `PYTHONPATH=sim`** (module imports are `calibration.*`/`contracts.*`/`engine.*`, rooted at `sim/`). Bare `uv run python sim/calibration/evaluate.py` fails with ModuleNotFoundError — the plan's verification command must include the PYTHONPATH prefix (same as Todo 3's notepad).

## Verification

- `python3 -c "import yaml; yaml.safe_load(open('references/calibration/parameters.yaml'))"` → exit 0, 25 entries.
- `PYTHONPATH=sim uv run python sim/calibration/evaluate.py` → exit 0.
- `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` → **29 passed**, exit 0.
- TrustGate T1 on WMMA IDs (`wmma_fragment_serialization_cycles`, `wmma_pe_ratio`, `wmma_pe_area_7nm`) with hw_config (ser=120, wmma 4.5, block 4.0 @7nm) → ok=True, max_trust=T1.
- Negative: serialization=9999 → flagged `out_of_calibration_range` (negative evidence file).
- Evidence: `.omo/evidence/task-8-wmma-gmma-pe-recalibration-params.json` + `-negative.txt`.

---

# Todo 7 — WMMA/GMMA per-node PE area regression tests

**Date:** 2026-07-31
## What was done

- Added `TestWmmaGmmaPeArea` to `sim/tests/test_area_cross_node.py` with the 4 planned tests:
  - `test_wmma_area_per_node` — WMMA PE baseline strictly decreases 28nm (72.0) > 22nm (44.45) > 12nm (12.15) > 7nm (4.5) mm².
  - `test_gmma_area_per_node` — GMMA PE baseline strictly decreases 28nm (88.0) > 22nm (54.33) > 12nm (14.85) > 7nm (5.5) mm².
  - `test_gmma_ge_wmma` — parametrized over all 4 nodes; gmma_pe_baseline > wmma_pe_baseline at every node (TMA premium).
  - `test_wmma_gmma_area_physically_plausible` — parametrized over all 4 nodes; block < wmma < gmma at every node.
- The tests read the **node-scaled `*_pe_baseline` attributes** of `AreaModel` directly (pure PE area, no SRAM/memory noise), using a new `_pe_area_config()` helper that injects the Todo 5/6 calibrated values (wmma 4.5, gmma 5.5, block 4.0, systolic 2.0) into the config.

## Key findings

1. **The code fallback defaults are still the pre-calibration values.** `ppa_model.py` keeps `wmma_pe_area_mm2` default 6.0 / `gmma_pe_area_mm2` default 7.0 (intentionally untouched by Todo 5/6). `_base_config()` does not carry the PE keys, so `AreaModel(_base_config(node))` would silently test the OLD 6.0/7.0 baselines. The new tests must pass the calibrated values explicitly via `_pe_area_config()` — otherwise they would not actually lock the calibration.
2. **Attribute access beats `estimate()["total_mm2"]` for PE-level tests.** `wmma_pe_baseline` etc. are already node-scaled (`baseline * node_scale`) and exclude SRAM/memory/package area, so the monotonicity and ordering assertions test exactly what Todo 5/6 calibrated without confounds.
3. **Strict inequality is safe across all nodes** — node_scale (16.0 / 9.88 / 2.70 / 1.0) multiplies every baseline identically, so both monotonicity (28>22>12>7) and ordering (4.0 < 4.5 < 5.5 × scale) hold with wide margin (min gap 0.5 mm² @7nm between block and wmma).
4. **Parametrize vs dict-comprehension split**: per-node ordering checks use `@pytest.mark.parametrize("process_node_nm", NODES_NM)` (4 test instances each); monotonicity checks use a dict comprehension over `NODES_NM` matching the file's existing `TestTotalAreaMonotonic` pattern.

## Verification

- `uv run pytest sim/tests/test_area_cross_node.py -q` → **38 passed** (28 baseline + 10 new instances), exit 0.
- `uv run pytest sim/tests/test_engine_physical_invariants.py -q` → **773 passed**, exit 0.
- `uv run ruff check sim/tests/test_area_cross_node.py` → clean.
- Evidence: `.omo/evidence/task-7-wmma-gmma-pe-recalibration-cross-node-area.json` (invariants all true, values per node) and `-negative.txt` (inverted wmma/gmma ordering correctly rejected).

---



**Date:** 2026-07-31

## What was done

- Changed `wmma_pe_area_mm2` default from **6.0 → 4.5 mm² @7nm** in `sim/config/design_space.yaml` (the H100 SM die-derived recommended value from the 3.5–5.0 mm² band). Updated the header comment ratio from `WMMA~1.5×` to `WMMA~1.13×`.
- Updated `wmma_pe_ratio` in `references/calibration/parameters.yaml`: value 1.5 → **1.125** (4.5/4.0), trust_level **T0 → T1**, source_uri → NVIDIA H100/Hopper whitepaper (https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper — the same verified URI used by `gmma_pipeline_scale`/`tma_overlap`), calibration_range **1.4–1.6x → 0.88–1.25x** (range_min/max 0.875/1.25, from the 3.5–5.0 mm² band over block 4.0 mm²), description rewritten to carry the die derivation.
- Added §7 "WMMA PE 面积推导（H100 SM die 参考）" to `references/area_sources.md` (the old §7 模型局限性 became §8): H100 SM ~6–8 mm² @4nm → 4 TCs/SM → ~1.0–1.5 mm²/TC @4nm → ~1.5–2.5 mm²/TC @7nm → our 128×128 INT4 array has ~16× the per-cycle MACs of an H100 TC, ×4–6 area factor → **3.5–5.0 mm² @7nm, recommended 4.5 mm² = 2.25× systolic = 1.125× block**. Also updated the §3 relative-ratio table (WMMA row + config table 6.0→4.5, 366→275 µm²/MAC) and the limitation item 4 (WMMA is no longer pure architecture reasoning).
- Did NOT touch `block_pe_area_mm2` (4.0), `systolic_pe_area_mm2` (2.0), `gmma_pe_area_mm2` (7.0 — Todo 6's scope), `AreaModel` logic, or engine performance formulas. The `ppa_model.py` code fallback default stays 6.0 — the config YAML is the source of truth (AreaModel reads `am.get("wmma_pe_area_mm2", 6.0)`), and ppa_model.py is explicitly out of scope for this todo.

## Key findings

1. **`wmma_pe_ratio` is consumed by `evaluate.py` `_actual_value()`** (line 128-133): it computes `wmma_pe_area_mm2 / block_pe_area_mm2` from the hw_config. With the new 4.5/4.0 config the actual value is 1.125, which sits inside the new registry range [0.875, 1.25] — so `TrustGate` in-range checks keep passing. Had the registry range stayed [1.4, 1.6], a hw_config with the new YAML would have flagged `out_of_calibration_range` and capped max_trust at T1.
2. **The registry exact-ID-set test (`test_valid_registry_has_all_parameters`) means new calibration IDs are a separate chore** — Todo 5 updates the existing `wmma_pe_ratio` entry only, so `EXPECTED_IDS` in test_calibration_registry.py is untouched. The plan's Todo 8 (`wmma_pe_area_7nm`, `gmma_pe_area_7nm` new entries) will have to update `EXPECTED_IDS` + add `_actual_value` extraction — not this todo.
3. **Area change does not affect performance**: WMMA tok/s is cycle-model-driven; the physical-invariant tests (`-k wmma`) and `test_area_cross_node.py` pass unchanged (area monotonicity uses strict inequality between nodes, preserved by a 4.5 vs 6.0 baseline since node_scale multiplies both).
4. **Derivation is a public proxy, not silicon measurement** (T1 by design): the 4–6× area factor for 16× MACs is a sub-linear growth assumption (regular MAC cells, wire/broadcast/accumulation networks dominate). The plan's own decision to sanity-check flags this as needing review.

## Verification

- `uv run pytest sim/tests/test_engine_physical_invariants.py -q -k wmma` → exit 0 (all wmma tests pass)
- `uv run pytest sim/tests/test_area_cross_node.py -q` → exit 0
- `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` → exit 0 (registry T1 + range-consistency holds)
- YAML parses: `python3 -c "import yaml; yaml.safe_load(open('sim/config/design_space.yaml')); yaml.safe_load(open('references/calibration/parameters.yaml'))"` → exit 0
- `_actual_value("wmma_pe_ratio", cfg)` with the new config → 1.125, inside [0.875, 1.25]

---

# Todo 6 — GMMA PE area calibration (7.0 → 5.5 mm² @7nm, H100 SM die reference)

**Date:** 2026-07-31

## What was done

- Changed `gmma_pe_area_mm2` default from **7.0 → 5.5** in `sim/config/design_space.yaml` (calibrated value = WMMA PE 4.5 + TMA/descriptor ~1.0 mm² @7nm, replacing the T0 "1.75× block" guess).
- Updated `gmma_pe_ratio` in `references/calibration/parameters.yaml`: value 1.75 → **1.375** (5.5/4.0), trust_level **T0 → T1**, source_uri = NVIDIA H100/Hopper whitepaper (https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper), calibration range [1.25, 1.625]× (= the [5.0, 6.5] mm² absolute band over block 4.0 mm²).
- Added **§8 GMMA PE 面积推导（H100 SM die 参考）** to `references/area_sources.md` (derivation table + T1 provenance), updated §3 tables (GMMA 5.5 mm² / 336 µm²/MAC / ~1.375× block), renumbered limitations to §9, and noted GMMA is no longer pure architectural reasoning in limitation #4.
- Updated `gmma_engine.py` `TMA_AREA_MM2` comment only: documented that 2.0 is a doc-only legacy constant not consumed by AreaModel, and that the die-derived value is ~1.0 mm² @7nm (see area_sources.md §8). Constant value unchanged.
- Also fixed Todo 5's stale cross-ref in area_sources.md §3 (WMMA row pointed to §8 → now §7) so the new GMMA §8 does not get misread as the WMMA section.

## Key findings

1. **`TMA_AREA_MM2 = 2.0` is documentation-only** — grep confirms it is never read by AreaModel (which takes `gmma_pe_area_mm2` from YAML) nor by any engine formula. The recalibration therefore has zero effect on performance/area math when the YAML default is consumed; the constant is purely a stale doc artifact.
2. **Ratio range must stay consistent with the schema** — `gmma_pe_ratio` is a *ratio* entry (unit: ratio), so the task's [5.0, 6.5] mm² band must be mapped over the block anchor: [5.0/4.0, 6.5/4.0] = [1.25, 1.625]. Putting mm² numbers into `range_min`/`range_max` would trip `TrustGate` out-of-calibration-range detection (value 1.375 < 5.0). The absolute band is recorded in the description and reserved for Todo 8's `gmma_pe_area_7nm` entry.
3. **Ordering invariant holds**: block 4.0 < WMMA 4.5 < GMMA 5.5 @7nm — GMMA keeps its TMA premium while both drop from the inflated 6.0/7.0 guesses. Per-node monotonicity is preserved because all PE baselines scale by the same `_node_scale_factor`.
4. **Sources**: Locuza Substack / Semianalysis URLs are not reliably fetchable from this network (timeouts/404s), so the retrievable anchor is the NVIDIA Hopper whitepaper URI (already verified in Todo 3); Locuza/Semianalysis die-shot analyses are cited as qualitative cross-reference in the description text.

## Verification

- `uv run pytest sim/tests/test_area_cross_node.py -q` → passed (area monotonicity for all 8 engines at 7/12/22/28nm).
- `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_evaluate.py -q` → passed (registry schema: T1 entries require non-null source_uri; gmma_pe_ratio still in EXPECTED_IDS; trust-gate decision-grade still fails via `tensor_core_descriptor_overhead`).
- YAML syntax: `python3 -c "import yaml; yaml.safe_load(open('references/calibration/parameters.yaml'))"` → exit 0.
- Cross-check: GMMA PE area 5.5 mm² > WMMA 4.5 mm² at 7nm (and by node_scale at every node) — GMMA keeps TMA premium.

---

# Todo 4 — WMMA/GMMA cycle-model physical invariant tests

**Date:** 2026-07-31

## What was done

- Added class `TestWmmaGmmaCycleCalibration` to `sim/tests/test_engine_physical_invariants.py` with the 4 planned tests:
  - `test_wmma_serialization_monotonic` — sweep `fragment_serialization_cycles ∈ [0, 50, 120, 200]` on FFN_down (M=1, K=11008, N=2048) @64×64/LPDDR5; asserts total_cycles non-decreasing with serialization.
  - `test_wmma_not_absurd` — WMMA per-GEMM tok/s at calibrated default (ser=120) asserted in **[1, 100]** band.
  - `test_gmma_pipeline_scale_effect` — `pipeline_scale=0.01` strict < `0.10`, verified on M=1, K=16, N=4096 (weight-resident sub-array shape, 641 < 832 cycles).
  - `test_gmma_tma_overlap_effect` — overlap sweep [0.1, 0.5, 0.9] via instance-attribute override (TMA_OVERLAP is a class constant, not YAML-driven); asserts total_cycles non-increasing AND `tma_exposed_dma` strictly decreasing.
- Evidence: `.omo/evidence/task-4-wmma-gmma-pe-recalibration-physics.json` (full suite), `-negative.txt` (ser=0 injection → utilization 0.0070 ∈ (0,1]).

## Key findings

1. **Band deviation, documented per plan rule**: the plan's [1, 30] tok/s band was written against the pre-calibration *full-model* target (10–17 tok/s). The realized per-FFN_down-GEMM value at the calibrated default is **67.6 tok/s** (ser=120, 64×64, LPDDR5-51.2 GB/s), so the task's **[1, 100]** band is used. Full-model WMMA tok/s is ~0.5 — below ANY absolute tok/s band — which is why the WMMA/block **ratio** band [0.015, 0.05] (locked in `test_wmma_calibration_ratio`, test_engines.py) remains the true calibration lock; the absolute band only catches physically absurd values.
2. **GMMA `pipeline_scale` is structurally masked in decode-shaped GEMMs.** At M=1 with K,N ≥ array dims, `raw_dma_floor ≈ (K·N/2 + M·K)/bw_raw` always exceeds `total_compute ≈ ceil((H+M+W)·scale)·K·N/4096` for scale ≤ 0.10 (13·102.4 < 4096 at 64×64), and for large M `total_compute ≈ 2·scale·ideal < ideal` whenever scale < 0.5. The knob only surfaces when weight bytes fit the SRAM weight buffer AND K < array width (sub-array, weight-resident): the test uses M=1, K=16, N=4096 → total(0.01)=641 < total(0.10)=832. Any future change to the GMMA formula that makes `pipeline_scale` visible on normal decode shapes will be caught by the strict `<` in this test.
3. **`TMA_OVERLAP` affects diagnostics only** — `total_cycles = max(compute, ideal, raw_dma_floor, total_dma_ceil)` never reads it, so the "never increases total_cycles" invariant holds as equality across the sweep. The test also asserts `tma_exposed_dma` strictly shrinks (71.5 → 35.8 → 7.2) so the knob provably applies to the overlap term. It is overridden at instance level (`engine.TMA_OVERLAP = x`) since `_parse_config` does not read it — mirrors the Todo 3 open question about whether `tma_overlap` should become config-driven.
4. **WMMA monotonicity is strict in practice**: ser 0/50/120/200 → 4.2M / 8.6M / 14.8M / 21.8M cycles (236.6 / 115.9 / 67.6 / 45.8 tok/s), so the non-decreasing assert has wide margin and also pins the calibration direction (smaller ser → strictly faster).

## Verification

- `uv run pytest sim/tests/test_engine_physical_invariants.py -q` → **773 passed** (769 baseline + 4 new), exit 0, all 8 engines still covered by the parametrized tests (untouched).
- `uv run pytest sim/tests/test_engines.py sim/tests/test_calibration_registry.py sim/tests/test_calibration_config.py -q` → 53 passed (no regression in the Todo 2/3 calibration locks).
- `uv run ruff check sim/tests/test_engine_physical_invariants.py` → clean.
- Negative path: `fragment_serialization_cycles=0` → utilization 0.0070 ∈ (0,1], physical oracle holds.

---

# Todo 3 — GMMA pipeline_scale & TMA_OVERLAP calibration references

**Date:** 2026-07-31

## What was done

- Upgraded `gmma_pipeline_scale` (value 0.05) in `references/calibration/parameters.yaml` from trust_level T0 → T1, with non-null `source_uri` pointing to the NVIDIA H100/Hopper whitepaper resource (https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper).
- Added new `tma_overlap` calibration entry: value=0.5, unit=ratio, trust_level=T1, same source_uri, `calibration_range: 0.3–0.7` (range_min/range_max fields for programmatic checks).
- Added `test_gmma_calibration_bounds()` to `sim/tests/test_engines.py` — locks `pipeline_scale ∈ [0.01, 0.10]`, `TMA_OVERLAP ∈ [0.3, 0.7]`, verifies `estimate()` details report the same values, and cross-checks both registry entries are T1 with non-empty source_uri.
- Updated `sim/tests/test_calibration_registry.py`: added `tma_overlap` to `EXPECTED_IDS`, and corrected the stale assertion that `gmma_pipeline_scale` is T0 → now asserts T1 + non-empty source_uri.

## Key findings

1. **Numerical values unchanged** — only provenance upgraded (T0→T1) and references added. No engine formula, no config default, no hardware measurement introduced.
2. **Registry schema requires `range_min`/`range_max`** — `CalibrationEntry.is_in_range()` uses the numeric fields, not the human-readable `calibration_range` string. The new entry carries both, so `TrustGate` can do programmatic out-of-range detection.
3. **Adding a calibration entry has downstream test surface** — `test_valid_registry_has_all_parameters` asserts an exact ID set, so any new `calibration_id` must be registered there too.
4. **The H100 whitepaper URI resolves** — https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper serves the NVIDIA Hopper/H100 Tensor Core whitepaper landing page (verified via fetch).

## Open questions

- Whether `tma_overlap` should be consumed directly by `evaluate.py`'s `_actual_value()` (currently it is registry-only; engine constant is class-level `TMA_OVERLAP`, not config-driven). Todo 8's registry work may add extraction logic.

## Verification

- `uv run pytest sim/tests/test_engines.py -q -k gmma_calibration` → 1 passed
- `uv run pytest sim/tests/test_engines.py -q` → 27 passed
- `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_config.py -q` → 25 passed
- `PYTHONPATH=sim uv run python sim/calibration/evaluate.py` → exit 0

---

# Todo 2 — WMMA fragment_serialization_cycles calibration (1600 → 120)

**Date:** 2026-07-31

## What was done

- Changed `wmma.fragment_serialization_cycles` default from **1600 → 120** in BOTH `sim/config/npu_config.yaml` and `sim/config/design_space.yaml` (the calibrated recommended value from the [50, 200] Volta-derived range).
- Added `wmma_fragment_serialization_cycles` entry to `references/calibration/parameters.yaml`: value=120, unit=cycles, trust_level=**T1**, source_uri=https://docs.nvidia.com/cuda/volta-tuning-guide/index.html (NVIDIA Volta Tuning Guide), calibration_range=50–200 (range_min/max=50.0/200.0).
- Updated `test_wmma_decode()` in `sim/tests/test_engines.py`: now exercises the calibrated serialization=120 explicitly and asserts `wmma_tok_s > 5` (was `< 10`); WMMA@120 = 67.6 tok/s per FFN_down GEMM, still `> 10×` all other engines' total_cycles.
- Added `test_wmma_calibration_ratio()` — the calibration lock (see "Key findings" for the band deviation).
- Updated `EXPECTED_IDS` in `sim/tests/test_calibration_registry.py` (exact-set assertion requires every registered ID).
- Fixed 2 stale trust-gate tests in `sim/tests/test_calibration_evaluate.py` that still assumed `gmma_pipeline_scale` is T0 (it was upgraded T0→T1 by Todo 3): now use the still-T0 `tensor_core_descriptor_overhead` to exercise the "T0 present" paths.

## Key findings

1. **The plan's [0.50, 0.80] WMMA/block ratio target is arithmetically unreachable in the frozen cycle model.** Every 16×16 fragment pays `serialization + WARP_SYNC(32) + FRAG_MAC(16)` cycles. FFN_down (M=1, K=11008, N=2048) has 88,064 fragments → at ser=120 that is 168 × 88,064 ≈ 14.79M cycles vs block ≈ 394K (DMA-bound) → **ratio ≈ 0.027** (2.7%). Even `ser=0` caps at ≈0.093 because `WARP_SYNC + FRAG_MAC = 48` cycles × 88,064 fragments already exceeds block's DMA cost. Reaching [0.50, 0.80] requires warp-level parallelism (dividing fragment overhead across concurrent warps) — an engine-model change explicitly out of scope for this todo.
2. **The calibration still delivers the intended 10× improvement**: ratio 0.0027 (ser=1600) → 0.027 (ser=120); WMMA full-model tok/s @LPDDR5-51.2GB/s goes from ~0.05 → ~0.5 tok/s; per-FFN_down-GEMM from 6.9 → 67.6 tok/s. The 1600 placeholder was a 400×-vs-block cliff; 120 is the physically-sourced recommendation (Volta warp-switch ~32c + DMA descriptor ~10-20c + routing ~50c).
3. **`test_wmma_calibration_ratio()` locks the realized behavior** instead of the unreachable [0.50, 0.80] band: ratio ∈ [0.015, 0.05] (covers the full [50, 200] serialization footprint: ser=200→0.018, ser=50→0.046), plus speedup ≥ 5× vs the 1600 placeholder, plus registry cross-check (T1, source_uri, 120 in range). The deviation from the plan's band is documented in the test docstring per plan rule "Do NOT change the calibration target range without documenting why".
4. **Adding a calibration entry has downstream test surface beyond the registry** — the exact-set `test_valid_registry_has_all_parameters` requires `EXPECTED_IDS` updates, and Todo 3's T0→T1 upgrade had silently left `test_calibration_evaluate.py`'s trust-gate tests stale (they hardcoded `gmma_pipeline_scale` as the "T0 example"). Any future trust-level upgrade must grep all tests for the upgraded ID.
5. **test_engines.py / test_engine_physical_invariants.py build configs inline** — they do NOT load npu_config.yaml, so the WMMA class-constant fallback (still 1600) applies unless the test passes `wmma.fragment_serialization_cycles` explicitly. Tests exercising the calibrated default must set it in the config dict.

## Verification

- `uv run pytest sim/tests/test_engines.py -q -k wmma` → 2 passed (test_wmma_decode, test_wmma_calibration_ratio)
- `uv run pytest sim/tests/test_engine_physical_invariants.py -q` → 769 passed (all 8 engines)
- `uv run pytest sim/tests/test_calibration_registry.py sim/tests/test_calibration_config.py sim/tests/test_calibration_evaluate.py sim/tests/test_engines.py -q` → 70 passed
- Full suite `uv run pytest` → 2212 passed, 0 failed
- YAML default consumed by engine: `create_engine(npu_config.yaml + type=wmma).fragment_serialization_cycles == 120`, WMMA tok/s = 67.59 per FFN_down GEMM
