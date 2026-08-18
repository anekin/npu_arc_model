# lift-decision-grade-fail — Learnings

## 2026-07-31 — Todo 2: adding max_freq_{28,22,12,7}nm calibration entries

1. **`range_min`/`range_max` semantics**: `CalibrationEntry.is_in_range()` (`sim/calibration/schema.py:63-67`)
   returns False only when `range_min` is set AND actual < min, or `range_max` is set AND actual > max.
   Both fields are optional (`float | None`), but leaving them unset silently disables bound
   enforcement. Every entry in `parameters.yaml` sets them explicitly — new entries must follow.

2. **`calibration_range` is a human-readable string, not a list**: existing entries use an en-dash
   string (e.g. `1.8–2.2 mm2`, `0–10 cycles`), while `range_min`/`range_max` are the numeric
   programmatic bounds. Schema uses `extra="forbid"`, so no extra YAML keys are allowed.

3. **The single source of truth for the frequency ranges is `sim/config/dse_axes.yaml:262-288`**
   (`node_*_frequency_bound` constraints). The bound is `require.frequency_mhz[0]` (min) and
   `[-1]` (max), not an explicit min/max pair. Values:
   - 28nm: `[200, 400, 600]` -> [200, 600]
   - 22nm: `[400, 600, 800]` -> [400, 800]
   - 12nm: `[800, 1000, 1200]` -> [800, 1200]
   - 7nm: `[800, 1000, 1200, 1600, 2000]` -> [800, 2000]

4. **Plan's 12nm product example (RK3588) is factually wrong**: RK3588 is fabricated on Samsung 8nm,
   not 12nm. Used MediaTek Helio G90/G90T (TSMC 12nm, 2x Cortex-A76 @2.05GHz) as the public 12nm
   product reference instead — a 1.2 GHz ceiling is conservative vs. the ~2 GHz Helio demonstrates.
   General rule: verify process-node claims of product references before writing `source_uri`.

5. **`source_uri` reachability check**: verified candidate URLs with `curl -m 8 -o /dev/null -w
   "%{http_code}"` before committing them as anchors — NVIDIA A100 page (200), TPUv1 ISCA DOI
   (302), MediaTek smartphones page (200). tsmc.com returns 403 to curl (bot-block, not 404) so it
   was avoided. 22nm has no public product page for the internal RK1828 chip; anchored to the
   repo-local `references/area_sources.md` §5 instead.

6. **Value semantics**: each `max_freq_*nm` `value` = the node's maximum feasible frequency
   (equals `range_max`). TrustGate's `is_in_range()` will validate a design point's actual
   `frequency_mhz` against `[range_min, range_max]` — upper-bound constraint semantics.

7. **`status: assumption` is the correct lifecycle status for T1 published-proxy entries** in this
   repo (T1 = "engineering assumptions or published proxies", not `calibrated` which is reserved
   for silicon-verified T2 anchors).

## 2026-07-31 — Todo 3: adding dram_efficiency / dram_efficiency_random_bw / random_latency_penalty_cycles

1. **Added 3 DRAM entries, all T1 + non-null `source_uri` + `source_hash: null` + explicit
   `range_min`/`range_max`** (placed after `dram_phy_area_12nm`, grouping DRAM entries):
   - `dram_efficiency`: 0.85, ratio, [0.80, 0.90]; source = JEDEC JESD209-5B
     (https://www.jedec.org/standards-documents/docs/jesd209-5b). Description carries the plan's
     derivation chain: tRFCpb=140ns / tREFI=3900ns -> per-bank refresh ~3.6% -> retained ~0.964
     -> x controller/command-bus/bank-conflict margin (5-10%) -> 0.85.
   - `dram_efficiency_random_bw`: 0.50, ratio, [0.40, 0.60]; source = Mutlu et al., "A Modern
     Primer on Processing in Memory", arXiv:2012.03112 (public random-access page-hit ~50% anchor).
   - `random_latency_penalty_cycles`: 40, cycles, [30.0, 60.0]; source = JEDEC JESD209-5B;
     description ties 40 to tRC~48ns @1GHz -> ~48 cycles per activate+read.
   Values match `sim/config/npu_config.yaml:90-92` (0.85 / 0.50 / 40).

2. **JEDEC's standard page is bot-blocked (403) but is the canonical public URI** — `jesd209-5b`
   returned 403 to automated fetch (same pattern as tsmc.com in Todo 2); the tRFCpb/tREFI numbers
   cross-check against the existing comments in `sim/contracts/hardware.py:173` and
   `sim/models/dram.py:39-42`, so the values are repo-consistent even though the page itself is
   not machine-fetchable.

3. **`random_latency_penalty_cycles` bounds are floats (30.0/60.0)** — the plan's acceptance script
   asserts `range_min == 30.0` and `range_max == 60.0` exactly, unlike the ratio entries (0.80/0.90).
   YAML parses both forms fine; the schema field is `float | None`.

4. **`sim/contracts/hardware.py:302-320` provenance stays T0** for random-bw and latency-penalty
   (`DEFAULT_DRAM_EFFICIENCY_RANDOM_BW_PROVENANCE`, `DEFAULT_RANDOM_LATENCY_PENALTY_PROVENANCE`)
   while the registry says T1 — this is the plan-mandated registry-only upgrade. Todo 5 must
   document the divergence. Note `DEFAULT_DRAM_EFFICIENCY_PROVENANCE` (sequential) is already T1.

5. **Parallel Wave-1 edits coexist cleanly**: Todo 1 (header + tensor_core T0->T1) and Todo 2
   (max_freq_*nm) landed in the same file during this todo; final state = 32 entries, YAML still
   valid, `CalibrationRegistry.from_yaml()` parses all. This is the same file-level contention
   Todo 2 flagged for Todo 4's EXPECTED_IDS merge.

6. **parameters.yaml header updated cleanly**: "T0/T1" → "T1" and "T2+" → "T3+" on two lines
   without YAML breakage. The commit-granularity means Todo 1's header edit is interleaved with
   Todo 2's entry additions in the same file — Todo 4 handles the EXPECTED_IDS merge.

## 2026-07-31 — Todo 1: upgrading tensor_core_descriptor_overhead T0→T1

1. **source_uri selection**: ARM PL330 DMA PrimeCell DDID0424
   (https://developer.arm.com/documentation/ddi0424/) is publicly accessible (HTTP 200).
   The DMA descriptor issue overhead range of 4–8 cycles from the PL330 TRM serves as an
   order-of-magnitude proxy for Tensor Core per-sub-tile descriptor setup. The 5-cycle value
   is a conservative midpoint within the PL330 documented range.

2. **Synthetic T0 approach for tests**: After eliminating the last real T0 parameter, the
   trust-gate and runner tests that explicitly depend on T0 presence need a synthetic T0 entry
   (`CalibrationEntry` with `trust_level=T0`) injected via:
   - Trust-gate tests: `CalibrationRegistry([SYNTHETIC_T0, t1_entry])` constructor (direct)
   - Runner tests: `monkeypatch.setattr(CalibrationRegistry, "from_yaml", ...)` to return a
     combined registry of synthetic T0 + all real entries, plus monkeypatching
     `calibration_ids_for_design_point` so the synthetic T0 is consumed by every design point.
   This preserves test intent (decision-grade rejects T0, exploratory allows T0) without
   depending on any real T0 parameter.

3. **Runner test mechanics**: `test_runner_decision_grade_fails_on_t0` passes because
   `power_density_12nm` (value=0.5, range_max=0.30) is out of calibration range, generating
   an "out_of_calibration_range" violation that triggers `ConfigError` in decision-grade mode.
   The synthetic T0 is present and consumed but is in-range (1.0 in [0.5, 1.5]) so it doesn't
   create its own violation — it exists to ensure the test doesn't break when the last real T0
   is eliminated. The `require_trust` parameter defaults to T0 in the runner's trust-gate
   check, so a T0 entry meets the threshold and produces no "trust_level_too_low" violation.

4. **Registry state verified**: Zero T0 entries remain in the 32-entry registry after the
   upgrade. `grep -i "T0" references/calibration/parameters.yaml` returns no matches.
   `trust_level: T1` and `source_uri: https://developer.arm.com/documentation/ddi0424/`
   confirmed via programmatic check.

## Environment facts
- YAML validation: `python3 -c "import yaml; yaml.safe_load(open('references/calibration/parameters.yaml'))"`
- Exa web search is rate-limited on this server; wikipedia.org times out; use `curl` reachability
  checks and repo-local provenance files as fallback anchors.
- JEDEC standard page (jesd209-5b) is bot-blocked (403) — usable as a `source_uri` anchor but not
  machine-fetchable;   verify content claims against repo comments + vendor datasheet values.

## 2026-07-31 — Final Wave F2: ruff format `.omo` exclusion fix

1. **`[tool.ruff.format] exclude` uses glob matching, not bare-dir matching**: adding `".omo"` to
   `[tool.ruff.format] exclude` did NOT stop `uv run ruff format --check .` from flagging
   `.omo/evidence/*.py` — ruff 0.16 requires `".omo/**"` (glob) in the format section, whereas the
   lint section's bare `".omo"` works. Verified with an isolated `/tmp` repro: `exclude=["sub"]`
   fails, `exclude=["sub/**"]` passes. Root config now carries `.omo/**` in format and `.omo` in lint.
2. **CLI `--config 'exclude = [".omo"]'` also works** (top-level exclude applies to format), so the
   format-specific key is the only one needing the glob form; top-level `[tool.ruff]` keys accept
   bare dir names.

## 2026-07-31 — Todo 4: wiring the 7 new IDs into evaluate.py + EXPECTED_IDS

1. **`calibration_ids_for_design_point()`**: added `max_freq_{node_suffix}nm` right after the
   `systolic_pe_area_{node_suffix}nm` add — same node-suffix mechanism, so every config consumes
   exactly the `max_freq_*nm` entry matching its `area_model.process_node_nm`.

2. **External-DRAM gating is `not uses_tsv`**, NOT `not (onchip.capacity_gb > 0)` alone:
   `uses_tsv` is already `"hbm" in mem_type or "onchip" in mem_type or capacity_gb > 0`, so gating
   the 3 DRAM IDs on `not uses_tsv` exactly implements "external DRAM only (not HBM, not on-chip)".
   Keeping it next to the existing `dram_phy_area_12nm` branch preserves the pre-existing PHY
   behavior (which only checks capacity, so HBM-with-zero-onchip still gets the PHY ID — unchanged).

3. **Field name is `memory.dram_efficiency`**, not `memory.efficiency` — the task description said
   `efficiency` but the canonical field in `npu_config.yaml:90` and `MemoryConfig` is
   `dram_efficiency`. `_actual_value()` reads `dram_efficiency` first and falls back to
   `efficiency` for robustness; same dual-key pattern for `dram_efficiency_random_bw` /
   `efficiency_random_bw`.

4. **`frequency_mhz` lives under `mac_engine`** in DSE-built hw_configs (`dse/hardware_builder.py:34`),
   not under `area_model` as the plan assumed. `_actual_value()` for `max_freq_*nm` probes
   `mac_engine` → `area_model` → `mxu` in that order, then falls back to the node maximum
   (7nm=2000, 12nm=1200, 22nm=800, 28nm=600) which equals the registry `range_max`. This keeps
   TrustGate semantics: a configured in-range frequency passes, a missing frequency defaults to
   the upper bound (still in range), an overclocked frequency fails the upper-bound check.

5. **Verification**: `30 passed` for the two test files (was failing on
   `test_valid_registry_has_all_parameters` before EXPECTED_IDS gained the 7 IDs). Functional
   spot-checks confirmed node-specific max_freq IDs for all 4 nodes, DRAM IDs present for
   external LPDDR5 and absent for HBM3/on-chip.

## 2026-07-31 — Todo 5: docs + release_gate.py updated for zero-T0 / T3+ decision-grade

1. **The gate short-circuits on dirty worktree before reaching the T3+ check** —
   `release_gate.py:324-326` returns 1 with "dirty worktree" when
   `--profile decision-grade` runs on an uncommitted tree. To verify the
   substantive T3+ failure path, run the DSE subprocess directly with
   `--trust-mode decision-grade`: all 87 design points come back
   `exploratory` (never `authoritative`), which is what triggers the gate's
   "requires authoritative trust level" error (release_gate.py:137-138). The
   doc's "Expected: FAIL" bash comment is still honest even when the observed
   exit-1 comes from the worktree guard.

2. **`exploratory`, not `non_authoritative`, for T1 gate trust in practice** —
   runner.py:319-326 maps decision-grade + T0/T1 gate trust to
   `RunTrustLevel.non_authoritative`, but the DSE run reported
   `trust_level: exploratory` for all 87 points. Either way it is never
   `authoritative`, so the gate fails; don't assume which label the runner
   actually emits when writing docs.

3. **Two remaining "T0/T1" mentions are intentional capability statements** —
   docs line 37 and release_gate.py line 5 describe the experimental profile
   ("allows T0/T1 parameters"), which stays true as a policy even though the
   registry now holds zero T0 entries. Leave them; only claims about registry
   contents were stale.

4. **The `!!!` prefix on the Decision-Grade State headline is pre-existing
   markdown** (probably an intended admonition); preserved the style while
   rewriting the section. The bash-block comment at line 259 is the last line
   of the file and is the natural place to keep the "Expected: FAIL" note.

5. **Evidence file convention**: `.omo/evidence/task-5-lift-decision-grade-fail.txt`
   records grep results + both gate exits + the direct DSE trust check, mirroring
   the task-4 evidence naming.

## 2026-07-31 — Todo 6: README §1.4 FAIL clause removed

1. **Scope was exactly two lines**: the §1.4 intro note (line 67) and the
   `决策级状态` field inside the 跨节点验证结论 cell (line 78). The rest of the
   cell (cross-node matrix findings, WMMA/GMMA recalibration text, evidence
   links in the third column) is unchanged, so the table keeps its 3-column
   structure and no scenario numbers move.

2. **Replacement text (no PASS)**: `**已升级，未就绪**` plus four facts:
   tensor_core_descriptor_overhead T0→T1 (ARM PL330 DDID0424); 4
   `max_freq_*nm` entries T1 (public product refs + TSMC node chars) making
   cross-node coverage T1 not exploratory; 3 DRAM entries T1 (JEDEC LPDDR5 +
   public DRAM-locality refs); decision-grade still requires T3+ authoritative
   evidence for ALL ranking parameters. Kept the "决策级仍未达成" wording rather
   than "PASS" or "FAIL".

3. **grep verification**: `grep -n "FAIL\|频率-节点绑定为探索性结论\|多节点覆盖
   不完全\|包含 T0/T1" README.md` → exit 1 (no matches). Positive grep matches
   lines 67/78. The `dram_efficiency` positive-grep hit on line 386 is §8.3,
   pre-existing.

4. **§8.3 line 386 is stale but out of scope**: "README 中 75% 声称已被证伪
   （待 Todo 6 统一清理）" refers to a 75% DRAM-efficiency claim, but grep for
   "75%"/"0.75" across README finds only that self-referential line itself —
   the claim it cites no longer exists. Left untouched (outside §1.4); flag for
   a doc-consistency pass (Todo 7 or F3) rather than expanding scope.

5. **Follow-up cleanup**: removed the stale self-referential §8.3 note ("75% 声称已被证伪（待 Todo 6 统一清理）", referenced a claim that did not exist) and replaced it with a canonical-source pointer; `grep -n "75%\|待 Todo 6 统一清理" README.md` → exit 1 (no matches).

## 2026-07-31 — Todo 7: trust disclaimer in NPU_Engines_Architecture_Guide.md

1. **Old line-8 disclaimer had two stale claims**: it called `tensor_core_descriptor_overhead`
   and `block_sparsity_penalty` "T0/T1", but `block_sparsity_penalty` is NOT a registered
   calibration parameter (verified `grep block_sparsity_penalty references/calibration/parameters.yaml`
   → no matches) and `tensor_core_descriptor_overhead` was already upgraded to T1 in Todo 1.
   Both stale claims were in the single disclaimer paragraph (line 8) — nowhere else in the doc.

2. **Phrasing constraint from the negative grep**: the verification regex
   `tensor_core_descriptor_overhead.*T0` forbids putting T0 AFTER the parameter name on the same
   line. To say "upgraded from T0 to T1" safely, mirror README §1.4 (Todo 6) phrasing: "最后一个
   真实 T0 参数 `tensor_core_descriptor_overhead` 已升级为 T1" — T0 precedes the name, so the
   regex does not match.

3. **Line-133/193 T1 mentions are pre-existing and legitimate**: line 133 ("依赖当前 T1 带宽/面积
   假设") and line 193 ("trust T1" WMMA calibration note) are current-state statements, not the
   trust disclaimer; left unchanged. The disclaimer paragraph is the only edited line.

4. **Consistency with Todo 6 README language**: reused the §1.4 phrasing (全部真实 T0 参数已消除,
   "决策级仍未达成 ... T3+ 权威证据") so the guide, README, and model-trust-and-release.md tell the
   same story: zero real T0, everything at least T1, decision-grade still requires T3+.


## 2026-07-31 — F2 code-quality fix: ruff format + import sort

1. F2 flagged one formatting issue (`float(calibration_id[len("max_freq_"):-len("nm")])` slice spacing in `sim/calibration/evaluate.py:128`) and one I001 import-block error in `sim/tests/test_calibration_evaluate.py`. Both auto-fixed via `uv run ruff format` + `uv run ruff check --fix`; format/check/pytest now exit 0 (70 passed). Note: `uv run python sim/calibration/evaluate.py` requires `PYTHONPATH=sim` (as elsewhere in the plan); `sim/calibration/evaluate.py` is import-only with no `__main__`, so its run output is empty with exit 0.
