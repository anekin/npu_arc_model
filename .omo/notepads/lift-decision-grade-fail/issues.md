# lift-decision-grade-fail — Issues

## 2026-07-31 — OPEN (Todo 4 dependency, expected)

**`sim/tests/test_calibration_registry.py::test_valid_registry_has_all_parameters` FAILS** after
adding the 7 new calibration entries from Todos 1-3. Missing from EXPECTED_IDS:
`max_freq_7nm`, `max_freq_28nm`, `max_freq_12nm`, `max_freq_22nm`,
`dram_efficiency`, `dram_efficiency_random_bw`, `random_latency_penalty_cycles`.

- Plan scopes this fix to Todo 4. Not caused by Todo 1.
- All Todo 1 tests pass independently (18/18 in test_calibration_evaluate.py).

## 2026-07-31 — INFO (not a bug)

JEDEC standard page `https://www.jedec.org/standards-documents/docs/jesd209-5b` returns 403 to
automated fetch (bot protection, same class as tsmc.com). Used as the `source_uri` anchor for
`dram_efficiency` + `random_latency_penalty_cycles` anyway because it is the canonical public
standard page; tRFCpb=140ns/tREFI=3900ns and tRC~48ns values are cross-checked against
`sim/contracts/hardware.py:173` and `sim/models/dram.py:39-42` (repo-consistent).

## 2026-07-31 — RESOLVED

**Plan's RK3588 12nm product example is inaccurate** (RK3588 = Samsung 8nm). Mitigated by using
MediaTek Helio G90/G90T (TSMC 12nm) as the public 12nm anchor. Note in case future todos reuse the
plan's example verbatim.

## 2026-07-31 — INFO (not a bug)

`calibration_range` bounds for 12nm/7nm share the same lower bound (800 MHz); the 12nm entry's
range `[800, 1200]` is a strict subset of 7nm's `[800, 2000]`. This is intentional and matches
`dse_axes.yaml` — do not "fix" the overlap.
