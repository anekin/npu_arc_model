# lift-decision-grade-fail — Decisions

## 2026-07-31 — Todo 3: DRAM efficiency calibration entries

1. **Registry-only T1 upgrade; code provenance untouched.** `parameters.yaml` entries for the 3 DRAM
   params are T1 with JEDEC/public-study `source_uri`s, while
   `sim/contracts/hardware.py:302-320` keeps `DEFAULT_DRAM_EFFICIENCY_RANDOM_BW_PROVENANCE` and
   `DEFAULT_RANDOM_LATENCY_PENALTY_PROVENANCE` at T0. Rationale: plan Todo 3 §5 mandates the split
   (code defaults belong to a separate contract layer; Todo 5 documents the difference). The task
   brief explicitly forbids modifying hardware.py provenance.
2. **Source URIs chosen for stability + repo consistency**:
   - `dram_efficiency` + `random_latency_penalty_cycles` -> JEDEC JESD209-5B standard page
     (`https://www.jedec.org/standards-documents/docs/jesd209-5b`), even though it is bot-blocked;
     the standard is the authoritative public origin for the timing values already cited in code.
   - `dram_efficiency_random_bw` -> Mutlu et al. arXiv:2012.03112 ("A Modern Primer on Processing
     in Memory"), a stable public DRAM access-behavior study supporting the ~50% page-hit anchor.
3. **Numeric bounds follow the plan's acceptance script exactly**: `random_latency_penalty_cycles`
   uses float bounds (30.0/60.0); ratio entries use 0.80/0.90 and 0.40/0.60. No values were changed
   from `sim/config/npu_config.yaml` (0.85 / 0.50 / 40).
4. **Entry placement**: after `dram_phy_area_12nm`, before `power_density_12nm` — groups all
   DRAM-related entries; `status: assumption` per T1 published-proxy convention (see Todo 2 note 7).
5. **EXPECTED_IDS not updated in this todo** — Todo 4 owns that (plan dependency matrix: Todo 4
   blocks on Todo 2+3). The registry test is expected to fail on the exact-set assert until then.
