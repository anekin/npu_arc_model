# Issues — arc-model-scenario-driven-dse-development

## Todo 3: Red Manifest — Physical Invariant Failures

### Red suite results
- **45 failed, 1139 passed** (exit code 1 = expected red phase)
- **Invalid-input tests**: 424/424 passed (all engines survive invalid inputs)
- **Physical invariants**: 715/760 passed; 45 failures are genuine formula violations

### Failure → Todo mapping

| Failure | Count | Mapped To | Severity | Notes |
|:---|:---|:---|:---|:---|
| SYSTOLIC M=2→3 latency decrease | 3 tests | **Todo 5** | High | `estimate()` dispatches M≤2→decode, M>2→prefill; formulas are discontinuous |
| OS M scaling (same compute for all M) | 1 test | **Todo 5** | High | `per_tile_compute` uses only `self.H` not M |
| InputStationary M monotonicity | 3 tests | **Todo 5** | High | `reuse_factor` transition at M=H creates drop |
| GMMA pipeline undercuts MAC floor | 6 tests | **Todo 5** | High | `pipeline_scale=0.05` × (H+M+W) < ceil(macs/peak) |
| GMMA DMA floor violation | 7 tests | **Todo 5/6** | High | `total_cycles = max(compute, total_dma)` uses non-ceil DMA |
| OS DMA floor violation | 2 tests | **Todo 5/6** | Medium | DMA not ceil'd; under 819.2 GB/s raw floor |
| Bandwidth saturation (all 8 engines) | 22 tests | **Todo 6** | High | BW increase doesn't monotonically approach compute floor |
| FSA mac_count = op_count | 4 tests | **Todo 4** | Medium | `result.ops` stores `M×K×N×2` instead of `M×K×N` |
| FSA weight_bytes=0 | 3 tests | **Todo 5** | Low | SRAM caching zeroes reported weight_bytes |
| WMMA diagnostics incomplete | 2 tests | **Todo 4** | Low | Direct path missing detail fields |

### Known non-bugs (test infrastructure)
- Diagnostics cache-pair relaxed: cache-pair paths have different detail key sets than direct estimate
- Invalid inputs don't always raise: some engines silently accept 0/negative (non-crash criterion met)
- `sim/tests/__init__.py` needed: created to enable `tests.oracles.physics` package imports

### Action items blocking on this todo
- Todo 4: engine registry/result contract must enforce ops semantics
- Todo 5: formula repairs for Systolic, OS, GMMA, TensorCore
- Todo 6: frequency/bandwidth unit propagation must fix BW saturation across all engines

## Todo 5 Completion Status (2026-07-30)

### Resolved Failures (from Todo 3 Red Manifest)

| Failure Category | Status | Resolution |
|:---|:---|:---|
| SYSTOLIC M=2→3 latency decrease | ✅ FIXED | Unified M-tiling formula, no decode/prefill dispatch |
| OS M scaling (same compute for all M) | ✅ FIXED | Added M-tiling to per_tile_compute |
| InputStationary M monotonicity | ✅ FIXED | Removed reuse_factor artifact |
| GMMA pipeline undercuts MAC floor | ✅ FIXED | Added ideal floor to max() guard |
| GMMA DMA floor violation | ✅ FIXED | raw DMA uses `ceil` and `self.bw_raw` |
| OS DMA floor violation | ✅ FIXED | raw DMA uses `ceil` and `self.bw_raw` |
| FSA mac_count = op_count | ✅ FIXED | Already fixed in Todo 4 |
| FSA weight_bytes=0 | ✅ FIXED | Report total weight bytes, not effective=0 |
| WMMA diagnostics incomplete | ✅ FIXED | Already fixed in Todo 4 |
| TensorCore partial M tile | ✅ FIXED | Last wave uses actual effective dimensions |

### Remaining: Bandwidth Saturation (16 tests, Todo 6)
All 8 engines fail bandwidth saturation at 819.2 GB/s because pipeline overheads (broadcast_sync, accumulate, etc.) are constant and don't scale down with BW increase. This requires Todo 6's frequency/bandwidth unit propagation fix.
