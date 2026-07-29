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
