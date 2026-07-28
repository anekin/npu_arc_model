# Task F2: Code Quality Review — Final Verdict

**Date:** 2026-07-28  
**VERDICT: APPROVE**

## Summary

All five target files pass code quality review. No duplicated formulas, unused calibration constants, silent exceptions, magic tuning, or TODO/FIXME/HACK/xxx markers found. Compileall clean, all 51 focused tests pass.

## Per-File Checklist

### ✅ `sim/engine/systolic_engine.py` — Formula Ownership

| Check | Result |
|-------|--------|
| Owns its own formulas (no shared helper) | PASS — no `systolic_timing.py` exists anywhere in repo |
| No MXUModel delegation | PASS — imports only from `engine.mac_engine` (ABC) |
| Decode formula matches MXUModel v2 | PASS — `per_tile_compute = self.H * (M + 1) + self.W` (line 32) |
| Prefill conditional drain | PASS — `M` for partial single M-tile, `self.H` otherwise (lines 86-94) |
| `ValueError` guard for `M <= 0` | PASS — `estimate()` line 142 |
| Rounding correctness | PASS — `round()` on DMA floats, `int()` on total, integer math for compute |

### ✅ `sim/engine/os_systolic_engine.py` — K-Reduction Depth + DMA Aggregation

| Check | Result |
|-------|--------|
| K-reduction depth added | PASS — `per_tile_compute = self.H + BROADCAST_SYNC_CYCLES + _accumulate_cycles(...)` (line 66) |
| DMA aggregation matches BlockEngine | PASS — `total_weight_bytes = K * N * w_bits // 8` with `_dram_eff_for_bytes` (lines 53-63) |
| Max-model timing | PASS — `total_cycles = max(int(total_compute_cycles), raw_dma_cycles)` (line 72) |
| Detail keys present | PASS — `k_reduction_cycles`, `raw_dma_cycles`, `total_compute_cycles`, `bottleneck_reason` (lines 104-108) |
| Pair path also fixed | PASS — same compute fix and aggregated DMA in `estimate_weight_cache_pair` (lines 112-189) |

### ✅ `sim/engine/tensor_core_engine.py` — Descriptor Overhead Per-Wave

| Check | Result |
|-------|--------|
| Config reads `dma.descriptor_overhead_cycles` | PASS — `_parse_config` lines 38-52, defaults to 5 |
| Validation: non-negative integer | PASS — rejects non-integer, negative, and bool (lines 44-51) |
| Applied per-wave | PASS — `descriptor_cycles_per_wave = num_tcs * self.descriptor_overhead_cycles` (line 95) |
| Partial last wave handled | PASS — `active_tcs = invocations_last`, `last_wave_descriptor` (lines 97-102) |
| Three-way branching for waves | PASS — waves==1, all full, partial last (lines 109-122) |
| Detail keys present | PASS — `active_tcs`, `num_waves`, `per_wave_payload_cycles`, `descriptor_cycles_per_wave`, `total_descriptor_cycles` (lines 148-155) |

### ✅ `sim/engine/gmma_engine.py` — Pipeline Scale + Raw-DMA Floor

| Check | Result |
|-------|--------|
| `pipeline_scale` used in compute | PASS — `_per_tile_compute(M)` uses `max(1, math.ceil((self.H + M + self.W) * self.pipeline_scale))` (line 79) |
| Config validation | PASS — `_parse_config` validates `0 < scale <= 1` (lines 55-71) |
| Raw DMA floor enforced | PASS — `total_cycles = max(total_compute, total_dma)` (line 111), not TMA-discounted |
| TMA overlap diagnostic-only | PASS — `tma_hidden_dma`, `tma_exposed_dma` only in details (lines 106-107, 133-134) |
| Detail keys present | PASS — `raw_dma_cycles`, `tma_hidden_dma`, `tma_exposed_dma`, `pipeline_scale` (lines 132-137) |
| `total_cycles >= raw_dma_cycles` | PASS — enforced by `max(total_compute, total_dma)` |

### ✅ `sim/design_space_explorer.py` — Structured Error Handling

| Check | Result |
|-------|--------|
| Structured error counts | PASS — `generated`, `evaluated`, `filtered_by_area`, `errors`, `error_details` (lines 605-609) |
| Errors printed to stderr | PASS — `print(..., file=sys.stderr)` with engine type, dims, memory mode (lines 629-632) |
| Non-zero exit on errors | PASS — `sys.exit(1)` when `errors > 0` and no `--allow-partial` (lines 650-656) |
| `--allow-partial` flag | PASS — CLI argument (line 545), skips the error exit (line 650) |
| Empty results exit | PASS — `sys.exit(1)` for `evaluated==0` (line 649) and empty results after filtering (line 658) |
| JSON metadata includes counts | PASS — LLM mode: top-level `generated`/`evaluated`/etc (lines 754-760, 784); CV mode: wrapped in metadata (lines 770-774) |

## Cross-Cutting Checks

| Check | Result |
|-------|--------|
| No TODO/FIXME/HACK/xxx markers in `sim/` | PASS — `grep` returned no matches |
| No shared `systolic_timing.py` helper | PASS — file does not exist, no references in codebase |
| No duplicated formulas across engines | PASS — each engine has its own compute model (systolic uses WS pipeline, OS uses broadcast+reduction, GMMA uses scaled pipeline, TensorCore uses sub-tile waves) |
| No unused calibration constants | PASS — `SUBTILE_PIPELINE_FILL`, `SUBTILE_OVERHEAD_CYCLES`, `GMMA_PIPELINE_SCALE`, `TMA_OVERLAP`, `BROADCAST_SYNC_CYCLES`, `DEFAULT_DESCRIPTOR_OVERHEAD_CYCLES` all used in active code paths |
| No silent exceptions | PASS — DSE now prints each error to stderr; engine constructors raise `ValueError` for invalid config |
| No magic tuning numbers | PASS — all constants are named class-level attributes (`GMMA_PIPELINE_SCALE = 0.05`, etc.) with config override paths; `GMMA_PIPELINE_SCALE` explicitly marked uncalibrated in learnings |

## Verification Evidence

```bash
# Compile check
$ python -m compileall -q sim/engine sim/design_space_explorer.py sim/tests
(no output = all compiled clean)

# Focused tests
$ python -m pytest sim/tests/test_engines.py sim/tests/test_engine_result_contract.py \
    sim/tests/test_calibration_config.py sim/tests/test_dse_strict.py -q
...................................................   [100%] (51 passed)

# No markers
$ grep -r 'TODO\|FIXME\|HACK\|xxx\|XXX' sim/ --include='*.py'
(no matches)
```
