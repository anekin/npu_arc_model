# Todo 14 — Cross-Node DSE Engine Ranking Matrix (Fixed)

**Date:** 2026-07-30

> **Note:** This is a re-run after fixing the process_node propagation bug
> in `_evaluate_ppa()` (runner.py:207). The previous run used 7nm area
> parameters for all nodes; this run uses the correct per-node parameters.

## 1. Summary

| Scenario | Node (nm) | Evaluated | Complete | Frontier | Winner Engine | Winner tok/s | Winner area |
|:---|:---|:---|:---|:---|:---|:---|:---|
| lpddr5_3b | 28 | 1 | 1 | — | block | 20.8 | 261.4 |
| lpddr5_3b | 22 | 1 | 1 | — | block | 20.8 | 195.4 |
| lpddr5_3b | 12 | 1 | 1 | — | block | 20.8 | 119.1 |
| lpddr5_3b | 7 | 61 | 61 | — | block | 36.6 | 99.0 |
| onchip_7b | 28 | 1 | 1 | — | block | 50.3 | 261.4 |
| onchip_7b | 22 | 1 | 1 | — | block | 50.3 | 195.4 |
| onchip_7b | 12 | 1 | 1 | — | block | 50.3 | 119.1 |
| onchip_7b | 7 | 60 | 60 | — | os_systolic | 310.9 | 99.0 |

## 2. Ranking Matrix (best tok/s per engine)

### 2.1. lpddr5_3b

**28nm** (1 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 20.8 | 261.4 | 51.3 |

**22nm** (1 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 20.8 | 195.4 | 34.3 |

**12nm** (1 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 20.8 | 119.1 | 14.4 |

**7nm** (8 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 36.6 | 99.0 | 9.5 |
| 2 | os_systolic | 31.8 | 99.0 | 9.5 |
| 3 | systolic | 22.0 | 97.0 | 8.5 |
| 4 | gmma | 20.8 | 102.0 | 11.0 |
| 5 | fsa | 20.5 | 97.2 | 8.6 |
| 6 | input_stationary | 11.1 | 99.0 | 9.5 |
| 7 | tensor_core | 9.9 | 99.0 | 9.5 |
| 8 | wmma | 0.1 | 101.0 | 10.5 |

### 2.2. onchip_7b

**28nm** (1 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 50.3 | 261.4 | 51.3 |

**22nm** (1 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 50.3 | 195.4 | 34.3 |

**12nm** (1 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | block | 50.3 | 119.1 | 14.5 |

**7nm** (8 engines):

| Rank | Engine | tok/s | area_mm² | power_w |
|:---|:---|:---|:---|:---|
| 1 | os_systolic | 310.9 | 99.0 | 9.6 |
| 2 | gmma | 203.5 | 102.0 | 11.1 |
| 3 | block | 131.4 | 107.0 | 13.6 |
| 4 | input_stationary | 108.1 | 99.0 | 9.6 |
| 5 | tensor_core | 48.2 | 99.0 | 9.6 |
| 6 | fsa | 26.4 | 97.2 | 8.7 |
| 7 | systolic | 26.4 | 97.0 | 8.6 |
| 8 | wmma | 0.1 | 101.0 | 10.6 |

## 3. Key Assumption Verification

### 3.1 Low BW (lpddr5_3b, 51.2 GB/s): Which engine wins at each node?

- **28nm:** `block` wins at 20.8 tok/s, 261.4 mm²
- **22nm:** `block` wins at 20.8 tok/s, 195.4 mm²
- **12nm:** `block` wins at 20.8 tok/s, 119.1 mm²
- **7nm:** `block` wins at 36.6 tok/s, 99.0 mm²

**Consistent across nodes:** `block` wins at all 4 nodes.

**Caveat:** 28/22/12nm each have only 1 engine (block) due to ci-all-axes sparse
coverage. Cross-node engine comparison is limited to the block engine only.

### 3.2 High BW (onchip_7b, 500 GB/s): Which engine wins at each node?

- **28nm:** `block` wins at 50.3 tok/s, 261.4 mm²
- **22nm:** `block` wins at 50.3 tok/s, 195.4 mm²
- **12nm:** `block` wins at 50.3 tok/s, 119.1 mm²
- **7nm:** `os_systolic` wins at 310.9 tok/s, 99.0 mm²

**NOT consistent across nodes:** winners vary — {28: 'block', 22: 'block', 12: 'block', 7: 'os_systolic'}

**Caveat:** 28/22/12nm only have block engine in the sparse ci-all-axes coverage.
The apparent inconsistency (block winner at 28/22/12 vs os_systolic at 7nm) is
an artifact of coverage, not a real ranking inversion.

### 3.3 Does 28nm show more extreme area differences than 7nm?

Comparing the block engine (the only engine with cross-node data):

| Scenario | Metric | 7nm | 12nm | 22nm | 28nm |
|:---|:---|:---|:---|:---|:---|
| lpddr5_3b | area_mm² | 99 | 119 | 195 | 261 |
| lpddr5_3b | vs 7nm | 1.0× | 1.2× | 2.0× | 2.6× |
| onchip_7b | area_mm² | 107 | 119 | 195 | 261 |
| onchip_7b | vs 7nm | 1.0× | 1.1× | 1.8× | 2.4× |

**Finding:** The 28nm node has ~2.6× larger area than 7nm for the same block engine.
This is dominated by logic-area scaling (PE×16) tempered by SRAM sub-quadratic
bitcell scaling. Area differences are monotonic and directionally correct.

## 4. Fix Summary

- **File:** `sim/dse/runner.py:_evaluate_ppa()` (line 205-208)
- **Root cause:** `AreaModel(base_cfg)` used base design_space.yaml config (process_node=7)
  instead of merging the design point's `area_model.process_node`
- **Fix:** Merge `point.hardware_config["area_model"]["process_node"]` into `base_cfg`
  before constructing AreaModel and PowerModel
- **Test:** 5 new tests in `test_dse_space.py::TestAreaModelProcessNodePropagation`
  verify node_scale propagation and cross-node area monotonicity

## 5. Evidence Files

- `dse-lpddr5_3b-cross-node-ci.json` — lpddr5_3b DSE results (fixed)
- `dse-onchip_7b-cross-node-ci.json` — onchip_7b DSE results (fixed)
- `task-14-engine-selection-p0-cross-node-dse.json` — structured ranking matrix
- `task-14-engine-selection-p0-cross-node-dse.md` — this report
- `task-14-engine-selection-p0-cross-node-dse-negative.txt` — negative path evidence

## 6. Methodology Notes

- DSE mode: `ci-all-axes` — one combo per axis value (sparse coverage)
- Non-7nm nodes each get 1 design point (block 128×128 default) due to ci-all-axes
- Area model: SRAM via TSMC bitcell table, logic via node_scale factor
- Process node propagation: **fixed** — AreaModel now receives correct per-node config
- Power model: scales with area via `power_density × area` approximations
- Trust level: exploratory for 28/22/12nm (geometric scaling, not silicon-calibrated)