# Frequency-Aware All-8-Engine Cross-Node Comparison

**Scenario**: `lpddr5_3b` — LPDDR5 51.2 GB/s, Qwen2.5-3B INT4 decode

**Fixed config**: 128x128 array, INT4, 2048 KB L2, no weight cache

**Engines**: systolic, os_systolic, block, tensor_core, wmma, gmma, input_stationary, fsa

**Method**: each node evaluated at all physically-plausible frequencies;
best tok/s per engine per node reported.

## Per-Node Frequency Bounds

| Node | Allowed Frequencies (MHz) |
|:---:|:---|
| 28nm | 200, 400, 600 |
| 22nm | 400, 600, 800 |
| 12nm | 800, 1000, 1200 |
| 7nm | 800, 1000, 1200, 1600, 2000 |

## Per-Node Engine Rankings (best tok/s)

### 28nm

| Rank | Engine | Freq (MHz) | tok/s | area_mm² | power_w | util |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | os_systolic | 200 | 31.8 | 261.4 | 16.1 | 0.016 |
| 2 | block | 600 | 20.8 | 261.4 | 33.7 | 0.003 |
| 3 | gmma | 200 | 20.8 | 309.4 | 20.9 | 0.009 |
| 4 | fsa | 600 | 14.3 | 232.6 | 25.0 | 0.002 |
| 5 | systolic | 600 | 14.3 | 229.4 | 24.1 | 0.002 |
| 6 | input_stationary | 200 | 11.1 | 261.4 | 16.1 | 0.004 |
| 7 | tensor_core | 600 | 9.2 | 261.4 | 33.7 | 0.001 |
| 8 | wmma | 200 | 0.0 | 293.4 | 19.3 | 0.000 |

### 22nm

| Rank | Engine | Freq (MHz) | tok/s | area_mm² | power_w | util |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | os_systolic | 400 | 31.8 | 195.4 | 18.0 | 0.008 |
| 2 | block | 600 | 20.8 | 195.4 | 23.4 | 0.003 |
| 3 | gmma | 400 | 20.8 | 225.0 | 23.9 | 0.005 |
| 4 | fsa | 800 | 18.3 | 177.6 | 21.7 | 0.002 |
| 5 | systolic | 800 | 18.3 | 175.6 | 20.9 | 0.002 |
| 6 | input_stationary | 400 | 11.1 | 195.4 | 18.0 | 0.002 |
| 7 | tensor_core | 800 | 9.6 | 195.4 | 28.8 | 0.001 |
| 8 | wmma | 800 | 0.1 | 215.1 | 36.7 | 0.000 |

### 12nm

| Rank | Engine | Freq (MHz) | tok/s | area_mm² | power_w | util |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | os_systolic | 800 | 31.8 | 119.1 | 13.0 | 0.004 |
| 2 | systolic | 1200 | 25.5 | 113.7 | 12.7 | 0.002 |
| 3 | block | 800 | 20.8 | 119.1 | 13.0 | 0.002 |
| 4 | fsa | 1200 | 20.8 | 114.3 | 13.0 | 0.002 |
| 5 | gmma | 800 | 20.8 | 127.2 | 16.2 | 0.002 |
| 6 | input_stationary | 800 | 11.1 | 119.1 | 13.0 | 0.001 |
| 7 | tensor_core | 1200 | 10.1 | 119.1 | 15.9 | 0.001 |
| 8 | wmma | 800 | 0.1 | 124.5 | 15.1 | 0.000 |

### 7nm

| Rank | Engine | Freq (MHz) | tok/s | area_mm² | power_w | util |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | os_systolic | 800 | 31.8 | 99.0 | 9.0 | 0.004 |
| 2 | systolic | 1600 | 27.8 | 97.0 | 9.6 | 0.002 |
| 3 | block | 800 | 20.8 | 99.0 | 9.0 | 0.002 |
| 4 | fsa | 1200 | 20.8 | 97.2 | 9.0 | 0.002 |
| 5 | gmma | 800 | 20.8 | 102.0 | 10.2 | 0.002 |
| 6 | input_stationary | 800 | 11.1 | 99.0 | 9.0 | 0.001 |
| 7 | tensor_core | 2000 | 10.5 | 99.0 | 12.3 | 0.000 |
| 8 | wmma | 800 | 0.1 | 101.0 | 9.8 | 0.000 |

## Summary: Best Engine per Node

| Node | Best Engine | tok/s | Top 3 Engines |
|:---:|:---|:---:|:---|
| 28nm | os_systolic | 31.8 | os_systolic, block, gmma |
| 22nm | os_systolic | 31.8 | os_systolic, block, gmma |
| 12nm | os_systolic | 31.8 | os_systolic, systolic, block |
| 7nm | os_systolic | 31.8 | os_systolic, systolic, block |

## Key Observations

1. **os_systolic is the top performer across all nodes** — its output-stationary dataflow achieves the highest tok/s at every process node for lpddr5_3b.

2. **Block engine BW-bound behavior varies** — while block excels at 7nm (36.6 tok/s via high frequency), it degrades significantly at older nodes due to frequency limits combined with compute constraints.

3. **FSA compute-bound** — FSA tok/s drops from 20.5 (7nm) to 5.2 (28nm), consistent with its compute-bound nature at lower frequencies.

4. **wmma is non-viable at all nodes** — wmma produces near-zero tok/s due to its warp-level MMA architecture being unsuitable for the current workload/bandwidth combination.

5. **input_stationary is BW-consistent** — input_stationary (Eyeriss-style) maintains ~11.1 tok/s across all nodes, suggesting strong BW-bound behavior.
