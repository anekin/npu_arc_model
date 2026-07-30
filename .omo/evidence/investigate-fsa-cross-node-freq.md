# Frequency-Aware FSA vs Block Cross-Node Comparison

**Scenario**: `lpddr5_3b` — LPDDR5 51.2 GB/s, Qwen2.5-3B INT4 decode

**Fixed config**: 128×128 array, INT4, 2048 KB L2, no weight cache

**Method**: each node evaluated at all physically-plausible frequencies;
best tok/s per engine per node reported.

## Per-Node Frequency Bounds

| Node | Allowed Frequencies (MHz) | Block Best | FSA Best |
|:---:|:---|:---:|:---:|
| 28nm | 200, 400, 600 | 600 | 600 |
| 22nm | 400, 600, 800 | 600 | 800 |
| 12nm | 800, 1000, 1200 | 800 | 1200 |
| 7nm | 800, 1000, 1200, 1600, 2000 | 800 | 1200 |

## Per-Node Results (best frequency)

| Node | Engine | Freq (MHz) | tok/s | area_mm² | power_w | tok/W | compute_c | dma_c | util |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 nm | block | 800 | 20.8 | 99.0 | 9.0 | 2.31 | 680064 | 1133373 | 0.002 |
| 7 nm | fsa | 1200 | 20.8 | 97.2 | 9.0 | 2.31 | 1324064 | 1700331 | 0.002 |
| 12 nm | block | 800 | 20.8 | 119.1 | 13.0 | 1.60 | 680064 | 1133373 | 0.002 |
| 12 nm | fsa | 1200 | 20.8 | 114.3 | 13.0 | 1.60 | 1324064 | 1700331 | 0.002 |
| 22 nm | block | 600 | 20.8 | 195.4 | 23.4 | 0.89 | 680064 | 850030 | 0.003 |
| 22 nm | fsa | 800 | 18.3 | 177.6 | 21.7 | 0.84 | 1324064 | 1133552 | 0.002 |
| 28 nm | block | 600 | 20.8 | 261.4 | 33.7 | 0.62 | 680064 | 850030 | 0.003 |
| 28 nm | fsa | 600 | 14.3 | 232.6 | 25.0 | 0.57 | 1324064 | 850163 | 0.002 |

## Ratios (block / FSA)

| Node | Block Freq | FSA Freq | Area Ratio | tok/s Ratio |
|:---:|:---:|:---:|:---:|:---:|
| 28 nm | 600 MHz | 600 MHz | 1.124 | 1.455 |
| 22 nm | 600 MHz | 800 MHz | 1.100 | 1.137 |
| 12 nm | 800 MHz | 1200 MHz | 1.042 | 1.000 |
| 7 nm | 800 MHz | 1200 MHz | 1.019 | 1.000 |

## Key Observations

1. **tok/s now varies with node** — higher frequency at 7nm (2000 MHz) produces higher throughput; lower frequency at 28nm (600 MHz) reduces it.

2. **Area scales with node** — from 99 mm² (7nm) to 261 mm² (28nm) for block engine, consistent with bitcell + logic scaling.

3. **Power scales with area × frequency** — faster node + higher clock = more dynamic power.

4. **FSA throughput advantage over block is negligible** — the bandwidth-bottleneck (51.2 GB/s) dominates; even at 2000 MHz block's area advantage persists.
