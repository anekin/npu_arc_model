# Task 12 — 5-Scenario DSE Engine Ranking Matrix

Ranking per scenario is by the best `tok_per_s` achieved by each engine.

| Scenario | Bandwidth (GB/s) | Rank | Engine | tok/s | area_mm2 | on_pareto | config_label |
|---|---:|---:|---|---:|---:|---|---|
| lpddr5_3b | 51.2 | 1 | block | 36.6 | 99.0 | No | `bloc 128×128 INT2 1000MHz  ` |
| lpddr5_3b | 51.2 | 2 | os_systolic | 31.8 | 99.0 | No | `os_s 128×128 INT4 1000MHz  ` |
| lpddr5_3b | 51.2 | 3 | systolic | 22.0 | 97.0 | No | `syst 128×128 INT4 1000MHz  ` |
| lpddr5_3b | 51.2 | 4 | gmma | 20.8 | 102.0 | No | `gmma 128×128 INT4 1000MHz  ` |
| lpddr5_3b | 51.2 | 5 | fsa | 20.5 | 97.2 | No | `fsa 128×128 INT4 1000MHz  ` |
| lpddr5_3b | 51.2 | 6 | input_stationary | 11.1 | 99.0 | No | `inpu 128×128 INT4 1000MHz  ` |
| lpddr5_3b | 51.2 | 7 | tensor_core | 9.9 | 99.0 | No | `tens 128×128 INT4 1000MHz  ` |
| lpddr5_3b | 51.2 | 8 | wmma | 0.1 | 101.0 | Yes | `wmma 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 1 | os_systolic | 42.3 | 99.0 | Yes | `os_s 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 2 | block | 39.5 | 99.0 | No | `bloc 128×128 INT2 1000MHz  ` |
| lpddr5x_7b | 68.0 | 3 | gmma | 27.7 | 102.0 | No | `gmma 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 4 | systolic | 23.1 | 97.0 | No | `syst 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 5 | fsa | 23.1 | 97.2 | No | `fsa 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 6 | input_stationary | 14.8 | 99.0 | No | `inpu 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 7 | tensor_core | 12.6 | 99.0 | No | `tens 128×128 INT4 1000MHz  ` |
| lpddr5x_7b | 68.0 | 8 | wmma | 0.1 | 101.0 | Yes | `wmma 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 1 | os_systolic | 255.0 | 99.0 | Yes | `os_s 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 2 | gmma | 166.9 | 102.0 | No | `gmma 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 3 | block | 128.3 | 107.0 | No | `bloc 128×384 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 4 | input_stationary | 88.8 | 99.0 | No | `inpu 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 5 | tensor_core | 43.9 | 99.0 | No | `tens 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 6 | systolic | 26.2 | 97.0 | No | `syst 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 7 | fsa | 26.2 | 97.2 | No | `fsa 128×128 INT4 1000MHz  ` |
| hbm2e_7b | 410.0 | 8 | wmma | 0.1 | 101.0 | Yes | `wmma 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 1 | os_systolic | 310.9 | 99.0 | Yes | `os_s 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 2 | gmma | 203.5 | 102.0 | No | `gmma 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 3 | block | 131.4 | 107.0 | No | `bloc 128×384 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 4 | input_stationary | 108.1 | 99.0 | No | `inpu 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 5 | tensor_core | 48.2 | 99.0 | No | `tens 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 6 | systolic | 26.4 | 97.0 | No | `syst 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 7 | fsa | 26.4 | 97.2 | No | `fsa 128×128 INT4 1000MHz  ` |
| onchip_7b | 500.0 | 8 | wmma | 0.1 | 101.0 | Yes | `wmma 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 1 | os_systolic | 310.9 | 99.0 | Yes | `os_s 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 2 | gmma | 203.5 | 102.0 | No | `gmma 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 3 | block | 131.4 | 107.0 | No | `bloc 128×384 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 4 | input_stationary | 108.1 | 99.0 | No | `inpu 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 5 | tensor_core | 48.2 | 99.0 | No | `tens 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 6 | systolic | 26.4 | 97.0 | No | `syst 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 7 | fsa | 26.4 | 97.2 | No | `fsa 128×128 INT4 1000MHz  ` |
| onchip_7b_chat | 500.0 | 8 | wmma | 0.1 | 101.0 | Yes | `wmma 128×128 INT4 1000MHz  ` |

## Summary

| Scenario | Bandwidth (GB/s) | Evaluated | Complete | Frontier | Winner |
|---:|---:|---:|---:|---:|---|
| lpddr5_3b | 51.2 | 61 | 61 | 3 | block |
| lpddr5x_7b | 68.0 | 60 | 60 | 3 | os_systolic |
| hbm2e_7b | 410.0 | 60 | 60 | 6 | os_systolic |
| onchip_7b | 500.0 | 60 | 60 | 6 | os_systolic |
| onchip_7b_chat | 500.0 | 60 | 60 | 6 | os_systolic |

## Bandwidth sensitivity (best tok/s per engine)

| Engine | 51.2 GB/s | 68 GB/s | 410 GB/s | 500 GB/s | 500 GB/s (chat) |
|---|---:|---:|---:|---:|---:|
| systolic | 22.0 | 23.1 | 26.2 | 26.4 | 26.4 |
| os_systolic | 31.8 | 42.3 | 255.0 | 310.9 | 310.9 |
| block | 36.6 | 39.5 | 128.3 | 131.4 | 131.4 |
| tensor_core | 9.9 | 12.6 | 43.9 | 48.2 | 48.2 |
| wmma | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| gmma | 20.8 | 27.7 | 166.9 | 203.5 | 203.5 |
| input_stationary | 11.1 | 14.8 | 88.8 | 108.1 | 108.1 |
| fsa | 20.5 | 23.1 | 26.2 | 26.4 | 26.4 |

## Continuity check

- **Winners by scenario:** {'lpddr5_3b': 'block', 'lpddr5x_7b': 'os_systolic', 'hbm2e_7b': 'os_systolic', 'onchip_7b': 'os_systolic', 'onchip_7b_chat': 'os_systolic'}
- **Transition smooth:** True
- **Note:** Low-BW lpddr5_3b favors block (area-efficient at 51.2 GB/s); medium/high BW favors os_systolic (scales with bandwidth).

Engine preference transitions from `block` at the lowest bandwidth (51.2 GB/s) to `os_systolic` at medium/high bandwidths (68–500 GB/s). This confirms that engine ranking is bandwidth-sensitive and that the sequential/random DRAM efficiency changes affect the relative standing of wide vs. area-efficient engines.