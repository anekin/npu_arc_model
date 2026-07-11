# MXUModel (64x64) vs RTL Calibration Report

> **Purpose**: Compare MXUModel (H=64, W=64) cycle predictions against module-level RTL measurements
> for the 18 configurations spanning all MX-P01..MX-P18 performance characterization cases.
>
> **Generated from**: `.omo/evidence/mxu-perf/MX-P15_calibration.md`
> **Generator script**: `CaduceusCore/scripts/calibrate_mxu_model.py`
> **How to regenerate**: `python3 CaduceusCore/scripts/calibrate_mxu_model.py`
>
> **Model**: MXUModel(H=64, W=64, f=1000MHz, INT4, double_buffer=True)
> **RTL**: 64x64 broadcast MAC array (module-level, no DMA/NoC overhead)
> **Tolerance**: |RTL - Model| / max(RTL, 1) <= 200% (wide — model includes BW-aware DMA stalls)

| # | M | N | K | RTL (cyc) | Model (cyc) | Delta | Delta% | Analysis |
|---|--:|--:|--:|:--:|:--:|:--:|:--:|----------|
| 1 | 64 | 64 | 64 | 134 | 333 | -199 | 148.5% | moderate deviation (149%); model uses prefill path with DMA/BW overhead |
| 2 | 64 | 64 | 128 | 202 | 525 | -323 | 159.9% | moderate deviation (160%); model uses prefill path with DMA/BW overhead |
| 3 | 64 | 64 | 256 | 338 | 909 | -571 | 168.9% | moderate deviation (169%); model uses prefill path with DMA/BW overhead |
| 4 | 64 | 64 | 512 | 610 | 1677 | -1067 | 174.9% | moderate deviation (175%); model uses prefill path with DMA/BW overhead |
| 5 | 64 | 64 | 1024 | 1154 | 3213 | -2059 | 178.4% | moderate deviation (178%); model uses prefill path with DMA/BW overhead |
| 6 | 64 | 128 | 64 | 267 | 525 | -258 | 96.6% | moderate deviation (97%); model uses prefill path with DMA/BW overhead |
| 7 | 64 | 256 | 64 | 533 | 909 | -376 | 70.5% | moderate deviation (71%); model uses prefill path with DMA/BW overhead |
| 8 | 64 | 512 | 64 | 1065 | 1677 | -612 | 57.5% | moderate deviation (57%); model uses prefill path with DMA/BW overhead |
| 9 | 1 | 64 | 64 | 71 | 240 | -169 | 238.0% | large deviation (238%); model uses decode path with DMA/BW overhead |
| 10 | 4 | 64 | 64 | 74 | 436 | -362 | 489.2% | extreme deviation (489%); model uses decode path with DMA/BW overhead |
| 11 | 16 | 64 | 64 | 86 | 214 | -128 | 148.8% | moderate deviation (149%); model uses prefill path with DMA/BW overhead |
| 12 | 32 | 64 | 64 | 102 | 254 | -152 | 149.0% | moderate deviation (149%); model uses prefill path with DMA/BW overhead |
| 13 | 128 | 64 | 64 | 267 | 807 | -540 | 202.2% | large deviation (202%); model uses prefill path with DMA/BW overhead |
| 14 | 64 | 64 | 80 | 154 | 525 | -371 | 240.9% | large deviation (241%); model uses prefill path with DMA/BW overhead |
| 15 | 1 | 1 | 1 | 8 | 240 | -232 | 2900.0% | extreme deviation (2900%); model uses decode path with DMA/BW overhead |
| 16 | 64 | 1 | 64 | 134 | 333 | -199 | 148.5% | moderate deviation (149%); model uses prefill path with DMA/BW overhead |
| 17 | 64 | 128 | 128 | 403 | 909 | -506 | 125.6% | moderate deviation (126%); model uses prefill path with DMA/BW overhead |
| 18 | 64 | 33 | 64 | 134 | 333 | -199 | 148.5% | moderate deviation (149%); model uses prefill path with DMA/BW overhead |

## Summary

- Rows compared: 18
- RTL total cycles (sum): 5736
- Model total cycles (sum): 14059
- Mean |delta%|: 324.8%
- Max |delta%|: 2900.0%
- Min |delta%|: 57.5%

## Interpretation

The MXUModel includes DMA and DRAM bandwidth overhead (tile weight/activation streaming) that the module-level RTL does not have. At the module level, weights and activations are loaded in a single cycle via direct bus drive. The model's DMA overhead dominates for small tiles (M=1, K=1), producing large deltas. For compute-bound prefill configurations (M≥64), the model approaches the RTL cycle counts more closely.

For accurate calibration, use the per-tile cycle formula in `analyze_perf.py` (which matches RTL exactly) rather than the DMA-aware MXUModel for module-level cycle prediction.

## Module-Level Back-to-Back Behavior (MX-P13 / MX-P14)

The back-to-back cases MX-P13 and MX-P14 are executed with `+repeat=10` in `tb_mxu_perf.v`.  The testbench inserts a deterministic **4-cycle GAP** between consecutive operations:

- MX-P13 (10× `64,64,64`): per-op total = 134 cycles, GAP = 4 cycles, std = 0
- MX-P14 (10× `64,64,256`): per-op total = 338 cycles, GAP = 4 cycles, std = 0

The GAP is **not** an RTL FSM delay. It originates in the `tb_mxu_perf.v` handshaking loop: after `status_done` rises, the testbench waits a fixed 3-cycle settling window (implemented as `repeat(3) @(posedge clk)`) before re-asserting `cmd_start`.  This yields a measured inter-op gap of 4 cycles and is **accepted as expected module-level behavior** for these cases.

Consequently, the module-level back-to-back throughput metric is:

```
throughput_cycles_per_op = per_op_total + 4
```

The per-op total itself (134 for MX-P13, 338 for MX-P14) matches the deterministic Controller FSM formula with delta = 0.

---

*This report was regenerated from `.omo/evidence/mxu-perf/MX-P15_calibration.md` as part of the MXU module-level performance characterization deliverable (MX-P01..MX-P18).*
