# Wave 2: Host-Connected NPU Coprocessors

## Outcome

The host-connected market spans low-power fixed-vision devices through
16-GB GenAI/VLA accelerators and partitioned PCIe cards. Model-level results,
local memory, runtime semantics, and availability are more informative than
headline TOPS.

## Compact references

- Hailo-10H M.2: PCIe 3.0 x4, 4/8 GB, 20 INT8 / 40 INT4 TOPS,
  2.5 W typical; HailoRT exposes async execution, priorities, thresholds,
  timeouts, status codes, statistics, and events.
- NXP Ara240 M.2: PCIe 4.0 x4/x2/x1, 8/16 GB, 40 proprietary eTOPS,
  6.5 W chip typical; hardware task scheduler exists but public queue/error
  ABI is incomplete.
- EdgeCortix SAKURA-II M.2: PCIe 3.0 x4, 8/16 GB, 68 GB/s,
  60 INT8 TOPS, 10 W typical; public runtime guarantees remain thin.
- DEEPX DX-M1: PCIe 3.0 x4, 4 GB LPDDR5, 25 INT8 TOPS, 2–5 W;
  async/job-ID/multi-model and structured device errors are documented.
- MemryX MX3: no external DRAM and only 40M INT8-weight capacity;
  strong deterministic fixed-model streaming, but unsuitable for VLA.
- Cambricon MLU220 M.2: PCIe 3.0 x2, 8 INT8 TOPS, 8.25 W;
  queue/telemetry mechanisms exist but public current benchmarks are weak.

## High-capacity references

- Huawei Atlas 300I Pro: PCIe 4.0 x16, 140 INT8 TOPS, 24 GB,
  204.8 GB/s, 72 W maximum.
- Atlas 300I Duo: 48/96 GB ECC, 408 GB/s aggregate, up to seven vNPUs per
  chip, but 150 W.
- SAKURA-II PCIe: 60 TOPS/16 GB/10 W single or 120 TOPS/32 GB/20 W dual,
  currently described as trial units.

## DSE consequences

- Local capacity and compiler coverage are hard feasibility gates.
- Module/card peak power, cooling, PCIe generation/lanes, and reset behavior
  must be explicit.
- Runtime fields must include async support, queue depth, priority, timeout,
  cancellation, telemetry, fault completion, context/core/device reset.
- Product maturity/status is a separate field; trial/preproduction hardware
  must not be treated like established production.

## EXPAND

- VENDOR GAP: exact MERA/Voyager, Ara, and Metis queue/error/reset contracts.
- BENCHMARK GAP: independent transformer end-to-end results for SAKURA-II and
  Ara240.

Full evidence was returned inline by the worker.
