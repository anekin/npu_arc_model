# Wave 3: China Host-Connected Coprocessors

## Outcome

BM1688/SE9 is not a direct match for the CPU-hosted coprocessor boundary:
official material describes BM1688/CV186AH in SoC mode and SE9 as an A53-hosted
microserver. Public documentation does not establish a shipping SE9 PCIe
endpoint accelerator mode.

## Useful findings

- BM1688: 16 INT8 TOPS with unspecified dense/sparse semantics; no official
  peak for FP16/BF16/INT4, memory bandwidth, or chip power.
- SE9: 8 GB standard/16 GB maximum shared system memory, 15 W typical whole
  microserver, Ethernet/USB-oriented system connection.
- A 16 GB BSP example exposes only about 4.14 GB NPU heap plus 2 GiB VPP heap
  after system/firmware partitioning. DSE therefore needs physical total,
  NPU-reservable, and system-reserved memory as distinct fields.
- Cambricon MLU220 M.2 is the closest domestic physical analogue to Hailo:
  8 INT8 TOPS, PCIe 3.0 x2, 8.25 W, queue/notifier runtime, but public queue
  depth, priority, preemption, DMA concurrency, and local memory capacity are
  incomplete.
- Atlas 200I A2 supports PCIe endpoint mode, 8/20 INT8 TOPS, 4/8/12 GB ECC,
  21–25 W typical depending on variant, pinned memory and async H2D/D2H/model
  execution. Stream priority is fixed/reserved, so multiple streams do not
  prove real-time priority or preemption.
- HailoRT exposes explicit async readiness/backpressure, scheduling timeout,
  threshold, network priority, and burst/batch controls. Public material still
  does not prove in-kernel preemption.

## DSE additions

- `physical_memory_total` versus `npu_reservable_memory`
- exact endpoint mode and PCIe generation/lanes
- measured H2D/D2H fixed latency and sustained bandwidth
- queue depth, backpressure, outstanding limit, timeout, cancel, recovery
- scheduler priority versus true preemption and maximum blocking segment
- power scope: chip/module/system and typical/max/thermal-steady

## EXPAND

- FUTURE PRODUCT GAP: New SM9/BM1688 PCIe endpoint product form.
- HARDWARE GAP: Hailo-10H capability query and Atlas EP DMA latency/bandwidth.
- DEAD END: Public MLU220 M.2 material does not disclose enough scheduling or
  memory detail for a P99 model.
