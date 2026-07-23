# Wave 1: official vendor envelopes

## TI AM69A / J784S4

- AM69A exposes four C7x+MMAv2 deep-learning engines, 8 TOPS each, for 32 TOPS total. Its external-memory subsystem is up to four 32-bit LPDDR4 interfaces with inline ECC and up to 68 GB/s; on-chip SRAM and caches are ECC protected.
- TI's useful application-level references are more specific than the "1-12 cameras" headline:
  - AI box: 12 x 2 MP input streams at 30 fps, inference on 0.4 MP tensors at 12 fps, 12 TOPS aggregate, and 9.49 GB/s SoC DDR bandwidth.
  - Machine vision: 3 x 8 MP at 30 fps, ROI inference at 10-30 fps, 24 TOPS aggregate, and 15.35 GB/s SoC DDR bandwidth.
  - Multi-camera AI: 8 x 2 MP at 30 fps, 24 TOPS aggregate, and 15.13 GB/s SoC DDR bandwidth.
- Sensor decode/ISP/resize/encode and their memory traffic are SoC features, not standalone-NPU requirements. The transferable workload is the resulting tensor set plus deadlines and bandwidth.

Sources:

- https://www.ti.com/lit/ds/sprsp92d/sprsp92d.pdf
- https://www.ti.com/lit/wp/spradb4/spradb4.pdf
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-edgeai/AM69A/08_06_00/exports/docs/devices/AM69A/linux/datasheet.html

## Qualcomm Dragonwing IQ-9075

- Two Hexagon tensor processors provide 50 or 100 dense INT8 TOPS, depending on SKU/power configuration.
- The EVK brief says it can concurrently process four of twelve supported 4K streams in computer-vision applications. The platform product brief separately lists up to 16 camera inputs, 12 MP maximum sensor resolution, and four 4-lane CSI-2 interfaces.
- Up to 36 GB LPDDR5 has inline/ECC protection. The platform includes a safety-oriented MCU-like subsystem with four real-time cores for monitoring, error detection, and self-test.
- Camera ports, encode/decode capacity, TSN, and MCU I/O are sensor/control-SoC features. Two tensor engines, ECC, and concurrent AI execution are accelerator-relevant; the public brief does not publish NPU scheduling, latency distribution, or per-context isolation guarantees.

Sources:

- https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_E_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf
- https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Qualcomm-Dragonwing-IQ-9075-EVK-Brief.pdf
- https://docs.qualcomm.com/doc/87-97354-1/87-97354-1_REV_C_Qualcomm_Dragonwing_IQ-9075_Module_Product_Brief.pdf

## Renesas R-Car V4H

- R-Car V4H provides 34 TOPS and four computer-vision engines for L2+/L3 ADAS/AD. Renesas states that a dual-V4H configuration supports fail-degraded operation required for L3.
- RegionID partitions the SoC into domains and extends traffic QoS to provide spatial and temporal isolation and freedom from interference.
- The platform integrates fast fault-detection/response mechanisms targeting ASIL B/D metrics over most of the processing chain, and lockstep R52 real-time cores.
- Dual-chip redundancy, CSI/ISP/IMR, TSN/CAN/FlexRay, and RT cores belong to the vehicle SoC/system. Region/domain isolation, traffic QoS, ECC, and accelerator fault containment are transferable requirements.

Sources:

- https://www.renesas.com/en/products/r-car-v4h
- https://www.renesas.com/en/blogs/exploring-entry-fusion-application-architecture-and-cost-effective-solution-utilizing-r-car-v4h
- https://www.renesas.com/en/document/tcu/r-car-v4h-document-correction-section-overview-pinassignment-apsystemcore-cpg-imp-x7-tn-rg4-b0048ae

## Horizon Robotics Journey

- Journey 5 is 128 TOPS, supports up to 16 HD cameras, and is offered in single-, dual-, and quad-chip deployments. Journey 6 spans 10-560 TOPS and advertises 24 HD cameras/multiple 4K streams.
- Journey 5 safety mechanisms include an isolated safety island, lockstep CPU, FCHM, DMA/WDT, ECC/parity-protected SRAM, inline-ECC DDR, parity/readback registers, voltage/clock/temperature monitors, and startup/runtime tests.
- Journey 6 runtime software has ASIL B certification and its development tools have ASIL D tool qualification.

Sources:

- https://www.horizon.auto/en/solutions/horizon-journey
- https://www.horizon.auto/news/technology/275
- https://en.horizon.auto/news/press/455

## EXPAND

- LEAD: Runtime task granularity and preemption — WHY: TOPS and camera counts do not establish deadline behavior — ANGLE: official TI TIDL and Horizon UCP APIs.
- LEAD: Model-memory and DDR admission contracts — WHY: tail latency is often bandwidth/queue limited — ANGLE: official model compiler/runtime profiling documentation.
- LEAD: Application frame periods versus NPU stage deadlines — WHY: no vendor publishes P99 — ANGLE: derive conservative, explicitly labeled SLOs from official workload rates.
- LEAD: Silent-hang and recovery semantics — WHY: safety requires bounded failure detection, not only ECC — ANGLE: runtime timeout/hang documentation and safety mechanisms.
