# Wave 1 Direct Web Research: Market Platforms

## Key findings

- NVIDIA Jetson AGX Orin spans 15–60 W, up to 275 sparse INT8 TOPS, 64 GB LPDDR5, and 204.8 GB/s. NVIDIA's current page separately lists 85 dense INT8 GPU TOPS for the 64 GB module, showing why peak TOPS must carry density/precision metadata.
  - https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
  - https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf
- Jetson Thor targets high-end physical AI at 40–130 W with 128 GB LPDDR5X, 273 GB/s, and 2070 sparse FP4 TFLOPS; this metric is not directly comparable with dense INT8 TOPS. The platform exposes MIG, suggesting hardware partitioning for concurrent robot workloads.
  - https://www.nvidia.com/en-gb/autonomous-machines/embedded-systems/jetson-thor/
- Qualcomm Dragonwing IQ-9075 provides 50 or 100 dense TOPS, 3.8–20 W SoC power, up to 36 GB ECC LPDDR5, two tensor processors, and a four-core real-time safety subsystem. The module brief claims a 13B model at 12 tok/s; the platform brief claims the 50-TOPS SKU runs Llama 2 7B at 22 tok/s.
  - https://docs.qualcomm.com/doc/87-83840-1/87-83840-1_REV_E_Qualcomm_Dragonwing_IQ9_Series_Platform_Product_Brief.pdf
  - https://docs.qualcomm.com/doc/87-97354-1/87-97354-1_REV_C_Qualcomm_Dragonwing_IQ-9075_Module_Product_Brief.pdf
- Hailo-10H is a direct host-connected coprocessor analogue: PCIe/USB, 40 INT4 / 20 INT8 TOPS, typical 2.5 W, direct LPDDR4/4X, and industrial/automotive temperature variants.
  - https://hailo.ai/files/hailo-10h-product-brief-en/
- TI AM69A represents the continuous-vision/AMR class at 32 TOPS with four C7x/MMA accelerators, ECC SRAM/LPDDR, CPU and real-time cores, and support for 1–12 cameras.
  - https://www.ti.com/product/AM69A
  - https://www.ti.com/lit/ds/symlink/am69.pdf
- Renesas R-Car V4H shows that production physical-AI systems combine moderate dense AI compute (34 TOPS) with lockstep real-time cores, QoS/freedom-from-interference, PCIe scaling, and ASIL-oriented safety.
  - https://www.renesas.com/en/products/r-car-v4h
- China market spans 6-TOPS RK3588, 10–18 INT8 TOPS AX650-family vision SoCs, 8–22 TOPS Ascend edge modules, and 10–560 effective-TOPS Horizon Journey 6 automotive platforms. Horizon's 560 number assumes 1/2 sparsity.
  - https://www.rock-chips.com/a/cn/news/rockchip/2022/0303/1544.html
  - https://www.axera-tech.com/zh-hans/news/2819.html
  - https://support.huawei.com/enterprise/zh/doc/EDOC1100223191/81134927
  - https://www.horizon.auto/solutions/horizon-journey/horizon-journey6?tp=1

## EXPAND

- LEAD: host-connected coprocessor runtime queues and concurrency — WHY: peak TOPS does not define multi-rate robot behavior — ANGLE: HailoRT/Ascend runtime scheduling and fault APIs
- LEAD: dense/sparse/precision normalization — WHY: market metrics are incomparable — ANGLE: preserve metric tags and use workload-measured cycles in DSE
- LEAD: model-memory headroom — WHY: market products provision 8–128 GB, far above raw weights — ANGLE: weights plus KV/state/activation/double-buffer model
