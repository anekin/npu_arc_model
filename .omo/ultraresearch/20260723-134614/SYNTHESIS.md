# Current edge-compute platforms for robots and physical AI

Research cut: 2026-07-23 (Asia/Shanghai). All sources are official vendor pages or
vendor-hosted documents. An undated product page is labeled as such rather than assigned a
guessed publication date.

## Bottom line

There is no defensible cross-vendor ordering by the headline `TOPS` number:

- Jetson Thor's 2,070 figure is **FP4 sparse TFLOPS**, not dense INT8 TOPS.
- Jetson Orin's family headline is **sparse INT8 aggregate** compute.
- Qualcomm IQ10 is unusually explicit: **700 sparse / 350 dense INT8 TOPS**.
- Renesas RZ/V2H is **8 dense / 80 sparse INT8 TOPS**; its tenfold headline depends on
  pruning.
- Ambarella CV3 and NXP Ara240 use vendor-defined **equivalent TOPS (eTOPS)**.
- Hailo-10H explicitly separates **40 INT4 / 20 INT8 TOPS**.

For design-space exploration (DSE), every compute value must therefore carry precision,
sparsity, scope, included engines, and power definition. Unknown public data must remain
unknown.

## NVIDIA

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| Jetson T5000 (Jetson Thor) | 2,070 FP4 sparse TFLOPS; Blackwell GPU, 2,560 CUDA cores, fifth-generation Tensor Cores; PVA v3. | 14-core Arm Neoverse V3AE; 128 GB 256-bit LPDDR5X, 273 GB/s; 40–130 W module mode. | MIG partitions ten TPCs; up to 20 HSB cameras (six direct MIPI, 32 virtual); physical-AI foundation models, VLA/VLM/LLM, perception and control. On-SoC Functional Safety Island is distinct from the main GPU. | Strong module-level constraints, but the compute peak is sparse FP4 and cannot be compared directly to INT8 NPU TOPS. |
| Jetson T4000 | 1,200 FP4 sparse TFLOPS; Blackwell GPU with 1,536 CUDA cores; MIG over six TPCs; PVA v3. | 12-core Neoverse V3AE; 64 GB 256-bit LPDDR5X, 273 GB/s; 40–70 W. | Same Thor software/physical-AI positioning; three 25 GbE ports versus four on T5000. | DSE-ready as a module if FP4-sparse is modeled explicitly. |
| Jetson AGX Orin 64 GB / 32 GB | Up to 275 / 200 sparse INT8 TOPS, from Ampere GPU plus two NVDLA v2 engines and PVA v2. The brief shows the 64 GB GPU Tensor path as 170 sparse versus 85 dense INT8 TOPS, illustrating the sparsity uplift. | 12 / 8 Cortex-A78AE cores; 64 / 32 GB 256-bit LPDDR5 at 204.8 GB/s; 15–60 / 15–40 W. | Multiple independent accelerators permit perception pipelines in parallel; up to six physical and 16 virtual cameras. Robotics, autonomous machines, multi-camera perception. | Good public module envelope. Store aggregate sparse peak separately from GPU-dense and DLA availability. |
| Jetson Orin NX 16 GB / 8 GB | 157 / 117 sparse INT8 TOPS; Ampere GPU, Tensor Cores, NVDLA and PVA. | 16 / 8 GB LPDDR5, 102.4 GB/s; 10–40 W family envelope. | Small robot/AMR, multi-camera perception, generative-AI inference under a lower SWaP envelope. | Good first-order DSE candidate; validate application-specific power and simultaneous-engine efficiency. |
| Jetson Orin Nano 8 GB / 4 GB | 67 / 34 sparse INT8 TOPS. | 8 GB model: 102 GB/s-class LPDDR5; 4 GB model has a narrower memory configuration; 7–25 W family envelope. | Entry robotics and vision; shares JetPack/CUDA deployment stack but has less accelerator and memory headroom. | Model individual SKU memory exactly; do not treat the family as one point. |
| NVIDIA IGX Orin | Integrated AGX Orin-class compute; optional RTX 6000 Ada raises the system headline to 1,705 TOPS. | 12 A78AE; 64 GB ECC LPDDR5, 204.8 GB/s; up to 125 W without and 400 W with discrete GPU; dual 100 GbE ConnectX-7. | Infineon AURIX TC397 safety MCU, BMC, enterprise/industrial I/O, 10-year lifecycle to 2033; safety-certified industrial/medical edge systems. | Treat as a board/system, not a SoM. The 1,705 number includes an optional discrete GPU and a different power/cooling class. |
| IGX Thor Developer Kit Mini / full | Mini: integrated T5000, 2,070 FP4 sparse. Full kit: up to 5,581 FP4 sparse with optional RTX PRO 6000 Blackwell Max-Q. MIG supports isolated concurrent AI workloads. | Integrated 128 GB LPDDR5X at 273 GB/s; full-kit dGPU adds 96 GB GDDR7 at 1,792 GB/s. Integrated module 40–130 W; optional dGPU up to 300 W. | Renesas RH850/U2A16 safety MCU plus independent on-SoC Safety Island; Linux/RTOS or hypervisor-hosted Linux and QNX with freedom-from-interference support. | Keep mini, full-integrated, and full-plus-dGPU as three distinct DSE configurations. |

The current [Jetson Thor product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/)
is undated; NVIDIA's [technical introduction](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/)
is dated 2025-08-25. The [Jetson Orin product page](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
and [Jetson AGX Orin technical brief v1.2](https://www.nvidia.com/content/dam/en-zz/Solutions/gtcf21/jetson-orin/nvidia-jetson-agx-orin-technical-brief.pdf)
are current but the page/brief exposes no reliable publication date. NVIDIA published the
[IGX Thor Mini brief](https://developer.download.nvidia.com/assets/igx/robotics-datasheet-igx-thor-developer-kit-mini-nvidia-us-web1.pdf),
[full IGX Thor brief](https://developer.download.nvidia.com/assets/igx/robotics-datasheet-igx-thor-developer-kit-nvidia-us-web.pdf),
and [IGX Thor safety brief](https://developer.download.nvidia.com/assets/igx/robotics-product-brief-igx-thor-safety-4473375.pdf)
in 2026-02. The [IGX product page](https://www.nvidia.com/en-gb/edge-computing/products/igx/)
is undated.

Safety wording needs exact scope. The IGX Thor safety brief describes development processes
compliant with ISO 26262 ASIL D and IEC 61508 SIL 3 / SC 3, while the Thor SoC's stated
random-hardware integrity is ASIL B / SIL 2 and the independent Safety Island carries the
higher SIL 3 role. These are not interchangeable claims.

## Qualcomm

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| Dragonwing IQ10 | 700 sparse / 350 dense INT8 TOPS; multicore NPUs plus GPU. | 18 Oryon CPU cores. SoC memory bandwidth and processor power are not public. The June 2026 reference design has 64 GB LPDDR5X and 512 GB UFS 4.0, but still no DRAM bandwidth or maximum input power. | Dedicated safety island up to SIL 3; multi-OS; 20+ sensors/cameras at processor level. Reference design: 12 GMSL2 cameras, LiDAR/ToF/IMU, 2x10 GbE main plus 2.5 GbE safety-domain Ethernet, 4x EtherCAT and 8x CAN-FD. Industrial AMRs, humanoids, VLA/VLM/LLM, perception, planning and motion control. | Compute normalization is excellent, deployability is incomplete. Obtain power and bandwidth under NDA or measure the RRD. |
| Dragonwing IQ9 / IQ-9075 | 50- and 100-**dense**-INT8-TOPS SKUs; two Hexagon Tensor Processors plus Adreno GPU. | Eight Kryo cores up to 2.36 GHz; up to 36 GB over six 16-bit LPDDR5 channels at a stated 3,200 MHz, with inline ECC. Public sustained bandwidth and power are not specified. | Four-core real-time MCU subsystem with independent I/O; up to 16 concurrent camera inputs and four concurrent 4K streams; industrial robotics, AMR, drones, GenAI and perception. | Compute/memory capacity are now public and suitable for first-order DSE. Power and unambiguous sustained DRAM bandwidth still require vendor data or measurement. |
| Dragonwing IQ8 | 40 **dense** TOPS; NPU, GPU, DSP and real-time MCU are independently usable. | Eight Kryo Gen 6 cores around 2.35 GHz; LPDDR5X interface at 3,200 MHz, ECC; public capacity/bandwidth/power absent. | Physically and electrically separated subsystem with four dedicated real-time CPU cores, Ethernet and CAN-FD; up to 12 cameras; -40 to +125 °C, AEC-Q100 Grade 3; support planned through 2038. | Good architecture/safety partition data, but requires vendor power and memory-capacity disclosure. |
| Dragonwing IQ-X7181 / IQ-X5121 | Up to 45 TOPS. | 12 / 8 Oryon cores up to 3.4 GHz; up to 64 GB LPDDR5X over eight 16-bit channels at a stated 4.2 GHz; no public power. | Up to six cameras, Windows 11 IoT Enterprise LTSC, industrial -40 to +105 °C. Targets industrial PCs, machine vision, automation and edge AI rather than safety-critical robot control. | Capacity is DSE-ready; bandwidth wording is insufficiently clear to normalize without a datasheet, and power is unknown. |
| Robotics RB5 / QRB5165 | 15 TOPS Qualcomm AI Engine aggregate; precision and sparsity are not stated. Hexagon 698 plus Tensor Accelerator, Adreno 650 GPU and CPU. | Kryo 585 4+4 cores up to 2.84 GHz; up to 16 GB LPDDR5 at 2,750 MHz or LPDDR4X at 2,133 MHz; no official TDP. | Up to seven concurrent cameras, ROS 2, -30 to +105 °C, product support to 2029. Robots, drones, industrial perception. | Mature but incomplete DSE point: measure power and model throughput; avoid comparing its aggregate 15 directly with dense INT8. |
| Dragonwing RB3 Gen 2 / QCS6490 | 12 TOPS; precision/sparsity not public. | Octa-core CPU and Adreno 643. Core kit provides 6 GB LPDDR4X and 128 GB UFS; no official power. | Camera and PCIe expansion; embedded vision, service robots and lower-tier industrial edge. | Board configuration is usable for prototyping; SoC power and metric basis remain gaps. |
| Snapdragon Ride / Ride Flex / Elite | Current Flex and Elite pages do not publish a comparable absolute TOPS/power point. Historical Ride platform scaling was 10 TOPS below 5 W through more than 700 TOPS around 130 W. The high end could combine multiple SoCs and accelerators. | Configuration-dependent; current Elite claims relative generational gains rather than absolute memory/power. | Flex combines cockpit, ADAS and AD on one SoC with hardware isolation, QoS, hypervisor-separated VMs, RTOS/AUTOSAR and a dedicated ASIL-D safety island; up to 20 cameras / 40+ sensors on Elite. | Use only a quoted, orderable SKU/BOM. The historical 700-TOPS/130-W point is a multi-device platform envelope, not a single-chip specification. |

Qualcomm's undated [IQ10 product page](https://www.qualcomm.com/internet-of-things/products/iq10-series)
states both dense and sparse compute. The
[IQ10 Robotics Reference Design brief, Rev A](https://docs.qualcomm.com/doc/87-A0789-1/87-A0789-1_REV_A_Qualcomm_Dragonwing_IQ10_Robotics_Reference_Design_Product_Brief.pdf)
and [reference-design article](https://www.qualcomm.com/news/onq/2026/06/dragonwing-iq10-robotics-reference-design)
were published in 2026-06. Other current official sources are the undated
[IQ-9075 page](https://www.qualcomm.com/internet-of-things/products/iq9-series/iq-9075),
[IQ8 page](https://www.qualcomm.com/internet-of-things/products/iq8-series),
[IQ8 product brief Rev A](https://docs.qualcomm.com/bundle/publicresource/87-83839-1_REV_A_Qualcomm_IQ8_Series_Product_Brief.pdf),
[RB3 Gen 2 brief Rev B](https://docs.qualcomm.com/bundle/publicresource/87-79890-1_REV_B_Qualcomm_Dragonwing_RB3_Gen_2_Vision_Kit_Product_Brief.pdf),
and [RB5 brief Rev B](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qualcomm-robotics-rb5-platform-product-brief.pdf).
The [IQ-X brief Rev A](https://docs.qualcomm.com/bundle/publicresource/87-94649-1_REV_A_Qualcomm_Dragonwing_IQ-X_Series_Product_Brief.pdf)
and [launch release](https://www.qualcomm.com/news/releases/2025/11/qualcomm-launches-dragonwing-iq-x-series--transforming-industria)
are dated 2025-11.

For Ride, the relevant exact dates are the
[original Ride release](https://www.qualcomm.com/news/releases/2020/01/qualcomm-accelerates-autonomous-driving-new-platform-qualcomm-snapdragon)
(2020-01-06), [expanded scaling release](https://www.qualcomm.com/news/releases/2021/01/qualcomm-announces-expansion-scalable-snapdragon-ride-platform-portfolio)
(2021-01), and [Ride Flex release](https://www.qualcomm.com/news/releases/2023/01/qualcomm-unveils-snapdragon-ride-flex---the-automotive-industry-)
(2023-01-04). The current [Ride page](https://www.qualcomm.com/automotive/solutions/snapdragon-ride)
and [Elite page](https://www.qualcomm.com/automotive/products/elite) are undated.

## Texas Instruments

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| TDA4VH-Q1 | 32 INT8 TOPS from four C7x DSP + MMA v2 accelerators; two VPAC and DMPAC vision blocks, GPU. No sparsity multiplier is stated. | Eight A72 at 2 GHz; four independent 32-bit LPDDR4-4266 interfaces, up to 68 GB/s aggregate; no fixed TDP. | Eight R5F cores, including two in an isolated MCU domain; hardware isolation and multiple OS options. Designed to support ISO 26262 ASIL D / IEC 61508 SIL 3 systems. Multi-camera ADAS, sensor fusion, robotics and machine vision. | Architecturally DSE-ready; use TI's use-case power-estimation spreadsheet and final board measurement, not a guessed TDP. |
| TDA4VM-Q1 | 8 INT8 TOPS from C7x+MMA; two C66x DSPs, VPAC/DMPAC and GPU. | Two A72 at 2 GHz; one 32-bit LPDDR4-4266 interface, about 17.1 GB/s raw theoretical; no fixed TDP. | Six R5F cores, two isolated in MCU domain; ASIL-D/SIL-3 design support. Front camera, surround view, entry autonomous machines. | Good low/mid-tier Jacinto candidate after power estimation. |
| TDA4VE/AL/VL-Q1 family | Up to 8 INT8 TOPS, SKU-dependent vision and graphics blocks. | Two A72-class application cores on relevant vision SKUs; 32-bit LPDDR4-4266-class interface; no public fixed TDP. | Safety-oriented Jacinto domains and real-time cores; smart camera, machine vision, transport, retail and surveillance. | Treat each orderable part separately; family-level feature aggregation can overstate an individual SKU. |
| AM69A | 32 INT8 TOPS from four C7x+MMA cores; two VPAC, DMPAC, video and GPU engines. | Eight A72 at 2 GHz; up to four 32-bit LPDDR4-4266 interfaces, up to 32 GB total and 68 GB/s raw aggregate; no fixed TDP. | One to 12 cameras; AMR, mobile DVR, machine vision and industrial AI. The AM69A datasheet marks functional-safety targeting **No**. | Strong non-safety industrial DSE point. Do not inherit TDA4VH's automotive safety claim merely because the silicon architecture is related. |

The [TDA4VH-Q1 page](https://www.ti.com/product/TDA4VH-Q1) lists the current
[datasheet Rev C](https://www.ti.com/lit/ds/symlink/tda4vh-q1.pdf) dated
2025-11-04 and a power-estimation guide dated 2024-12-23. The
[TDA4VM-Q1 datasheet Rev L](https://www.ti.com/lit/ds/symlink/tda4vm-q1.pdf)
was current on 2026-06-04; the [TDA4VE-Q1 datasheet Rev C](https://www.ti.com/lit/ds/symlink/tda4ve-q1.pdf)
is dated 2025-11. The [AM69A page](https://www.ti.com/product/AM69A) and
[AM69A datasheet Rev E](https://www.ti.com/lit/ds/symlink/am69a.pdf) are
current in 2026. TI's [Jacinto functional-safety overview](https://www.ti.com/document-viewer/lit/html/SPRAD57/GUID-8BB2A388-A8F2-440F-ADED-AF094C0DC075)
explains SEooC/process support; it does not make every catalog derivative safety-targeted.

## Hailo

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| Hailo-10H | 40 INT4 / 20 INT8 TOPS. It is a host-attached AI accelerator, not a standalone robot processor. | M.2 module options carry 4 or 8 GB LPDDR4/4X; PCIe Gen3 x4; 2.5 W typical. Host CPU is external. | GenAI, LLM/VLM and vision inference; industrial -40 to +85 °C, automotive -40 to +105 °C options. | Excellent accelerator-level DSE point if host power, PCIe traffic and supported model/operator set are added. |
| Hailo-8 | 26-TOPS vendor headline; official benchmarks use INT8. All required neural-network memory is on die, so no external DRAM is needed. | Host-attached PCIe accelerator; 2.5 W typical. | Simultaneous multi-stream and multi-model inference; scalable across multiple devices. AEC-Q100 Grade 2 and vendor wording “ISO 26262 ASIL-B(D) compliant” require exact certificate/safety-manual review. | Strong vision-accelerator option; it cannot replace host CPU, control MCU or general GPU. |
| Hailo-8L | 13 TOPS, vision-inference accelerator; no external DRAM. | PCIe Gen3 x2 M.2; 1.5 W typical; host CPU external. | Entry multi-camera vision and small edge endpoints. | Very favorable accelerator SWaP; operator fit and host overhead dominate beyond the accelerator headline. |
| Hailo-15H | 20 TOPS integrated vision processor; ISP and DSP can operate alongside AI. | Quad A53 at 1.3 GHz; minimum 2 GB DRAM; 32-bit LPDDR4/4X-4266, about 17.1 GB/s raw; under 5 W. | Concurrent image denoising and analytics; smart cameras and on-camera perception rather than central robot planning. | DSE-ready for camera nodes, not a foundation-model central computer. |

All Hailo product pages are current but undated: [Hailo-10H](https://hailo.ai/products/ai-accelerators/hailo-10h-ai-accelerator/),
[Hailo-10H M.2](https://hailo.ai/products/ai-accelerators/hailo-10h-m-2-ai-acceleration-module/),
[Hailo-8](https://hailo.ai/products/ai-accelerators/hailo-8-ai-accelerator/),
[Hailo-8 M.2](https://hailo.ai/products/ai-accelerators/hailo-8-m2-ai-acceleration-module/),
[Hailo-8L](https://hailo.ai/products/ai-accelerators/hailo-8l-ai-accelerator-for-ai-light-applications/),
and [Hailo-15H](https://hailo.ai/products/ai-vision-processors/hailo-15h-ai-vision-processor/).
The [Hailo-8L M.2 brief Rev 2.0](https://hailo.ai/wp-content/uploads/2023/10/Hailo-8L-M.2-ET-Product-Brief-Rev2.0.pdf)
and [Hailo-15 family brief](https://hailo.ai/wp-content/uploads/2023/12/hailo-15-product-brief.pdf)
are vendor-hosted current documents.

## Ambarella

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| CV7 | Third-generation CVflow AI, advertised as over 2.5 times CV5 AI performance; **no absolute public TOPS**. ISP and codec provide 8K60 and multi-stream processing. | Quad Cortex-A73; public memory capacity/bandwidth absent. Claimed 20% lower power than CV5, but no absolute watts for a defined workload. | CNN and transformer networks run concurrently with ISP/video; drones, industrial robots, security, 360-view and passive ADAS. | Not numerically DSE-ready without a gated product brief, vendor data, or board measurement. Relative multipliers are not substitutes for absolute constraints. |
| CV52S | Absolute AI TOPS not published. Integrated CVflow, ISP, codecs and vision processing. | Dual A76 at 1.6 GHz; up to 16 GB; 32-bit LPDDR4X at 4 Gb/s or LPDDR5(X) at 5 Gb/s, about 16 / 20 GB/s raw; under 3 W for the specified 4K60 recording plus advanced AI at 30 fps workload. | Multiple neural networks in parallel, virtualization, up to 14 cameras; robots, drones, cameras and conferencing. | Useful workload-specific DSE point even without TOPS; benchmark candidate models directly. |
| N1 / N1-655 | No public TOPS. Original N1 runs Llama 2 13B at up to 25 output tokens/s in a single stream under 50 W. N1-655 decodes 12 simultaneous 1080p30 streams while running multiple multimodal VLMs and CNNs at about/under 20 W. | N1-655 has eight A78AE cores; public memory capacity and bandwidth absent. | On-prem GenAI, AMR, smart-city NVR and multi-camera VLM/CNN pipelines. | Workload throughput is more useful than an opaque TOPS number, but memory and exact power test conditions are missing. |
| CV3 / CV3-AD685 | Family headline up to 500 **eTOPS**, a proprietary equivalent-performance measure with no public conversion to dense TOPS. NVP and GVP vision engines. | Up to 16 A78AE family-level; AD685 has 12 A78AE plus three dual-core lockstep R52 pairs. Public DRAM and power absent. | Up to 12 physical/20 virtual cameras; HSM, DRAM virtualization, chip target ASIL B and ASIL-D safety island; L2+ through L4 ADAS/AD, sensor fusion and path planning. | Safety architecture is relevant, but eTOPS, watts and memory gaps make the public data inadequate for quantitative cross-vendor DSE. |

Ambarella announced [CV7](https://www.ambarella.com/news/ambarella-launches-powerful-edge-ai-8k-vision-soc-with-industry-leading-ai-and-multi-sensor-perception-performance/)
on 2026-01-05; its [CES 2026 briefing](https://www.ambarella.com/wp-content/uploads/CES-2026_Product-and-Technology-Briefing_Final_No-Video.pdf)
contains the relative-performance chart. The current
[AIoT/industrial robotics page](https://www.ambarella.com/products/aiot-industrial-robotics/)
is undated. The [CV52S product brief](https://www.ambarella.com/wp-content/uploads/Ambarella_CV52S_Product_Brief.pdf)
is current but lacks a visible reliable publication date.

The [original N1 release](https://www.ambarella.com/news/ambarella-brings-generative-ai-capabilities-to-edge-devices-introduces-n1-system-on-chip-series-for-on-premise-applications/)
is dated 2024-01-08 and the [N1-655 release](https://www.ambarella.com/news/ambarella-expands-n1-edge-genai-family-with-soc-targeted-at-on-premise-multi-channel-vlm-and-nn-processing-in-under-20-watts/)
is dated 2025-01-07. Ambarella launched the
[CV3 family](https://www.ambarella.com/news/ambarella-launches-ai-domain-controller-soc-family-for-single-chip-multi-sensor-perception-fusion-and-path-planning-in-adas-to-l4-autonomous-vehicles/)
on 2022-01-04 and [CV3-AD685](https://www.ambarella.com/news/ambarella-expands-cv3-family-of-automotive-ai-domain-controllers-with-new-cv3-ad685/)
on 2023-01-05.

## NXP

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| i.MX 95 | Integrated eIQ Neutron NPU up to 2 TOPS plus Mali GPU, ISP and video engines. | Six A55 up to 1.8 GHz, M7 at 800 MHz and M33 at 333 MHz; x32 LPDDR5 up to 6.4 GT/s or LPDDR4X, 25.6 GB/s LPDDR5 raw theoretical; inline ECC/encryption. No fixed TDP. | Application, real-time and safety domains; IEC 61508 SIL 2 / ISO 26262 ASIL B platform; up to eight 1080p30 virtual cameras, 10GbE plus TSN. Industrial vision, gateway, HMI, distributed perception and deterministic control. | Strong low-power heterogeneous DSE point. NXP's rail measurements are workload values, not a board TDP. |
| i.MX 8M Plus | 2.3 TOPS integrated NPU plus ISP/GPU/VPU. | Quad/dual A53 at 1.8 GHz, M7 at 800 MHz, HiFi 4 DSP; 32-bit LPDDR4/DDR4 at up to 4.0 GT/s, about 16 GB/s raw; no fixed TDP. | Dual-camera vision, TSN/CAN-FD and real-time control. Machine vision, smart camera, HMI and compact robot node. | Mature entry perception/control point; benchmark memory contention and concurrent NPU/ISP load. |
| i.MX 94 (preproduction) | 0.5 TOPS Neutron NPU. | Four A55, two M7 and two M33 safety/control cores; x32 LPDDR4/LPDDR5 up to 4.2 GT/s with inline ECC; public power not fixed. | Integrated safety island and 2.5-GbE TSN switch; PLC, gateway and control workloads rather than central high-resolution perception. | Use where mixed-criticality control matters more than AI peak; freeze DSE inputs only after production specifications. |
| i.MX 952 (preproduction) | Integrated Neutron NPU, but the current public page does not disclose an absolute TOPS value. | Four A55, M7 and M33; x32 LPDDR5-6000 or LPDDR4X-4266, 24 / 17.1 GB/s raw theoretical. | ASIL B / SIL 2, camera/ISP and vision/sensor-fusion positioning. | Roadmap candidate only. Specifications are explicitly preproduction and AI peak/power are missing. |
| Ara240 discrete NPU | Up to 40 eTOPS; official application figures include Llama 2 7B at 14 output tokens/s, ResNet34 at 660 images/s and YOLOv8n at 313 images/s. | Host-attached NPU; up to 16 GB LPDDR4; PCIe Gen4 x4 or USB 3.2; 6.5 W typical. | LLM/VLM/VLA and vision acceleration; scale by adding devices. Host CPU/control functions remain external. | Good DSE accelerator when workload throughput is used; eTOPS itself remains proprietary and memory bandwidth is not public. |
| S32N55 control companion | No AI NPU/TOPS; not an inference accelerator. | Sixteen split-lock R52 at 1.2 GHz, two lockstep M7 pairs and 48 MB system SRAM; LPDDR4X/5/5X interfaces. | Dozens of mixed-criticality functions with hardware-enforced isolation, independent fault/reset/update domains and ASIL-D support. | Model as a real-time/safety companion to central AI, not as additive AI compute. |

The current [i.MX 95 page](https://www.nxp.com/products/i.MX95) points to the
[industrial datasheet Rev 8](https://www.nxp.com/docs/en/data-sheet/IMX95IEC.pdf)
dated 2026-04-07. NXP's [FRDM-i.MX95 guide](https://www.nxp.com/document/guide/getting-started-with-frdm-imx95%3AGS-FRDM-IMX95)
explicitly states the 2-TOPS NPU. The [power note Rev 1](https://www.nxp.com/docs/en/application-note/AN14449.pdf)
is dated 2026-02-09. At room temperature on a small sample, it reports about 1.186 W
system-idle without display, 2.792 W for an NPU workload, 3.825 W for STREAM memory, and
5.145 W for a heterogeneous stress case on the measured SoC/full rail groups. These are not
whole-board maxima or guaranteed TDPs. The [NETC virtualization note](https://www.nxp.com/docs/en/application-note/AN14542.pdf)
was revised 2026-04-20.

The [i.MX 8M Plus page](https://www.nxp.com/products/i.MX8MPLUS),
[i.MX 94 page](https://www.nxp.com/products/i.MX94), and
[i.MX 952 page](https://www.nxp.com/products/i.MX-952) are current; the latter identifies
the part as preproduction and lists a fact sheet dated 2026-03-04. The i.MX 94 page also
labels that part preproduction; its [launch release](https://www.nxp.com/company/about-nxp/newsroom/NW-NXP-NEW-IMX94-APPLICATIONS-PROCESSORS)
is dated 2024-11-12 and states the 0.5-TOPS NPU. The
[Ara240 page](https://www.nxp.com/products/ARA240) lists industrial/commercial datasheets
Rev 3 dated 2026-07-03, while the
[Ara240 fact sheet Rev 2](https://www.nxp.com/docs/en/fact-sheet/ARA240DNPUFS.pdf)
is dated 2026-04-01. The [S32N55 introduction](https://www.nxp.com/company/about-nxp/newsroom/NW-NXP-PIONEERS-REAL-TIME-S32N55)
and [technical article](https://community.nxp.com/t5/NXP-Tech-Blog/Introducing-the-S32N55-processor-for-real-time-super-integration/ba-p/1842706)
are dated 2024.

NXP also documents a valuable system partition: i.MX processors can perform distributed
sensor preprocessing and deterministic control while a central NVIDIA GPU runs larger
models. The official [physical-AI sensor-bridge solution](https://www.nxp.com/design/design-center/development-boards-and-designs/perception-physical-ai-solutions%3APERCEPTION-PHYSICAL-AI-HOLOSCAN-SENSOR-BRIDGE)
supports this architecture and should be modeled as an alternative to routing every raw
sensor stream to the central compute.

## Renesas

| Platform | AI compute and partition | CPU, memory, power | Concurrency, safety, and workload fit | DSE assessment |
|---|---|---|---|---|
| RZ/V2H | 8 dense / 80 sparse INT8 TOPS from DRP-AI3. The sparse figure assumes hardware-supported pruning. A separate DRP handles OpenCV/image/dynamic processing. | Four A55 at 1.8 GHz, two R8 real-time cores at 800 MHz and M33 at 200 MHz; two x32 LPDDR4/4X-3200 channels, 25.6 GB/s raw aggregate; 6 MB ECC SRAM. Exact SoC watts are not public. | Simultaneous Linux plus real-time processing, four CSI-2 camera inputs, optional ISP/GPU. Robotics, AMR, industrial vision and drones. No public functional-safety certification claim. | Good architecture point if dense 8 is the comparable peak. The 80 and 10-TOPS/W marketing figures must not be used as dense/whole-SoC values. |
| RZ/V2N | 4 dense / 15 sparse INT8 TOPS from DRP-AI3. | Four A55 at 1.8 GHz plus M33; one x32 LPDDR4/4X-3200 channel, 12.8 GB/s raw; 1.5 MB ECC SRAM; no exact watts. | Two cameras, optional Mali-G31/ISP; fanless mid-tier smart factory, robot and vision endpoint. It lacks V2H's R8 real-time pair. | DSE-ready for memory/compute after power measurement; model dense and sparse separately. |
| RZ/V2L | 0.5 dense TOPS. | Dual/single A55 at 1.2 GHz plus M33; 16-bit DDR4-1600, about 3.2 GB/s raw; no exact watts. | Fanless entry vision; one camera; longevity program to 2038. | Appropriate for compact single-camera inference, not central physical-AI planning. |
| R-Car V4H | 34 TOPS deep-learning accelerator plus computer-vision engines and a 150-GFLOPS GPU; precision/sparsity basis is not stated on the product page. | Four A76 at 1.8 GHz plus three lockstep R52 at 1.4 GHz; 64-bit LPDDR5-6400, about 51.2 GB/s raw; exact whole-SoC power absent. | Single-chip L2+/L3 ADAS central ECU; dual-chip fail-degraded L3; sensor fusion, surround view, parking. Development process targets ASIL-D systematic capability; signal IP and real-time domains have differentiated ASIL B/D claims. | Useful automotive architecture point, but the advertised 16 TOPS/W is accelerator efficiency and cannot be inverted into whole-SoC watts. |

The [RZ/V2H page](https://www.renesas.com/en/products/rz-v2h) and
[datasheet Rev 1.20](https://www.renesas.com/en/document/dst/rzv2h-group-datasheet)
(2025-03-07) state both dense and sparse values. The
[launch release](https://www.renesas.com/en/about/newsroom/renesas-unveils-powerful-single-chip-rzv2h-mpu-next-gen-robotics-vision-ai-and-real-time-control)
is dated 2024-02-29. The [RZ/V2N flyer](https://www.renesas.com/en/document/fly/renesas-rzv2n-group)
and [launch release](https://www.renesas.com/en/about/newsroom/renesas-extends-mid-class-ai-processor-line-rzv2n-integrating-drp-ai-accelerator-smart-factories-and)
are dated 2025-02 and 2025-03-11 respectively. The
[RZ/V2L page](https://www.renesas.com/en/products/rz-v2l) is current and undated.

The current [R-Car V4H page](https://www.renesas.com/en/products/r-car-v4h) is undated; the
[R-Car V4H flyer](https://www.renesas.com/en/document/fly/renesas-r-car-v4h) is dated
2022-05.

## DSE-ready model

Use a configuration record, not a single score:

1. **Scope:** SoC, accelerator, module, developer kit, or multi-chip platform.
2. **Compute tuple:** numeric peak, operation/precision, dense or sparse, structured or
   unstructured pruning, included engines, and whether CPU/GPU/DSP values are aggregated.
3. **Measured workload:** model name/version, quantization, input resolution or token
   context, batch, concurrency, latency percentile, throughput and accuracy delta.
4. **Memory:** capacity, interface width/rate, vendor peak bandwidth, raw calculated
   bandwidth, ECC overhead, reservations, and measured sustainable bandwidth.
5. **Power:** idle, defined AI workload, simultaneous worst case, module/SoC/board scope,
   input power versus rail power, voltage mode, temperature, cooling and throttling.
6. **Partitionability:** independently schedulable NPU/DLA/GPU/vision blocks, MIG/VM
   partitions, SRAM/local-memory ownership, DMA/IOMMU, QoS and freedom from interference.
7. **Real-time/control:** lockstep or deterministic cores, RTOS availability, worst-case
   response, CAN/EtherCAT/TSN and whether control survives main-domain reset.
8. **Safety:** standard, target integrity level, certification status, development-process
   level, random-hardware level, safety-manual availability and claim scope.
9. **Sensor and network ingress:** physical/virtual cameras, pixel rate, ISP count,
   LiDAR/radar paths, codec capacity, Ethernet/PCIe lane topology and synchronization.
10. **Deployment:** operating temperature, lifecycle, software/driver support, model/operator
    coverage, secure boot/update, form factor, cooling volume, price and availability.

Hard constraints should reject a platform before scoring if it cannot hold the model and
KV-cache, ingest the required sensors, meet deterministic-control/safety requirements,
operate in the thermal envelope, or compile the critical operators. A useful objective is
then total system energy and cost at a required accuracy/latency/concurrency point, not
headline TOPS/W.

Raw theoretical bandwidth calculations above use `data rate × bus width / 8` and are labeled
as such. They are not guaranteed application bandwidth and should be derated with benchmark
data.

## Marketing and comparability flags

- **NVIDIA Thor:** 2,070/1,200 are FP4 sparse floating-point peaks. They are especially
  misleading beside dense INT8 NPU figures without a model-level benchmark.
- **NVIDIA Orin:** family TOPS are sparse INT8 aggregate; dense GPU Tensor throughput and
  usable DLA capacity are lower and separate.
- **NVIDIA IGX totals:** 1,705 and 5,581 can include optional discrete GPUs; these are not
  integrated-module numbers and carry much higher power/cooling.
- **Qualcomm IQ10:** 700 sparse should be paired with the published 350 dense INT8. The
  current power and memory-bandwidth omission is the bigger uncertainty.
- **Snapdragon Ride:** the historical 700-TOPS/130-W point is a scalable multi-chip platform,
  not a single SoC. Current Elite claims are relative.
- **TI:** public TOPS are clean INT8 accelerator peaks, but no fixed universal TDP exists;
  the official estimator is workload/configuration dependent.
- **Hailo:** accelerator watts exclude host CPU, host DRAM, sensor ingest and control.
- **Ambarella:** CV7's “2.5x” and “20% lower” are relative; CV3's 500 eTOPS is proprietary.
  Neither belongs in a raw-TOPS ranking.
- **NXP Ara240:** 40 eTOPS is proprietary; use disclosed model throughput and 6.5-W typical
  power instead. i.MX 95 rail measurements are not guaranteed board TDP.
- **Renesas RZ/V2H/V2N:** 80/15 sparse TOPS depend on pruning; dense peaks are 8/4. Published
  10 TOPS/W is not a whole-SoC dense efficiency value.
- **Renesas R-Car V4H:** dividing 34 TOPS by 16 TOPS/W to infer a 2.125-W SoC is invalid;
  the efficiency refers to the deep-learning subsystem.

## Public-data gaps that require vendor engagement or measurement

- IQ10 processor/RRD maximum input power and DRAM bandwidth; IQ9 power and sustained
  bandwidth; IQ8 capacity/power; current Ride SKU BOMs.
- CV7 absolute AI throughput, DRAM, power test point and safety data; CV3 conversion from
  eTOPS plus memory/power; N1/N1-655 memory details.
- TI use-case power estimates for the exact accelerator/DDR/camera load.
- Hailo operator fallback, simultaneous-model scheduling efficiency and host overhead for
  target networks.
- Ara240 memory bandwidth and eTOPS definition; i.MX 952 NPU peak/power after production
  release.
- RZ/V2H/V2N whole-SoC workload power and safety evidence; R-Car V4H whole-SoC power and
  precision basis.
- Across all vendors: sustained performance at target ambient, compiler coverage, model
  accuracy after quantization/pruning, and concurrent worst-case latency.

## EXPAND leads

1. Obtain gated/NDA datasheets for Qualcomm IQ10/IQ9, Ambarella CV7/CV3/N1, NXP Ara240 and
   current Snapdragon Ride SKUs, then populate the remaining power/bandwidth/precision fields.
2. Build one reproducible benchmark matrix: identical detector, segmentation model, VLM/VLA
   policy and LLM; fixed accuracy; batch 1 plus realistic concurrency; p50/p99 latency,
   throughput, rail/board energy and thermals.
3. Test concurrency explicitly: central policy model plus multi-camera perception plus
   deterministic control, including memory-bandwidth contention and accelerator partitioning.
4. Audit safety artifacts, not headlines: certificates, safety manuals, assumed safety
   mechanisms, failure containment, reset domains and the exact ASIL/SIL scope.
5. Compare centralized raw-sensor ingest with distributed preprocessing using i.MX,
   RZ/V/Hailo camera nodes; include cable bandwidth, synchronization, fault containment and
   total system energy.
6. Revisit NXP's i.MX 95 + Ara240 “Pro” direction and i.MX 952 after production specifications,
   and Qualcomm IQ10 production modules after vendors disclose measured power.
