# Wave 1 — official platform sweep

## Method

Ninety-six distinct web searches were executed across the full session. Searches used
official-domain restrictions, exact API names, PDF/app-note targeting, product-specific
terms, and repository searches. This file records the first platform-level findings.

## NVIDIA

- DriveWorks CGF decomposes a node into passes, each bound to one resource such as CPU,
  GPU, or DLA. Channels support in-process shared memory, cross-process shared memory,
  NvSciBuf/NvSciStream, and sockets.
- NvSciBuf reconciles every accessor's allocation constraints before allocating shared
  memory. NvSciSync supplies producer/consumer fences and integrates with CUDA streams.
- NvSciStream has FIFO and latest-wins mailbox queues.
- DLA submission is asynchronous relative to the host and accepts a task timeout.
- Leads: CGF/STM deadline behavior, FSI/AURIX recovery authority, DLA fallback semantics.

Sources:

- https://developer.nvidia.com/docs/drive/drive-os/6.0.5/public/driveworks-nvcgf/cgf_data_pipeline.html
- https://docs.nvidia.com/drive/drive-os-5.1.15.2L/drive-os/DRIVE_OS_Linux_SDK_Development_Guide/Graphics/nvsci_nvscibuf.html
- https://docs.nvidia.com/drive/archive/drive_os_5.1.12.4/drive-os/DRIVE_OS_Linux_SDK_Development_Guide/Graphics/nvsci_nvscisync.html
- https://docs.nvidia.com/drive/drive-os-5.2.0.0L/drive-os/DRIVE_OS_Linux_SDK_Development_Guide/Graphics/nvsci_nvscistream.html

## Qualcomm

- Robotics RB5 combines Kryo CPU, Adreno GPU, Hexagon/HTP, ISP, and dedicated vision
  engines. QNN can target CPU, GPU, HTP/cDSP, or LPAI.
- QIM's object-detection pipeline makes the split explicit: a converter performs color
  conversion, scaling, and normalization; the QNN/SNPE inference plug-in runs the model;
  a postprocess plug-in thresholds and decodes outputs.
- Snapdragon Ride Flex provides mixed-criticality isolation, QoS controls, multiple OS/VM
  support, and a dedicated ASIL-D safety island.
- Leads: FastRPC/rpcmem data plane, QNN async queue and priority, subsystem-restart errors.

Sources:

- https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/qualcomm-robotics-rb5-platform-product-brief.pdf
- https://docs.qualcomm.com/bundle/publicresource/topics/80-80020-50/gst-ai-object-detection.html
- https://www.qualcomm.com/news/releases/2023/01/qualcomm-unveils-snapdragon-ride-flex---the-automotive-industry-
- https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-10/linux_setup.html

## TI

- TDA4VM combines A72 application cores, R5F real-time cores, C7x/MMA inference,
  VPAC/DMPAC, GPU, shared DDR, and an isolated MCU island.
- The A72 owns the OpenVX application and accesses TIDL through a TIOVX node; TIDL runs
  on C7x/MMA. TI's front-camera example maps VISS/LDC/MSC to accelerators, preprocessing
  and postprocessing to A72/A53, and inference to C7x.
- TIOVX graph parameters are explicitly enqueued/dequeued. Pipeline and buffer depth make
  independent engines operate concurrently.
- Leads: cache ownership and DMA-BUF representation, timeout semantics, MCU-island
  watchdog/recovery.

Sources:

- https://www.ti.com/lit/ds/symlink/tda4vm-q1.pdf
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/07_00_00_11/exports/docs/psdk_rtos_auto/docs/user_guide/developer_notes_tidl.html
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-j784s4/11_02_00_06/exports/docs/vision_apps/docs/user_guide/group_apps_dl_demos_app_tidl_front_cam.html
- https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/latest/exports/docs/tiovx/docs/user_guide/TIOVX_PIPELINING.html

## Open stack

- Autoware uses ROS 2 modular components; sensing performs acquisition/preprocessing,
  perception performs DNN and classical detection, and planning consumes semantic outputs.
- ROS 2 supplies QoS history/depth, reliability, deadline, lifespan, and liveliness.
- Loaned messages and iceoryx support zero-copy shared-memory ownership.
- Leads: ROS 2 executor determinism, lifecycle recovery, Autoware minimum-risk behavior.

Sources:

- https://autowarefoundation.github.io/autoware-documentation/1.5.0/design/autoware-architecture-v1/components/perception/
- https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html
- https://design.ros2.org/articles/zero_copy.html
- https://github.com/eclipse-iceoryx/iceoryx/wiki/Eclipse-iceoryx%E2%84%A2-in-1000-words

## EXPAND

- LEAD: NVIDIA CGF/STM scheduling constraints — WHY: determines how host deadlines map to
  heterogeneous engines — ANGLE: WCET, deadline, periodicity, overrun policy.
- LEAD: Qualcomm FastRPC/QNN queues — WHY: establishes actual descriptor and completion
  mechanics — ANGLE: rpcmem, DMA-BUF, dspqueue, executeAsync, SSR.
- LEAD: TI cache and MCU reset behavior — WHY: shared DDR is unsafe without coherency and
  failure ownership — ANGLE: TIOVX map/unmap, watchdog, MCU-only recovery.
- LEAD: ROS 2 executor and fail-safe limits — WHY: middleware QoS is not a hard compute
  deadline — ANGLE: executor scheduling, lifecycle, Autoware diagnostics and MRM.
