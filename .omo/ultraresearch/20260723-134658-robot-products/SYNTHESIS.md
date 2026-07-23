# Deployed Robot Compute, Power, Boundaries, and Rates

Accessed 2026-07-23. This memo uses vendor product pages/manuals/SDKs and primary
papers. “Undisclosed” means no first-party disclosure was found. SDK example timing
is labeled as such and is not treated as a guaranteed hardware limit.

## Figure

- **Helix / Figure 02:** the deployed stack uses two low-power embedded GPUs. S2 is
  a 7B vision-language model on one GPU at 7–9 Hz; S1 is an 80M visuomotor
  transformer on the other at 200 Hz, outputting 35-DoF upper-body targets. S2
  consumes the latest monocular camera image, robot state, and language; its latent
  is shared asynchronously with S1. Exact GPU SKUs, CPU, power draw, and numeric
  inference latency are undisclosed. The source says deployment-latency offset is
  calibrated during training, but gives no value.
  [Helix](https://www.figure.ai/news/helix)
- **Helix 02:** adds S0, a learned 10M-parameter whole-body controller. S1 consumes
  head/palm cameras, fingertip tactile sensing, and full proprioception and emits
  full-body joint targets at 200 Hz; S0 consumes body state and desired targets and
  emits actuator commands at 1 kHz. The demonstration uses only onboard sensors.
  [Helix 02](https://www.figure.ai/news/helix-02)
- **Figure 03 sensing/power:** cameras have twice Figure 02’s frame rate, one
  quarter its latency, and 60% wider field of view; absolute FPS and latency are
  not disclosed. Palm cameras and fingertip tactile sensing expand the inputs.
  Wireless offload is advertised at 10 Gbps. The battery is 2.3 kWh, claims five
  hours at peak performance, supports 2 kW fast charging, and uses active cooling.
  [Figure 03](https://www.figure.ai/news/introducing-figure-03),
  [battery](https://www.figure.ai/news/f-03-battery-development)
- **Application behavior:** the logistics policy uses action chunks and state
  history for more frequent replanning; chunk length and replan frequency are not
  disclosed. The walking policy is ultimately applied through a kHz closed-loop
  torque controller, but the neural walking-policy rate is not given.
  [logistics](https://www.figure.ai/news/scaling-helix-logistics),
  [walking](https://www.figure.ai/news/reinforcement-learning-walking)

## Agility Robotics Digit

- **Current product direction:** Agility says real-time AI runs onboard. Its earlier
  Digit platform used NVIDIA RTX-class acceleration and is evolving toward Jetson
  AGX Thor; a separate announcement describes Jetson Thor exploration. These
  sources do not establish one exact SKU for every currently deployed Digit.
  Training is described as offboard/cloud. Current CPU SKU, accelerator division,
  model latency, and neural-policy frequency remain undisclosed.
  [NVIDIA relationship](https://www.agilityrobotics.com/content/agility-robotics-powers-the-future-of-robotics-with-nvidia),
  [Jetson Thor](https://www.agilityrobotics.com/content/agility-robotics-powering-the-future-of-robotics-with-nvidia-jetson-thor)
- **Safety and power:** current Digit claims up to four hours per charge,
  autonomous docking, and a dedicated onboard safety PLC with FSoE. Battery energy
  is not published. A later V5 discussion specifically assigns NVIDIA IGX Thor to
  safe human detection; that future-generation disclosure should not be backfilled
  into older field units.
  [product innovations](https://www.agilityrobotics.com/content/agility-robotics-announces-new-innovations-for-market-leading-humanoid-robot-digit),
  [safety architecture](https://www.agilityrobotics.com/content/built-for-the-real-world)
- **Historical real-Digit research stack (not the current product spec):** a
  primary 2021 paper ran the neural trajectory planner at 250 Hz, the main feedback
  controller at 1 kHz, and Agility’s low-level UDP interface at 2 kHz. Sensors
  included lidar, RGB, monocular depth cameras, RGB-D, IMU, and joint state. The
  neural planner produced Bézier trajectory coefficients rather than motor torque.
  [primary paper](https://arxiv.org/abs/2103.15309)
- Agility separately describes an onboard internal physics model using IMU and
  joint-position inputs for sim-to-real control, but gives no rate.
  [IsaacLab article](https://www.agilityrobotics.com/content/crossing-sim2real-gap-with-isaaclab)

## Unitree

- **G1:** base model lists an 8-core CPU; G1 EDU offers an optional high-compute
  module such as Orin without fixing a SKU. Inputs include a depth camera and 3D
  lidar. The quick-release battery is 13S, 9,000 mAh; the charger is 54 V/5 A and
  runtime is about two hours. Workload split and sensor-to-processor routing are
  undisclosed. [G1](https://www.unitree.com/g1/)
- **H1:** lists an Intel i5 for platform functions and Intel i7 for user
  development, with optional i7/Orin NX expansion. Battery is 15 Ah / 0.864 kWh,
  maximum 67.2 V. Inputs include 3D lidar and depth camera. Exact workload split
  and model latency are undisclosed. [H1](https://www.unitree.com/h1/)
- **Go2:** base lists an 8-core CPU; EDU optionally lists Orin at 40–100 TOPS. A
  standard 8,000 mAh pack claims 1–2 hours, while EDU’s 15,000 mAh pack claims 2–4
  hours; operating range is 28–33.6 V and maximum working power is approximately
  3 kW. Inputs include 4D lidar and an HD camera. [Go2](https://www.unitree.com/go2/)
- **B2:** lists Intel i5 platform compute and Intel i7 user compute, optionally
  i7/Orin NX. Battery is 45 Ah / 2,250 Wh / 58 V, with 4–6 hours general runtime
  and payload-qualified runtime claims. Sensors include 3D lidar, two depth
  cameras, and two optical cameras. [B2](https://www.unitree.com/b2/)
- **Low-level boundary/rate:** official SDK2 clients publish `rt/lowcmd` and
  subscribe to `rt/lowstate` via DDS. SHA-pinned G1 and H1 low-level examples use a
  2 ms command/control thread, i.e. 500 Hz. Go2/B2 examples also use 2 ms and
  comment that `dt` may be 1–10 ms. These are official example timings, not
  guaranteed product maxima or neural-policy rates.
  [G1 example](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/example/g1/low_level/g1_ankle_swing_example.cpp#L194-L203),
  [H1 example](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/example/h1/low_level/h1_27dof_example.cpp#L193-L208),
  [Go2 example](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/example/go2/go2_stand_example.cpp#L46),
  [B2 example](https://github.com/unitreerobotics/unitree_sdk2/blob/21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b/example/b2/b2_stand_example.cpp)

## Fourier Intelligence

- **Stock GR2:** the product claims two hours average runtime with a detachable
  battery, 53 joints, vision, and tactile sensing. IsaacLab/ROS/MuJoCo support is
  software compatibility, not evidence of an onboard NVIDIA SKU. Stock processor,
  accelerator, model latency, and task split are undisclosed.
  [GR2](https://fftai.com/products-gr2),
  [brochure](https://www.fftai.com/uploads/upload/files/20250515/2fb8d1d0dceb49cb004fdfb6996dfd25.pdf)
- **Aurora boundary/rates:** the GR2 status API reports base, contact, joint, and
  Cartesian status at 500 Hz; motor configuration is 1 Hz. The lower-body RL
  locomotion task is 50 Hz while the upper-body state manager is 500 Hz. The DDS
  client can run on the chest computer or another device; Aurora Server runs on the
  chest computer.
  [status example](https://support.fftai.com/en/docs/GR-X-Humanoid-Robot/GR2/SDK/Aurora-SDK/examples/robot_status_example/),
  [RL locomotion reference](https://support.fftai.com/en/docs/GR-X-Humanoid-Robot/GR2/SDK/Aurora-SDK/reference/controller_reference/rl_locomotion_state/)
- **ARMOR research prototype on GR1, not stock GR1:** forty ToF sensors were grouped
  through XIAO ESP boards over I2C/USB to an onboard Jetson Xavier NX, then sent
  wirelessly to a Linux RTX 4090 host. The prototype stream was 15 Hz; the deployed
  real-GR1 version used 28 ToF sensors and updated trajectories at 15 Hz. The
  84M-parameter ACT-like model consumes current/goal joints plus ToF arrays and
  emits future arm trajectories. The paper reports 50 ms for ACT-Depth and 240 ms
  for an optimized ARMOR-Policy in its evaluation, but these are not a stock GR1
  product latency.
  [Fourier deployment note](https://www.fftai.com/blog/6),
  [primary paper](https://daehwakim.com/paper/ARMOR_paper.pdf)

## UBTECH

- **Legacy Walker:** Intel i7-7500U plus i5-6200U, EtherCAT, and a 54.6 V / 10 Ah
  LiFePO4 battery; claimed runtime and charge time are both two hours. Workload
  split and rates are undisclosed.
  [Walker](https://www.ubtrobot.com/en/humanoid/products/walker)
- **Walker X:** two Intel i7-8665U processors plus NVIDIA GT 1030 (384 cores),
  EtherCAT, four-eye vision plus two RGB-D cameras, and a 54.6 V / 10 Ah battery;
  runtime is two hours. The CPU/GPU task split, action rate, and model latency are
  undisclosed. [Walker X](https://www.ubtrobot.com/en/humanoid/products/walker-x)
- **Walker Tienkung:** all variants assign motion control to an Intel i7-1355U
  (10 cores/12 threads, up to 5 GHz). The Embodied Intelligence variant adds two
  64 GB AGX Orin modules rated 275 TOPS each; Voice/Vision has one; base has none.
  The battery is 48 V, 30 Ah plus 3 Ah, with more than 3.5 hours runtime and less
  than four hours charge time. Full-body control uses CAN/EtherCAT; documented
  maxima are at least 1 kHz for legs and 400 Hz for arms. This is unusually clear
  CPU/accelerator role labeling, though exact AI-model placement and latency remain
  undisclosed.
  [Tienkung manual](https://docs.ubtrobot.com/walker-tienkung/en/docs/user-guide/6/)
- **Walker S2:** current page discloses dual-battery operation, autonomous swapping
  in under three minutes, and RGB binocular/deep-stereo vision. The accessible
  product material does not disclose processor SKU, battery energy/per-pack
  runtime, control/action rate, or model latency. A “24/7” claim refers to swapping,
  not one-pack runtime. [Walker S2](https://www.ubtrobot.com/en/humanoid/products/walker-s2)

## Boston Dynamics

- **Atlas:** current specifications claim four hours typical runtime, two hours
  under heavy lifting, autonomous battery swap in three minutes, and charging in
  1.5 hours. The robot has tactile sensing and a 360-degree camera system. Onboard
  processor, battery energy, control/action rate, model latency, and CPU/accelerator
  division are undisclosed.
  [Atlas](https://bostondynamics.com/products/atlas),
  [spec sheet](https://bostondynamics.com/wp-content/uploads/2026/01/atlas-spec-sheet.pdf)
- **Spot:** the current developer page specifies a 605 Wh Gamma battery, 90 minutes
  runtime, five stereo camera pairs, and image access through robot services. An
  older support specification says 564 Wh, so these should remain edition-specific
  rather than be blended. Base processor and fixed policy rate are undisclosed.
  [current developer specification](https://dev.bostondynamics.com/docs/concepts/about_spot.html),
  [older support specification](https://support.bostondynamics.com/articles/Knowledge/Spot-Specifications-49916),
  [robot services](https://dev.bostondynamics.com/docs/concepts/robot_services.html)
- **Optional Spot perception compute:** CORE I/O is an optional payload with NVIDIA
  Xavier NX/384-core Volta GPU. Image inference workers may run on CORE I/O, another
  network server, or cloud; Network Compute Bridge captures an image, sends it to a
  worker, and returns the result. This is a clear selectable sensor-to-host
  boundary and must not be reported as Spot’s base computer.
  [CORE I/O](https://dev.bostondynamics.com/docs/payload/coreio_documentation.html),
  [Network Compute Bridge](https://dev.bostondynamics.com/docs/concepts/network_compute_bridge.html)
- **Stretch:** the 2026 brochure claims up to 16 hours runtime, 90% charge in two
  hours, and 600–800 cases per hour. It says machine-learning vision makes
  real-time decisions, but publishes no compute SKU, model rate, or latency.
  [Stretch brochure](https://bostondynamics.com/wp-content/uploads/2026/03/Stretch-Brochure-2026-Final-1.pdf)

## Mobile ALOHA and Open X-Embodiment

- **Mobile ALOHA compute/boundary:** all data collection and inference run onboard
  a consumer laptop with RTX 3070 Ti 8 GB and Intel i7-12800H. Three Logitech
  C922x cameras send 480×640 RGB at 50 Hz to the laptop; four arms attach through
  USB serial and the mobile base through CAN. Observations include joint positions;
  actions contain 14 follower-arm targets plus base linear/angular commands.
  Numeric inference latency and CPU/GPU workload division are undisclosed.
  [primary paper](https://mobile-aloha.github.io/resources/mobile-aloha.pdf)
- **Power conflict:** paper prose calls the onboard battery 1.26 kWh / 14 kg while
  Figure 2 labels it 1,620 Wh / 12 hours. The primary source is internally
  inconsistent; both values should be reported as written rather than resolved by
  inference.
- **Loop rates:** the SHA-pinned data-collection code sets `DT=0.02` and `FPS=50`;
  it rejects an episode when average acquisition falls below 30 Hz. ACT++ real
  evaluation also schedules a 20 ms action loop. With temporal aggregation the
  policy is queried every step; otherwise it is queried at a configured chunk
  interval. This is a scheduled 50 Hz action loop, not proof of sustained 50 Hz
  neural inference; the code warns on overruns but the paper gives no achieved
  inference latency.
  [constants](https://github.com/MarkFzp/mobile-aloha/blob/0e403249c76054a68e757e590d4da4dba401c9e3/aloha_scripts/constants.py#L235-L236),
  [collection loop](https://github.com/MarkFzp/mobile-aloha/blob/0e403249c76054a68e757e590d4da4dba401c9e3/aloha_scripts/record_episodes.py#L96-L119),
  [ACT++ evaluation](https://github.com/MarkFzp/act-plus-plus/blob/26bab0789d05b7496bacef04f5c6b2541a4403b5/imitate_episodes.py#L351-L470)
- **Open X-Embodiment:** it is a cross-robot dataset/model effort, not a physical
  platform, so it has no common onboard computer or battery. RT-1-X uses the latest
  workspace RGB image plus task text at nominal 3 Hz and emits seven gripper-motion
  dimensions. The primary Open-X paper says embodiment inference needed 3–10 Hz:
  RT-1 ran locally, RT-2 via cloud networking.
  [official repository](https://github.com/google-deepmind/open_x_embodiment),
  [primary paper](https://arxiv.org/abs/2310.08864)
- **RT-1:** 35M parameters, 15 ms inference in the paper’s latency table, and 3 Hz
  closed-loop actions. Inputs are image history and language; outputs include arm,
  base, and mode commands. The exact deployed inference hardware SKU and battery
  are not disclosed.
  [project](https://robotics-transformer1.github.io/),
  [paper](https://robotics-transformer1.github.io/assets/rt1.pdf)

## Additional deployed/research platforms

- **1X current NEO:** NEO Cortex is identified as Jetson Thor, up to 2,070 FP4
  TFLOPS. Inputs include dual 8.85 MP, 90 Hz stereo fisheye cameras and link-wise
  IMUs. Battery is 842 Wh with a four-hour claim; charging is specified as six
  minutes per hour of runtime. 1X says models execute locally on Thor and Argus
  handles low-latency camera processing, while HGX B200 is used for training.
  Numeric current-NEO action rate, latency, and CPU split are undisclosed.
  [NEO](https://www.1x.tech/neo),
  [NVIDIA GTC 2026](https://www.1x.tech/discover/nvidia-gtc-2026)
- **1X generation-specific research:** NEO Gamma’s learned whole-body controller
  ran at 100 Hz. A separate GR00T collaboration demonstrated a continuous 5 Hz
  vision-action loop on either the onboard head NVIDIA GPU or an offboard GPU.
  These cannot be silently transferred to current NEO.
  [NEO Gamma](https://www.1x.tech/discover/introducing-neo-gamma),
  [GR00T collaboration](https://www.1x.tech/discover/1X-NVIDIA-Research-Collaboration)
- **Apptronik:** the previous Apollo generation explicitly included Jetson AGX Orin
  plus Orin NX onboard and a four-hour hot-swappable pack. Current Apollo 2 pages
  describe swappable batteries, opportunity charging/tethering, and deployment,
  but do not restate processor SKU or per-pack runtime. Keep the generations
  distinct. Action rate, latency, and workload split are undisclosed.
  [previous Apollo/NVIDIA](https://apptronik.com/news-collection/apptronik-collaborates-with-nvidia),
  [Apollo 2](https://apptronik.com/apollo/apollo-2)
- **PAL TALOS:** two Intel Core i7 computers are labeled control and multimedia
  PCs. Battery is 1,080 Wh, with 1.5 hours walking or three hours standby. EtherCAT
  runs at 2 kHz, up to 5 kHz; real-time `ros_control` runs at 1 kHz. IMU is 1 kHz,
  RGB is 30 Hz, and depth is 30 Hz. No accelerator or model latency is disclosed.
  [TALOS datasheet](https://pal-robotics.com/datasheet/talos/)
- **PAL TIAGo Pro:** base i5/i7 with optional NVIDIA Jetson PC, 1 kHz EtherCAT,
  36 V / 20 Ah battery, four-to-five hours with one battery and eight-to-ten with
  two. CPU/GPU workload split and model latency are undisclosed.
  [TIAGo Pro datasheet](https://pal-robotics.com/wp-content/uploads/2024/05/Datasheet-TIAGo-Pro.pdf)
- **PAL KANGAROO:** two i7 computers plus NVIDIA Jetson GPU, four RealSense D435i
  cameras, 15 Ah battery, and three-hour runtime. Control/action rate, voltage,
  workload split, and model latency are undisclosed.
  [KANGAROO datasheet](https://pal-robotics.com/wp-content/uploads/2026/05/Datasheet_ENG_KANGAROO-2026.pdf)
- **Sanctuary Phoenix:** official pages establish wheeled Phoenix generations,
  industrial data capture/deployment, improved vision/depth/telemetry, and Azure
  collaboration, but no defensible public onboard processor, battery, control or
  action rate, inference latency, or CPU/accelerator division was found.
  [generation 8](https://sanctuary.ai/news/sanctuary-ai-releases-new-generation-of-ai-robots-for-high-quality-data-capture/),
  [Microsoft collaboration](https://sanctuary.ai/news/sanctuary-ai-announces-microsoft-collaboration-to-accelerate-ai-development-for-general-purpose-robots/)
- **Tesla Optimus:** the current official AI page describes perception, navigation,
  balance, and “efficient inference hardware,” but does not publish a product
  processor SKU, battery energy/runtime, sensor rates/routing, action/control rate,
  or model latency. [Tesla AI](https://www.tesla.com/AI)

## Cross-platform conclusions

1. Publicly disclosed real-time hierarchies span three orders of magnitude:
   semantic/vision-action policies are commonly 3–15 Hz, learned joint-target
   policies 50–250 Hz, and inner state/actuator loops 500 Hz–2 kHz. These are
   different layers and should never be compared as one “robot frequency.”
2. Accelerator presence does not establish task placement. Clear public divisions
   are rare: Figure’s S2/S1 per-GPU split, Tienkung’s Intel motion controller versus
   Orin development computers, Mobile ALOHA’s one onboard laptop boundary, and
   Spot’s optional local/network/cloud worker are the strongest examples.
3. Product-generation labels are mandatory. Apollo, NEO Gamma, Digit research
   stacks, the GR1 ARMOR prototype, and future Digit V5 disclosures cannot be
   transferred to current commercial products.
4. Numeric neural inference latency is unusually sparse. RT-1’s 15 ms and the
   ARMOR research figures are exceptions; most vendors publish control frequency
   or camera-latency improvement without a model latency.
