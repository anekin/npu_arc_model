# Ultraresearch Synthesis: Deployment Properties of Representative VLA Policies

Date: 2026-07-23

## Scope and method

This synthesis is based on more than 40 varied primary-source searches, full-text extraction of fourteen papers/technical reports, official model cards and project pages, and direct inspection of eight repositories pinned to immutable commits. It distinguishes:

- policy inference/replanning rate from low-level action-execution/control rate;
- the originally published model from the latest public successor;
- paper facts from repository defaults and embodiment-specific fine-tuning configs;
- measured values from undocumented or merely qualitative claims.

Repository pins:

- RT-1: [`google-research/robotics_transformer@4569641`](https://github.com/google-research/robotics_transformer/tree/4569641b8111f3f402c32d8e24becd2a6e952ecc)
- OpenVLA: [`openvla/openvla@c8f03f4`](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9)
- Octo: [`octo-models/octo@241fb35`](https://github.com/octo-models/octo/tree/241fb3514b7c40957a86d869fecb7c7fc353f540)
- openpi: [`Physical-Intelligence/openpi@15a9616`](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac)
- GR00T: [`NVIDIA/Isaac-GR00T@9c7e746`](https://github.com/NVIDIA/Isaac-GR00T/tree/9c7e746b2cd37a810070a98ef41d290a07e806c2)
- LeRobot/SmolVLA: [`huggingface/lerobot@73dbb6f`](https://github.com/huggingface/lerobot/tree/73dbb6f43a5088583706c91fb73c6957bca5f806)
- ACT: [`tonyzhaozh/act@742c753`](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488)
- Diffusion Policy: [`real-stanford/diffusion_policy@5ba07ac`](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)

## RT-1

Primary sources: [paper](https://arxiv.org/abs/2212.06817), [project page](https://robotics-transformer1.github.io/), [pinned repository](https://github.com/google-research/robotics_transformer/tree/4569641b8111f3f402c32d8e24becd2a6e952ecc).

- Scale: 35M parameters: 16M FiLM-conditioned EfficientNet-B3 image/instruction tokenizer and 19M eight-layer decoder-only transformer.
- Inputs/context: natural-language instruction embedded with Universal Sentence Encoder plus a six-frame RGB image history from the robot camera.
- Vision: every frame is 300×300. EfficientNet produces 9×9×512 features; TokenLearner reduces 81 features to eight tokens/frame, so the transformer receives 48 visual tokens.
- Actions: a single non-autoregressive action at each policy call, not a chunk. Continuous dimensions are uniformly discretized into 256 bins: seven arm dimensions (x/y/z, roll/pitch/yaw, gripper), three base dimensions (x/y/yaw), plus a three-way arm/base/terminate mode.
- Runtime: 3 Hz closed-loop control. The network itself takes 15 ms, but the system intentionally waits until 280 ms after observation capture before applying the action, using the maximum observed full-pipeline latency to suppress jitter.
- Precision/memory/hardware: inference accelerator, numerical precision, and runtime memory are not disclosed. The 15 ms figure is therefore not portable across hardware.

## RT-2

Primary sources: [paper](https://arxiv.org/abs/2307.15818), [full project PDF](https://robotics-transformer2.github.io/assets/rt2.pdf), [project page](https://robotics-transformer2.github.io/).

- Scale: evaluated VLA variants include RT-2-PaLI-X 5B and 55B and RT-2-PaLM-E 12B. The 55B PaLI-X combines a ViT-22B image encoder with a 32B/50-layer encoder-decoder; PaLM-E-12B uses a ViT-4B image encoder. The paper also uses PaLI-3B on Language Table.
- Inputs/context: one robot camera image and a textual task prompt in VQA form. The deployed robot policy does not disclose temporal image history, proprioceptive input, image resolution, or camera count beyond the singular camera observation. PaLI-X can technically accept image sequences, but that is not evidence that RT-2 robot control did.
- Actions: one eight-integer action string: terminate, 3D translational delta, 3D rotational delta, and gripper extension. Seven continuous quantities use 256 uniform bins. PaLI-X reuses number tokens; PaLM-E overwrites the 256 least frequent vocabulary tokens. Decoding is autoregressive but constrained to valid action tokens; there is no action chunk.
- Runtime: models are served remotely from a multi-TPU cloud service. PaLI-X-55B runs at 1–3 Hz and the 5B model at about 5 Hz.
- Missing: exact image resolution, robot temporal context, TPU type/count/topology, precision, memory/HBM footprint, end-to-end and network latency, and PaLM-E-12B control rate are not disclosed.

## OpenVLA

Primary sources: [paper](https://arxiv.org/abs/2406.09246), [model card](https://huggingface.co/openvla/openvla-7b), [pinned repository](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9).

- Scale: nominally 7B. It combines a roughly 600M fused DINOv2+SigLIP visual encoder, a two-layer projector, and Llama-2 7B. Hugging Face metadata rounds the checkpoint to 8B; the paper and project name use 7B.
- Inputs/context: language plus one current 224×224 third-person RGB image. The evaluations deliberately use no wrist image, proprioception, temporal history, or multiple cameras. The paper calls these limitations/future work.
- Actions: one continuous 7D robot action at a time, with each dimension independently mapped to one of 256 bins between the dataset's 1st and 99th percentiles and represented with the 256 least-used Llama tokens. It predicts relative end-effector position/rotation plus gripper action; no chunking.
- Runtime: about 6 action predictions/s in BF16 on one RTX 4090; the physical testbeds use 5 Hz and 15 Hz non-blocking controllers, so controller rate is not the same as achieved policy rate. On an A5000, int8 is 1.2 Hz and int4 about 3 Hz. The paper's multi-GPU throughput graph is based on a slightly smaller SigLIP-only internal model, not the released fused-encoder checkpoint, so its exact per-device values should not be treated as released-checkpoint benchmarks.
- Memory/precision: paper measurements are BF16 16.8 GB, int8 10.2 GB, int4 7.0 GB; the main infrastructure section rounds BF16 to 15 GB. BF16 and int4 had similar task success; int8 success fell because its kernels were slower. Repository guidance gives roughly 14/18 GB passive/active BF16, 9/10 GB int8, and 6/7 GB int4.
- Training/fine-tuning context: pretraining used 64 A100s for 14 days (21,500 A100-hours). Full fine-tuning used eight A100s for 5–15 hours and 163.3 GB aggregate/sharded memory; LoRA rank 32 used 59.7 GB and could run 10–15 hours on one A100.
- Missing: no standardized end-to-end observation-to-actuation latency or CPU/edge benchmark; action semantics remain dataset/embodiment-specific.

## Octo

Primary sources: [paper](https://arxiv.org/abs/2405.12213), [full PDF](https://octo-models.github.io/paper.pdf), [pinned repository/config](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/scripts/configs/octo_pretrain_config.py).

- Scale: released Octo-Small is 27M; Octo-Base is advertised as 93M, while the paper's transformer-backbone table lists 86M. The frozen T5-base language encoder is 111M and is separate from the advertised policy/backbone count.
- Inputs/context: language or goal-image task conditioning; primary third-person RGB, optional wrist RGB, and proprioception. Pretraining uses current plus one history frame (`window_size=2`). Goal images are channel-stacked with observations.
- Vision: primary stream is cropped/resized to 256×256, wrist to 128×128. 16×16 patches yield 256 primary and 64 wrist tokens. Only 27% of pretraining data has a wrist stream, and the paper reports that wrist input can hurt downstream results.
- Actions: default config predicts a four-step, 7D continuous chunk using a diffusion MLP head and 20 denoising steps at train and inference. The canonical 7D space is generally end-effector delta/velocity plus gripper, but the data loader normalizes/remaps per dataset and embodiment.
- Runtime: there is no single Octo control frequency. Demonstrations/evaluations use task-specific 5, 10, and 15 Hz controllers. The paper does not report standard policy latency. Pretraining took about 14 hours on TPU v4-128; a downstream fine-tune takes under five hours on a 24 GB A5000.
- Precision/memory: repository training uses float32 transformer computation with BF16 optimizer moments in the published config, but the paper does not provide a validated inference precision, runtime memory, or accelerator benchmark.

## Physical Intelligence π family

### π0

Primary sources: [paper](https://www.pi.website/download/pi0.pdf), [official overview](https://www.physicalintelligence.company/blog/pi0), [pinned openpi](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac).

- Scale: 3.3B total = PaliGemma 3B VLM + roughly 300M action expert. A non-VLM π0-small baseline is 470M.
- Inputs/context: one current observation: language, proprioceptive robot configuration, and two or three RGB cameras depending on embodiment. There is no temporal observation history in the original model. The paper does not state pixel resolution; openpi resizes each stream to 224×224.
- Actions: continuous, embodiment-specific joint or end-effector actions via flow matching. The paper uses horizon 50 and 10 Euler denoising/integration steps. The public implementation pads actions/state to 32 dimensions.
- Runtime: action execution is 20 Hz on UR5e/Franka and 50 Hz on other platforms. At 20 Hz it replans after 16 actions (0.8 s); at 50 Hz it replans after 25 actions (0.5 s). Thus 50 Hz is low-level execution, while the deployed policy-call rate can be 2 Hz.
- Latency: three-camera RTX 4090 measurement is 14 ms image encoding + 32 ms observation-prefix forward + 27 ms for 10 flow forwards = 73 ms onboard; Wi-Fi adds 13 ms for 86 ms offboard.
- Precision/memory: current openpi says most inference is BF16 with a few FP32 stability operations and requires more than 8 GB GPU memory; LoRA fine-tuning requires more than 22.5 GB and full fine-tuning more than 70 GB. These repository requirements are shared across supported π0/π0.5 checkpoints, not publication-specific peak-memory measurements.

### π0.5

Primary sources: [paper](https://www.pi.website/download/pi05.pdf), [knowledge-insulation paper](https://www.pi.website/download/pi05_KI.pdf), [pinned openpi](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac).

- Scale/architecture: π0-derived PaliGemma VLM plus smaller action expert; the accompanying KI report describes the 3B VLM + 300M expert. π0.5 pretraining represents actions with FAST discrete tokens, while post-training adds the continuous flow-matching expert.
- Inputs/context: current images from all cameras, proprioceptive configuration, overall instruction, and at inference a generated high-level subtask. There is explicitly no temporal observation memory. The mobile platforms have four cameras (front, rear, two wrists); high-level inference uses all four, while low-level inference uses front + both wrists. openpi uses 224×224 preprocessing.
- Actions: the model first autoregressively generates a semantic subtask, then conditions continuous low-level flow matching on that subtask. The paper uses 10 denoising steps and a 50-step horizon; mobile outputs have 18–19 DoF and directly command arm/gripper/torso targets and base velocity. Public base config is padded 32D, horizon 50, max 200 non-image tokens, but checkpoint-specific configs differ (e.g. DROID 15, LIBERO 10), so 50 is not universal to every released fine-tune.
- Runtime: low-level targets execute at 50 Hz with action chunking. The paper does not publish π0.5-specific policy-call latency/hardware or replanning interval.
- Precision/memory: openpi's shared BF16/FP32 and >8 GB inference guidance applies. Crucially, the public openpi implementation currently supports only action decoding for π0.5, not the paper's autoregressive high-level subtask-generation stage; the project maintainers confirmed this in an [official repository issue](https://github.com/Physical-Intelligence/openpi/issues/647).
- Missing: no paper-specific runtime memory, exact policy-call frequency/latency, or training accelerator count; the public checkpoint does not reproduce the entire published two-level inference system.

### Current: π0.7

Primary source: [π0.7 paper](https://arxiv.org/abs/2604.15483).

- Scale: about 5B: Gemma 3 4B VLM (including a 400M vision encoder), MEM-style video-history encoder, and 860M flow action expert.
- Inputs/context: up to four current cameras (front, two wrists, optional rear), up to six history frames per camera sampled at one-second stride, history proprioception, overall/subtask language, optional three multi-view subgoal images, episode-quality/speed/mistake metadata, and a joint-vs-end-effector control-mode token. Observation/subgoal images are 448×448. The six-frame history is compressed to the token count of a single frame.
- Actions: 50 continuous flow-matched tokens/steps; joint-space or end-effector commands depending on prompt/embodiment. Five denoising steps. It executes 15 or 25 steps before replanning and uses training-time Real-Time Chunking to tolerate up to 12 steps/240 ms of delay on a 50 Hz robot.
- Runtime/control: UR5e runs at 20 Hz; other platforms at 50 Hz. A minimal three-camera policy takes 38 ms on one H100. MEM history plus subgoal context raises the worst case to 127 ms. The high-level semantic policy also runs on one H100.
- Optional world model: a separate 14B BAGEL-derived generator produces multi-view subgoals in 1.25 s using 25 denoising steps, four-way tensor parallelism on 4×H100, 8-bit large matrix multiplies, and modified SageAttention. It runs asynchronously and refreshes every four seconds or on subtask changes; it should not be confused with low-level VLA latency.
- Missing: low-level VLA inference precision and runtime memory are not published; the 8-bit statement applies to the 14B world model. Training compute and exact embodiment action dimensions are not summarized as one fixed number.

## Gemini Robotics

### Original Gemini Robotics (2025)

Primary sources: [technical report](https://arxiv.org/abs/2503.20020), [official launch](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/).

- Scale/architecture: parameter counts are undisclosed. The VLA is a distilled Gemini Robotics-ER backbone hosted in the cloud plus a local onboard action decoder.
- Inputs/context: a set of current-scene images, robot state, and text instruction. The report does not disclose camera count, image resolution, token context, or temporal history.
- Actions: continuous low-level action chunks; dimensionality, semantics, and chunk length are undisclosed and embodiment-dependent.
- Runtime: cloud backbone response is under 160 ms; raw-observation-to-action-chunk end-to-end is about 250 ms. Chunk execution yields an effective 50 Hz action rate. Baselines ran locally on RTX 4090, but that is not the Gemini Robotics inference hardware.
- Missing: model size, precision, memory, exact cloud/onboard hardware, chunk horizon, denoising/generation formulation, camera/resolution/context details, and policy replanning rate.

### Gemini Robotics On-Device

Primary source: [official model page/card](https://deepmind.google/models/gemini-robotics/gemini-robotics-on-device/).

- Private-preview local VLA. Model card exposes image+text+action input, action output, and 1,088 input tokens.
- It only claims “low latency” and efficient local execution. Parameters, image resolution/cameras, temporal context, action representation/chunking, numerical precision, memory, hardware, frequency, and latency are undisclosed.

### Current VLA: Gemini Robotics 1.5

Primary sources: [technical report](https://arxiv.org/abs/2510.03342), [official model page/card](https://deepmind.google/en/models/gemini-robotics/gemini-robotics/).

- Private-preview multi-embodiment VLA with Motion Transfer and optional natural-language “thinking” before continuous numerical actions. A GR-ER 1.5 orchestrator can plan and call the VLA; the VLA itself can append its own thinking trace before acting.
- Official card: text+image input, text+action output, 32k input-token context; training hardware TPU v4/v5p/v6e with JAX/Pathways.
- Missing: parameter count, image resolution/camera count, temporal/proprioceptive context, continuous action semantics/dimension, chunk length, replanning/control frequency, latency, inference hardware, precision, and memory. A report experiment running GR-ER success detection at 5 Hz is not a GR 1.5 VLA-control benchmark.
- Gemini Robotics-ER 1.6 (April 2026) is a newer embodied-reasoning VLM that emits text, not the current VLA; it must not be presented as an action model.

## NVIDIA GR00T

### Original GR00T N1

Primary sources: [paper](https://arxiv.org/abs/2503.14734), [official launch](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-groot-n1/).

- Scale: public N1-2B is 2.2B total, with 1.34B in the Eagle-2 VLM (SmolLM2 + SigLIP-2) and the remainder in the DiT action system and embodiment adapters.
- Inputs/context: task text, possibly multiple current 224×224 RGB images, and current proprioceptive state. Each image becomes 64 tokens after pixel shuffle. No temporal observation history is specified.
- Actions: continuous embodiment-specific chunks via a flow-matching DiT, horizon 16, four Euler denoising steps. State/action MLP encoders and decoders handle variable-dimensional joint/end-effector/gripper/whole-body spaces.
- Runtime: one 16-action chunk takes 63.9 ms on an L40 in BF16. The paper describes VLM/System-2 at 10 Hz and higher-frequency motor output up to 120 Hz; these are different layers of the system.
- Training/hardware: up to 1,024 H100s; the 2B model used about 50,000 H100 GPU-hours. A single A6000 supports compute-constrained adapter+DiT fine-tuning.
- Missing: runtime VRAM, exact camera count/action dimension across embodiments, and whether the quoted 120 Hz applies uniformly to physical deployments.

### Current: GR00T N1.7

Primary sources: [official current repository](https://github.com/NVIDIA/Isaac-GR00T), [pinned configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/gr00t/configs/model/gr00t_n1d7.py), [pinned hardware guide](https://github.com/NVIDIA/Isaac-GR00T/blob/9c7e746b2cd37a810070a98ef41d290a07e806c2/getting_started/hardware_recommendation.md).

- Scale/architecture: 3B, with Cosmos-Reason2-2B/Qwen3-VL backbone plus 16-layer flow-matching DiT. It uses relative end-effector action deltas and is pretrained with 20k hours of EgoScale human video.
- Inputs/context: language, one or more current camera images, current state; current embodiment configs use one frame (`[0]`) even though the loader supports negative history indices. Default images resize to 256×256, crop to 230×230, then resize; max model sequence length 1,024.
- Actions: continuous relative/joint actions padded to at most 132 dimensions; predicted horizon 40, four denoising steps. Execution horizon is separately configurable and must be ≤40.
- Precision/memory: BF16 model. Minimum inference requirement is one 16 GB GPU; default action-head/projector fine-tuning is under about 35 GB, while the fine-tuning minimum is 40 GB and tuning visual/LLM modules recommends 80+ GB.
- Measured replanning rate, four denoising steps, one camera: H100 11.7 Hz eager / 35.9 Hz TensorRT (85.8/27.9 ms); L40 7.8/26.0 Hz; AGX Thor 6.9/10.7 Hz; Orin 2.9/4.6 Hz. Cameras and actions execute around 30 FPS through asynchronous chunking; 30 FPS is not the model inference rate.
- Missing/config-dependent: action semantics/dimension and executed steps vary by embodiment/checkpoint. Multi-camera latency and whole-body-controller latency are not given in the one-camera benchmark.

## SmolVLA

Primary sources: [paper](https://arxiv.org/abs/2506.01844), [official Hugging Face article](https://huggingface.co/blog/smolvla), [model card](https://huggingface.co/lerobot/smolvla_base), [pinned LeRobot implementation](https://github.com/huggingface/lerobot/tree/73dbb6f43a5088583706c91fb73c6957bca5f806/src/lerobot/policies/smolvla).

- Scale: main model 450M, roughly 100M action expert; paper also evaluates 240M and 2.25B variants. It retains the first 16 layers/half of SmolVLM2 and freezes the VLM during training.
- Inputs/context: one current observation (`n_obs_steps=1`), multiple RGB cameras, one sensorimotor-state token, and text instruction. Real SO100 uses top+wrist cameras; SO101 uses top+side. Camera topology is therefore not fixed across embodiments.
- Vision: images resize/pad to 512×512. Tiling is disabled; pixel shuffle limits each frame to 64 visual tokens.
- Actions: continuous flow-matched action chunk, default horizon 50, ten flow steps, state/action padded to 32D. Causal self-attention within action tokens is interleaved with VLM cross-attention. Default synchronous config executes all 50 actions, but simulation and asynchronous serving can replace/replan after fewer actions.
- Runtime/control: the real control loop is 30 FPS (33 ms/action tick). The paper does not publish raw policy latency on a named CPU/GPU. It reports that asynchronous serving reduced mean pick-place task time from 13.75 s to 9.70 s and doubled fixed-time completions (9 to 19); this is task throughput, not per-call latency.
- Precision/hardware/memory: BF16 + `torch.compile`; four GPUs for pretraining, but the authors state it can train on one GPU and run on CPU/Mac/consumer GPU. No named-device latency or peak inference memory is disclosed.

## ACT

Primary sources: [paper](https://arxiv.org/abs/2304.13705), [project](https://tonyzhaozh.github.io/aloha/), [pinned repository](https://github.com/tonyzhaozh/act/tree/742c753c0d4a5d87076c8f69e5628c79a8cc5488).

- Scale: about 80M; per-task conditional VAE with four ResNet-18 image backbones and transformer encoder/decoder.
- Inputs/context: current observation only: four 480×640 RGB views (top, front, left wrist, right wrist) and 14D follower-arm joint positions. Each image becomes a 15×20×512 feature map; no observation history or language.
- Actions: a 100×14 tensor of future absolute target joint positions (not deltas), trained with L1 reconstruction plus CVAE KL. During inference the style latent is zero. Temporal ensembling combines overlapping chunks for each current timestep.
- Runtime: ALOHA data/action loop is 50 Hz. Model inference is about 0.01 s (about 100 model calls/s) on an 11 GB RTX 2080 Ti, where training takes about five hours. The 50 Hz number is the control/data rate.
- Precision/memory: paper does not state numerical precision or measured inference VRAM beyond the fact that the training/inference machine is an 11 GB GPU.

## Diffusion Policy

Primary sources: [paper](https://arxiv.org/abs/2303.04137), [project](https://diffusion-policy.cs.columbia.edu/), [pinned repository](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f).

- This is a configurable policy family, not one fixed model. It accepts a short state/image history and predicts a continuous action trajectory with a conditional DDPM, using receding-horizon execution. It has CNN U-Net and time-series transformer variants.
- Canonical Push-T image config: one 96×96 camera (84×84 training crop), observation horizon 2, prediction horizon 16, execute 8, 100 diffusion steps. Real setups use observation horizon 2, prediction horizon 16, and execute 6 or 8.
- Real vision: two RealSense D415 RGB views downsampled to 320×240 at 10 FPS, with 288×216 training crops; robot state accompanies images. Bimanual extensions use two scene + two wrist cameras, but this is a separate task-specific setup.
- Actions: continuous absolute position/end-effector/gripper trajectories; dimension varies by robot/task. The paper found an eight-step execution horizon best for most tasks. Real policies and demonstrations run at 10 Hz; UR5 setpoints interpolate to 125 Hz, and Franka mid-level control is around 1 kHz.
- Scale: task/config dependent. CNN table spans 22–264M diffusion-network parameters plus 0/22/45M vision encoders; real CNN configs are 67M + 22M = about 89M. The real transformer Push-T config is 80M + 22M = about 102M.
- Runtime: the methods section reports 10-step DDIM at 0.1 s on RTX 3080; final real-world hyperparameter tables use 16 inference steps. Simulation configs use 100 train and inference steps. The difference must be retained rather than quoting a universal denoising count.
- Precision/memory: not disclosed. No standard peak VRAM measurement; exact model size, cameras, horizons, and action dimensions depend on the task configuration.

## Cross-model conclusions

1. “50 Hz” often describes action execution, not model inference. π0 commonly calls the model at 1.25–2 Hz, Gemini Robotics takes roughly 250 ms per chunk, π0.7 replans after 15/25 actions, SmolVLA overlaps inference with 30 FPS execution, and N1.7 explicitly publishes a separate replanning-rate table.
2. Closed/open weights differ dramatically in deployment transparency. N1.7 provides BF16, minimum VRAM, device-specific E2E latency, image preprocessing, horizon, max dimensions, and denoising steps. Gemini Robotics 1.5 publishes a 32k context and training TPU generations but almost none of those deployment fields.
3. Model-name parameter counts are not always apples-to-apples. Octo's 27/93M excludes a separately described frozen 111M T5 encoder; OpenVLA's nominal 7B is rounded to 8B by checkpoint metadata; Diffusion Policy and ACT count policy modules differently.
4. The most common observation context evolved from single current images (RT-2/OpenVLA) toward camera history and structured prompts (π0.7). Multiple cameras are not interchangeable with temporal history.
5. Action paradigms cluster into single-step discrete bins/tokens (RT-1/2, OpenVLA), regression chunks (ACT), and diffusion/flow chunks (Octo, π family, GR00T, SmolVLA, Diffusion Policy). Horizon alone does not imply responsiveness: execution horizon and asynchronous replanning matter.
6. The public π0.5 repository is not the entire published π0.5 inference system: it omits the autoregressive high-level subtask stage.

## Missing-disclosure matrix

| Model | Major fields still missing from primary sources |
|---|---|
| RT-1 | inference accelerator, precision, runtime memory |
| RT-2 | image resolution/history/cameras, exact latency, TPU topology, precision/memory |
| OpenVLA | standardized end-to-end actuation latency, CPU/edge benchmark |
| Octo | standard inference latency/hardware/precision/memory |
| π0 | paper image resolution and publication-specific runtime memory |
| π0.5 | policy-call rate/latency/hardware/memory; full high-level inference absent from openpi |
| π0.7 | low-level precision/memory and training compute |
| Gemini Robotics family | model sizes and almost all low-level action/runtime fields |
| GR00T N1 | runtime VRAM and uniform physical-deployment camera/action spec |
| GR00T N1.7 | multi-camera/whole-body E2E latency; embodiment-specific actions |
| SmolVLA | named-device policy latency and peak inference memory |
| ACT | numerical precision and measured runtime memory |
| Diffusion Policy | universal values do not exist; per-config precision/memory largely absent |

## EXPAND

- Obtain immutable artifact revisions for the Gemini Robotics model cards; the public pages are mutable and private-preview details may change.
- Check whether Physical Intelligence releases π0.7 weights/code or a π0.7-specific memory/precision benchmark; openpi currently targets π0/π0.5.
- Benchmark released OpenVLA fused DINOv2+SigLIP checkpoints across the same devices as the paper's SigLIP-only figure.
- Benchmark SmolVLA policy-call latency and peak resident memory on the CPU, MacBook, and consumer GPU classes claimed by the official article.
- Resolve RT-2's exact robot image preprocessing/resolution from an author implementation if one becomes public.
- Add per-embodiment N1.7 multi-camera and whole-body-controller latency, because the current official table is one camera and policy-only.
