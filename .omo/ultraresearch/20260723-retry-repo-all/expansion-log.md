# Expansion Log

## Phase 0

- Core question: Exhaustively identify embodied/VLA/physical-AI scenario definitions, performance formulas, workload traces, CPU-NPU boundary modeling, missing DSE fields, and their git history in this repository.
- Axes: scenarios/workloads; formulas and metrics; CPU-NPU/runtime boundaries; DSE schemas and omissions; git archaeology.
- Scope: local codebase and all reachable git history; read-only investigation.

## Phase 1 — current `main`

- Scenario lead: `reports/arch-dse-three-scenarios.md` defines documentary S3; `sim/config/scenarios.yaml` has only generic LLM scenarios. Resolved by tracing every VLA/embodied keyword hit outside generated artifacts.
- Trace lead: Qwen-VL ViT and SD UNet trace generators exist, but the DSE CLI does not expose either. Empirical trace enumeration found missing attention matrix products and CV dispatch gaps that turn normalization, softmax, activation, pooling, and SD convolution entries into zero-cycle metadata.
- Formula lead: `dse_scenario.py` has a factor-of-two decode compute-ceiling error and a contradictory printed TTFT derivation. `design_space_explorer.py` hard-codes 1 GHz when converting cycles to both LLM tok/s and CV FPS, so the frequency sweep changes power but not performance.
- Integration lead: scenario loading/preflight functions are defined but not used to configure or constrain the current-main sweep. The YAML constraints and on-chip-memory scenario cannot be reproduced through the current CLI.
- Boundary lead: the PCIe model is functional, the software-overhead model is standalone, and the NPU simulator omits both. The documented host/firmware/DMA path is a plan, not an end-to-end timed path.

## Phase 2 — empirical checks

- Generated trace totals: Qwen-VL 1 crop 645.645 GMAC; 4 crop 2582.580 GMAC; SD UNet 206.440 GMAC.
- CV dispatch check: all Qwen `layer_norm`/`softmax`/`gelu` entries and SD `conv`/`group_norm`/`silu`/`softmax` entries are routed to zero-cycle metadata.
- At block 80x1536, the Qwen 4-crop trace simulated as 25.650 ms, reproducing the report's 26 ms while zeroing 516 non-GEMM entries. The SD trace simulated as 0.643 ms per step because all 51 convolutions and all SFU entries were zeroed.
- Frequency check: identical CV FPS and LLM tok/s at 800/1000/1200 MHz, confirming hard-coded 1 GHz conversion.
- Requirement check: `onchip_7b_chat` is reported incomplete because nested constraints are ignored by the clarification layer.
- S3 formula check: 20 output tokens at the reported 198 TPS require about 101 ms and cap the pipeline at 9.9 FPS for every listed larger array; selecting H=96 cannot meet the stated 10 FPS requirement.

## Phase 3 — all reachable git history

- `dc05eea` introduced the current scenario report, VLA/CV traces, software-overhead model, and PCIe model.
- `10fc8f2` created a scenario-driven DSE pipeline on the non-main branch.
- `10b44e7` added agent-aware LLM workload fields and the `warehouse_vla` example.
- `c93b3f1`, `2757fc7`, `2897dff`, `2ddcccf`, and `38409d8` successively added attention/dataflow/admission/cache/bandwidth-cap modeling.
- `891248c` added the scenario-B embodied report.
- `origin/feat/scenario-driven-dse` at `6288a4d` wires scenarios and hard constraints into the evaluator, fixes LLM frequency units, and models QK/PV attention, but its embodied inputs remain only an LLM token workload. It still lacks sensor, action, control-loop, QoS-tail, and host-boundary contracts; CV performance still assumes 1 GHz.
- Deletion/rename audit: no relevant source or scenario deletion/rename exists; `930331e` only removed tracked bytecode.

## Phase 4 — convergence

- Repeated keyword searches over the working tree and feature branch produced no new physical-AI schema or executable action/control trace.
- Symbol/AST and import-use searches found no hidden scenario integration or CPU-NPU timing integration on current `main`.
- `git log -S`, `git log -G`, path history, branch refs, and deletion/rename history converged on the same lineage.
- Unchecked leads: none.
