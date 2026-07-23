# Expansion Log

## Phase 0

Core question: 调研当前市场具身智能与物理 AI 解决方案，将应用闭环需求投影到由主控 CPU 驱动的端侧 NPU 协处理器，并形成可用于本仓库 DSE 的场景参数。

Research axes:

1. 市场端侧计算平台：机器人、自动驾驶、工业/AMR 芯片的算力、功耗、内存和系统分工。
2. 公开模型负载：VLA、Vision/Language/Action、Diffusion Policy、世界模型的模型尺寸、输入输出和推理频率。
3. 产品系统边界：主控 CPU、NPU、GPU、ISP、MCU/安全岛之间的任务划分。
4. 性能需求：闭环频率、P95/P99 延迟、并行任务、传输带宽、内存容量和功耗。
5. 中国市场：机器人与物理 AI 芯片/模组公开方案及落地指标。
6. 当前代码库：已有场景、DSE 约束、视觉/LLM trace 和协处理器接口能否表达上述需求。

Codebase relevant: yes
External research: yes
Browsing: yes
Execution verification likely: yes
Requested report: no separate format; Markdown synthesis is the deliverable

## Wave 1

Planned workers: 14 total

- 4 codebase exploration lanes
- 6 external market/model research lanes
- 2 dynamic-page/browsing lanes
- 2 open-source implementation deep dives

Initial result: all specialist-role workers disconnected before findings. Six default-role retry lanes were opened, and direct orchestrator research continues. No research axis was dropped.

Direct verification opened and closed:

- Current S3 trace completeness and pipeline arithmetic: see `verify-current-s3-workload.md`.

## Wave 2

Opened four expansion lanes from direct-source and verification leads:

- `expand_action_chunk`: action execution rate versus policy inference rate and asynchronous queues.
- `expand_coprocessors`: discrete host-connected NPU products and runtime contracts.
- `expand_dual_rate`: slow reasoning plus fast reactive policy scheduling.
- `expand_physical_ai`: transferable AMR/industrial/automotive NPU job requirements when sensors remain on the host.

Completed:

- `wave-2-physical-ai.md`: converted multi-camera/product examples to a portable
  tensor-job contract, proposed P99/staleness SLOs, and isolated hardware/vendor
  validation gaps.
- `wave-2-dual-rate.md`: separated semantic, policy-inference, action-execution,
  and MCU-loop rates; public-source search converged without unchecked leads.
- `wave-2-action-chunk.md`
- `wave-2-coprocessors.md`

Open repository lead already being investigated by `retry_repo_all`:

- `origin/feat/scenario-driven-dse` contains a newer scenario-driven DSE stack absent from current `main`.

Direct wave-1 web digests:

- `wave-1-web-market-platforms.md`
- `wave-1-web-vla-models.md`
- `wave-1-system-boundary.md`
- `wave-1-repository-audit.md`
- `wave-1-china-market.md`
- `wave-1-market-platforms-saturation.md`
- `wave-1-vla-models-saturation.md`
- `wave-1-robot-products.md`

Boundary lead disposition:

- Vendor FMEDA/safety manuals: open product-specific gap, not expandable without a
  selected vendor and NDA access.
- Fault-injection recovery timing: open validation gap requiring target hardware.
- Exact SDK/firmware release matrix: deferred until a production platform is selected.
- Public architecture-pattern search: closed; two internal worker expansion waves
  converged on bounded asynchronous jobs, registered buffers, explicit fences, and
  independent safety supervision.

## Wave 3

Opened from Wave-1/Wave-2 leads:

- `wave3_china_deployments`: distinguish production robot deployments from
  demos, partnerships, and roadmaps.
- `wave3_china_coprocessors`: close official Chinese host-connected accelerator
  compute/memory/power/runtime gaps.
- `wave3_requirement_envelope`: independently check the four-profile
  coprocessor requirement envelope and sourced-versus-assumed values.

Completed:

- `wave-3-china-deployments.md`
- `wave-3-china-coprocessors.md`
- `wave-3-requirement-envelope.md`

Repository expansion is closed. Gated vendor specifications, FMEDAs, exact
target-board recovery timings, and cross-platform identical-workload benchmarks
are recorded as product-selection validation gaps because public research cannot
close them.

Wave-3 public-source leads converged. Remaining items require vendor/NDA
materials, unreleased products, target hardware, compiler measurements, or
application tensor traces.
