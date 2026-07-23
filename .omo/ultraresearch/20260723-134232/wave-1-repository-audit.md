# Wave 1: Repository and History Audit

## Outcome

Current `main` cannot yet execute a defensible embodied/VLA/physical-AI DSE.
The S3 result is documentary; the executable sweep remains a generic LLM/CV
model without an action/control contract or integrated CPU-NPU timing.

## Material findings

- `sim/config/scenarios.yaml` labels `onchip_7b` as VLM/VLA but defines only a
  Qwen2.5-7B token workload. It has no vision encoder, action head, policy rate,
  action horizon, diffusion/flow steps, or task DAG.
- `sim/design_space_explorer.py` on `main` does not apply scenario YAML hard
  constraints to the sweep.
- Frequency is swept but hard-coded conversion factors make it ineffective for
  LLM/CV performance.
- Qwen-VL and SD traces omit QK^T and attention-value GEMMs. Qwen-VL is missing
  at least 0.344 TMAC for four crops, an 11.76% minimum undercount.
- `cv_sim.py` treats unrecognized operation names as zero-cycle metadata.
  Qwen `layer_norm`, `softmax`, and `gelu`, plus major SD convolution/SFU
  operations, are therefore not timed.
- The S3 report's 20-token path is 9.9 FPS at 198 TPS, so changing only the
  array height cannot make it meet 10 FPS after decode has saturated.
- PCIe, software-overhead, and NPU timing models exist as disconnected pieces;
  host submission, transfer, cache maintenance, queueing, and completion do not
  participate in the current closed-loop result.

## Feature-branch finding

`origin/feat/scenario-driven-dse` at commit
`6288a4dd24926094c2f923af81fdfc4f18a5bdb9` improves scenario loading,
constraints, attention/SFU accounting, frequency, memory units, and capacity.
It still represents `warehouse_vla` as an LLM prompt/output workload:

- `sim/dse/types.py` is token/LLM oriented.
- `sim/dse/workload.py` rejects non-LLM workloads.
- Constraints cover token TPS/latency, area, power, and capacity, not action
  timing, multi-rate scheduling, boundary timing, or safety.

This branch is useful groundwork but is not yet an embodied workload model.

## Requirement gaps

- Submodel graph and per-stage precision/parameter/operator definitions
- Observation history and camera tensors per decision
- Policy refresh versus action execution frequency
- Action dimension, horizon/chunk length, and flow/diffusion steps
- Prefix caching and asynchronous action-queue behavior
- Sensor-to-action P50/P95/P99/WCET, jitter, staleness, miss policy
- CPU-NPU bytes/copies/fences/queues/deadlines
- Concurrent periodic jobs, priority, preemption, and resource isolation
- Safety watchdog/fallback/recovery and independent actuator gating
- Quality/accuracy/task-success gates after quantization

## Evidence

- `reports/arch-dse-three-scenarios.md`
- `sim/config/scenarios.yaml`
- `sim/dse_scenario.py`
- `sim/design_space_explorer.py`
- `sim/cv/traces/qwen_vl_vit_trace.py`
- `sim/cv/traces/sd_unet_trace.py`
- `sim/cv/cv_sim.py`
- `sim/models/sw_overhead.py`
- `sim/models/pcie.py`
- `sim/npu_sim.py`
- `docs/arc_vs_func.md`
- `verify-current-s3-workload.md`

## EXPAND

None. Working-tree, feature-branch, import/use, empirical trace, and git-history
passes converged without an unchecked repository lead.
