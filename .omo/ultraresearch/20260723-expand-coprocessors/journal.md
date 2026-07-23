# Discrete host-connected NPU coprocessors — research journal

Date: 2026-07-23

Scope: M.2, PCIe, USB, or endpoint-capable accelerator modules/cards for
physical and embodied AI. Integrated sensor SoCs were excluded from the core
set; endpoint-capable SoC cards/modules are marked borderline.

## Candidate set

- Hailo-10H; Hailo-8/8L as a current vision baseline
- Huawei Atlas 300I Pro / 300I Duo; Atlas 200I A2 as a borderline MXM endpoint
- Cambricon MLU220 M.2; MLU270-S4 as an older PCIe baseline
- Axelera Metis M.2, M.2 Max, and PCIe cards
- MemryX MX3 M.2
- DEEPX DX-M1 / DX-M1M and DX-H1 Quattro
- NXP/Kinara Ara240 M.2 and USB module
- EdgeCortix SAKURA-II M.2 and PCIe cards
- SiMa.ai first-generation MLSoC and Modalix PCIe as borderline SoC cards
- Coral Edge TPU as a legacy constrained-model baseline

## Key cross-vendor findings

- The strongest compact local-memory offerings found were SAKURA-II M.2
  (8/16 GB, 68 GB/s), Ara240 M.2 (up to 16 GB, raw 34.1 GB/s for the 64-bit
  configuration), Hailo-10H (4/8 GB, approximately 17 GB/s theoretical), and
  Metis M.2 Max (2/8 GB; bandwidth not numerically published).
- Public LLM results are sparse and frequently not comparable. The most useful
  vendor figures are Hailo-10H Qwen2-1.5B at 9.45 token/s and 289 ms TTFT for
  96 input tokens at 2.1 W average, and Ara240 Llama2-7B at 14 output token/s.
- Queue and error semantics are unevenly documented. Ascend, MemryX, DEEPX,
  HailoRT, and SiMa expose asynchronous execution and status/error mechanisms.
  Ara240 identifies hardware task queues/notifications. Axelera, Cambricon, and
  EdgeCortix public product material does not disclose queue depth or a
  production error ABI.
- Availability labels matter: Hailo-10H is generally available; Ara240 M.2 is
  active but its USB module is preproduction; Modalix documentation still says
  early access despite a 2026 press-release GA timetable; SAKURA-II 16 GB and
  PCIe products are sold as trial units; Metis M.2 Max is a preliminary 2026
  design; MLU220 and Coral remain listed but are legacy architectures.

## Evidence handling

- Power is labeled chip, module, card, typical, TDP, or maximum where the
  source distinguishes it.
- Host-link bandwidth and local-memory bandwidth are kept separate.
- Peak TOPS, effective TOPS, and TFLOPS are not normalized across vendors.
- Vendor throughput is reported with model, resolution, precision, and
  end-to-end qualification when the source provides them.
- Missing public queue depth, memory bandwidth, and error-reporting details are
  reported as gaps instead of inferred.
