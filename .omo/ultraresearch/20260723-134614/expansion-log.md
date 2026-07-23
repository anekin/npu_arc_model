# Expansion log

Access date: 2026-07-23

## Phase 0

Core question: What are the current official edge-compute platforms for robots and physical AI from NVIDIA, Qualcomm, TI, Hailo, Ambarella, NXP, and Renesas, and what engineering constraints can be put into a design-space exploration model?

Axes:

- Compute architecture: CPU, GPU, NPU/DLA, advertised precision and TOPS basis.
- Deployable module constraints: power mode or TDP, memory capacity/type/bandwidth, I/O, environmental and life-cycle constraints.
- Concurrency and safety: independently usable accelerators, virtualization/isolation, functional-safety evidence, safety domains.
- Workload fit: robotics foundation models, perception, sensor fusion, planning, control, industrial vision, autonomous driving.
- DSE normalization: measurable variables, non-comparable metrics, missing disclosures, hard constraints, and marketing caveats.
- Currency: currently marketed products and current official documentation as of 2026-07-23.

Codebase relevant: no. External: yes. Browsing: yes. Verification likely: source cross-checking rather than executable benchmarking. Report requested: Markdown synthesis to parent.

## Wave 1

Direct official-source saturation across all seven vendors; 15+ varied search queries required.

Result: more than 60 query variants were run across vendor product pages, newsrooms,
datasheets, technical briefs, power notes, and safety documents. The first pass established
the active/current families and exposed a cross-vendor normalization problem: NVIDIA mixes
FP4-sparse and INT8-sparse peaks, Qualcomm and Renesas publish both dense and sparse figures
on their newest parts, Ambarella and NXP use proprietary "equivalent TOPS" on some products,
and several vendors do not publish a fixed SoC power envelope.

Primary product groups retained:

- NVIDIA Jetson Thor, Jetson Orin, IGX Orin, and IGX Thor.
- Qualcomm Dragonwing IQ10/IQ9/IQ8/IQ-X, RB3 Gen 2, RB5, and Snapdragon Ride.
- TI TDA4VH/TDA4VM/TDA4VE and AM69A.
- Hailo-10H, Hailo-8/8L, and Hailo-15H.
- Ambarella CV7/CV52S/CV3 and N1/N1-655.
- NXP i.MX 95/i.MX 8M Plus/i.MX 952, Ara240, and S32N55 as a control companion.
- Renesas RZ/V2H/RZ/V2N/RZ/V2L and R-Car V4H.

## Wave 2: metric-normalization expansion

Lead: headline TOPS were not comparable enough for design-space exploration.

Actions and findings:

- Opened NVIDIA architecture briefs to distinguish FP4 sparse Thor figures from INT8
  sparse Orin figures and to separate integrated GPU from optional discrete-GPU totals.
- Located Qualcomm's explicit IQ10 split: 700 sparse versus 350 dense INT8 TOPS.
- Located Renesas datasheets with the explicit RZ/V2H 8 dense / 80 sparse and RZ/V2N
  4 dense / 15 sparse split; the sparse value assumes model pruning.
- Verified Ambarella's CV3 headline is 500 "eTOPS" and that no official conversion to
  conventional dense TOPS is published.
- Added NXP Ara240's explicit 40 eTOPS, 6.5 W typical, and model-throughput figures so
  it can be modeled by measured workload throughput rather than eTOPS alone.

Convergence: the additional searches changed caveats and normalization, but did not reveal
a common vendor-independent TOPS basis. The synthesis therefore preserves precision and
sparsity as first-class fields and rejects bare-TOPS ranking.

## Wave 3: deployability and safety expansion

Lead: SoC peak compute alone does not determine a robot-compute architecture.

Actions and findings:

- Found the June 2026 Qualcomm IQ10 Robotics Reference Design brief: 64 GB LPDDR5X,
  forced-air cooling, 12/24 V input, 12 GMSL2 cameras, and 12 deterministic Ethernet/CAN
  links; no input-power maximum or DRAM bandwidth is disclosed.
- Re-opened the live IQ-9075 page after the initial search result proved incomplete. It
  explicitly publishes 50/100 dense INT8 TOPS SKUs, up to 36 GB across six 16-bit LPDDR5
  channels, a four-core real-time subsystem, and up to 16 concurrent cameras.
- Used NXP's February 2026 i.MX 95 power note to obtain rail-measured workload values
  (about 1.19 W idle, 2.79 W NPU workload, and 5.15 W heterogeneous stress), explicitly
  not board TDPs.
- Confirmed TI supplies use-case power-estimation tools rather than fixed TDPs for
  TDA4VH/AM69A; low-power modes and power-domain sequencing are documented.
- Opened NVIDIA IGX safety material to separate development-process claims
  (ISO 26262 ASIL D / IEC 61508 SIL 3) from the Thor SoC's random-hardware integrity
  (ASIL B / SIL 2) and the independent safety island's higher target.
- Confirmed NXP S32N55 and the Qualcomm/TI/Renesas real-time islands are control and
  freedom-from-interference resources, not additive AI TOPS.
- Rechecked Ambarella CV7 and N1 official material. CV7 remains non-DSE-ready for absolute
  AI throughput, memory, and watts; N1-655 discloses a useful under-20-W workload envelope
  but not TOPS or DRAM specifications.

Convergence: a final targeted pass produced two material Qualcomm updates (IQ10 RRD memory
and fuller IQ9 specifications) and otherwise confirmed the public-data gaps. Further search
is unlikely to close those gaps; vendor NDA material or board measurement is required.

## Stop condition

Stop after three waves. All named vendors and requested fields were covered, the required
query count was exceeded, two lead-driven expansion passes converged, and remaining gaps are
explicitly non-public rather than undiscovered in current official material.
