# Wave 1: China Physical-AI Edge Market

## Outcome

Domestic silicon spans roughly 6-TOPS low-cost SoCs through 80/128-TOPS robot
development platforms and higher automotive/physical-AI products. Publicly
documented shipping humanoid stacks still commonly use Jetson for the high-level
AI computer; the clearest domestic deployment found is AgiBot X2's base
RK3588S + RK3588 configuration.

## Coprocessor-relevant reference points

- RK3588: 6 TOPS, INT4/8/16 plus FP16/BF16/TF32, deployed in AgiBot X2 base.
- Huawei Atlas 200I A2: 8 or 20 INT8 TOPS, 4–12 GB ECC LPDDR4X,
  approximately 21–25 W module depending on variant.
- Axera AX8850 M.2: current 24 INT8 TOPS device, official M.2 below 8 W.
- Cambricon MLU220 M.2: 8 INT8 TOPS, PCIe 3.0 x2, 8.25 W passive.
- Cambricon MLU220 SOM: 16 INT8 TOPS, 8 GB, at most 15 W, standalone or
  coprocessor use.
- Black Sesame A1000: 58 INT8 TOPS, 18 W typical, but automotive SoC rather
  than a robot module.
- D-Robotics RDK S100/S100P: 80/128 TOPS development platform with CPU+BPU+MCU,
  12/24 GB LPDDR5; public board supply recommendations are not measured power.

## Shipping robot partition evidence

- AgiBot A2 separates the real-time "cerebellum" from an AGX Orin "brain".
- UBTECH Walker Tienkung uses a separate motion-control computer and one or two
  AGX Orin AI computers; published leg/arm loops are much faster than the VLA
  reasoning path.
- Unitree high-compute Orin/Thor options are SKU-dependent; they must not be
  assumed to be standard in every robot.
- XPeng IRON's three in-house Turing chips and 2250 effective TOPS remain
  pre-mass-production with an end-2026 target.

## Metric hygiene

- Do not rank dense INT8 TOPS against sparse/effective TOPS, equivalent TOPS,
  or FP4 TFLOPS without retaining the precision and sparsity tag.
- Tie Axera figures to the exact bin/revision; official older and current
  materials differ.
- Treat collaborations, demos, and development kits separately from named
  production robot deployments.

## EXPAND

- LEAD: Official BM1688/SE9 compute, power, and mechanical table.
- LEAD: Named production robot using Black Sesame A2000/C1200.
- LEAD: Named production robot using Axera AX8850/AX8910.
- LEAD: Vendor-primary compute disclosures for Fourier GR-1, Deep Robotics X30,
  and Leju Kuavo.
- LEAD: XPeng IRON final production power, memory, precision, and shipment status.
- LOW IMPACT: RDK X5 carrier-board mechanical drawing revision.

Full worker artifact:
`.omo/ultraresearch/20260723-china-physical-ai-chips/SYNTHESIS.md`.
