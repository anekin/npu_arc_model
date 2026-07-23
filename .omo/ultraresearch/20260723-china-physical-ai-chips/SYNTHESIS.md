# China physical-AI edge compute — evidence synthesis

Accessed 2026-07-23. This file records the source-backed conclusions sent to the parent
agent; it is not a substitute for the linked primary documents.

## Strongest production-ready domestic platforms

- D-Robotics RDK X5 module: 10 TOPS BPU, eight Cortex-A55 CPUs, 2/4/8 GB LPDDR4,
  55 × 40 mm, 300-pin connector, published lifecycle through at least 2031.
- Huawei Atlas 200I A2: 8 or 20 TOPS INT8 module, 4–12 GB LPDDR4X ECC depending on
  SKU, 82 × 60 mm MXM form factor, specified 21–25 W.
- Cambricon MLU220 SOM: 16 TOPS INT8, four A55 CPUs, 8 GB LPDDR4X, 32 GB eMMC,
  50 × 66 mm, no more than 15 W; MLU220 family cumulative sales exceed one million.
- Black Sesame A1000: mass-produced automotive SoC, 58 TOPS INT8, eight A55 CPUs,
  64-bit LPDDR4, typical 18 W, 25 × 25 mm package.
- Axera AX8850 M.2: current 24 TOPS INT8 silicon; official M.2 2242/2280 accelerator
  is specified below 8 W.
- Rockchip RK3588/RK3576: 6 TOPS NPUs with broad integer/floating precision support;
  Rockchip publishes SoCs/toolchains rather than a first-party standardized robot SOM.

## Robot-stack finding

The clearest production robot disclosure using a domestic edge SoC is AgiBot X2's base
RK3588S + RK3588 computer. Its higher-compute/developer configuration uses Jetson Orin
NX 16 GB. UBTECH Walker Tienkung and AgiBot A2 explicitly separate x86/control or
"cerebellum" computers from Jetson AGX Orin AI/"brain" computers. Unitree publishes
Orin options on several products, but often does not make them the standard base
configuration. XPeng IRON is the notable vertically integrated exception, using three
in-house Turing AI chips, but remains targeted for mass production at the end of 2026.

## Status caveats

- Black Sesame A2000 samples and commercialization announcements are real, but no
  official source found names a shipping robot SKU containing it.
- Axera's embodied-intelligence demos and partner ecosystem do not prove a named
  production robot deployment.
- Sophgo's public sources support edge-server/module readiness but not a named robot OEM.
- Huawei's CloudMinds collaboration concerns cloud-edge-end infrastructure and does not
  establish that Atlas 200I is installed in a particular commercial robot.
- Fourier GR-1 and Deep Robotics X30 official pages do not disclose the onboard processor.
