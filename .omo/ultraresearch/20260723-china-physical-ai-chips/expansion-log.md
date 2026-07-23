# Expansion log

## Phase 0

- Core question: Which China-origin edge-AI chips/modules are usable for embodied-intelligence / physical-AI, what are their verified technical characteristics and product status, and which robot vendors publicly disclose using them?
- Axis A — Horizon, Rockchip: official product/developer documentation, modules, SDK/model support, deployment disclosures.
- Axis B — Ascend, Sophgo: official datasheets/developer portals, accelerator modules/cards, CPU/NPU partition, robot deployments.
- Axis C — Black Sesame, Axera, Cambricon: official product/news/filing sources, edge modules, product availability, deployments.
- Axis D — robot OEM stacks: official OEM product pages, manuals, filings, conference material, government/academic primary sources.
- Axis E — contradictions/rumors: compare official primary evidence against secondary claims and flag unsupported specifications/deployments.
- External: yes. Browsing: yes. Codebase: no. Verification by execution: not applicable to unavailable proprietary silicon; cross-source documentary verification is required.
- Requested output: source-cited synthesis with an `## EXPAND` tail.

## Query log

### Search waves

1. Current vendor product pages and downloadable briefs.
2. SDK/toolchain documentation for operator coverage, model conversion, and CPU/NPU fallback.
3. Company filings and launch notices for mass-production versus announced/sample status.
4. Robot-vendor manuals and product pages for onboard-compute partitioning.
5. Contradiction checks across current pages, older PDFs, Chinese/English localizations, and precision conventions.
6. Negative-evidence searches for commonly repeated but unsupported robot-chip pairings.

Executed more than 25 search batches (about 100 varied query strings), primarily restricted to
vendor domains, official documentation portals, stock-exchange filings, and official robot manuals.

## Expanded leads closed

- D-Robotics RDK X5 module lifecycle and RDK S100 CPU/BPU/MCU architecture.
- Rockchip RK3588/RK3576 precision support and RKNN conversion/runtime workflow.
- Ascend Atlas 200I A2 module-versus-developer-kit power and ATC `.om` conversion.
- Sophgo BM1684X/BM1688 SoC-versus-PCIe execution modes and TPU-MLIR deployment flow.
- Black Sesame A1000 production evidence and A2000 announcement/sample/commercialization timeline.
- Axera AX8850/AX650 conflicting compute figures and current M.2 accelerator power/form factor.
- Cambricon MLU220 SOM/M.2 specifications and cumulative shipment evidence.
- AgiBot, UBTECH, Unitree, XPeng, LimX, Fourier, Leju, and Deep Robotics disclosures.

## Evidence rules applied

- Kept raw INT8 TOPS, effective/equivalent TOPS, INT4 TOPS, and FP4 TFLOPS separate.
- Treated power-supply sizing as distinct from measured or specified device power.
- Treated optional developer computers as options, not standard robot configurations.
- Treated ecosystem agreements and demonstrations as non-deployment unless the robot SKU is named.
- Marked current-page/older-PDF and Chinese/English localization conflicts instead of averaging them.

## Remaining open leads

- Official BM1688/SE9 public TOPS and module-power table.
- Exact production SKU using Black Sesame A2000 or C1200 inside a named robot.
- Exact production robot SKU using Axera AX8850/AX8910.
- Vendor-primary compute disclosure for Fourier GR-1 and Deep Robotics X30.
- Reconciliation of D-Robotics RDK X5 carrier-board mechanical dimensions across current page/PDF.
