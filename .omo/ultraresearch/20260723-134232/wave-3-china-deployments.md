# Wave 3: China Robot Deployment Verification

## Outcome

Public first-party sources do not prove a named shipping robot SKU using
Black Sesame A2000/C1200/C1236, Axera AX8850, or AX8910. The strongest public
evidence is strategic cooperation/prototype, trade-show dynamic demonstration,
or a chip-vendor reference design. These must not be labeled production BOMs.

## Verified distinctions

- Black Sesame A2000+C1236 with Wuhan University's Tianwen is an active
  brain/cerebellum cooperation and prototype direction, not disclosed volume
  production.
- Deep Robotics X30 is a commercial deployed quadruped; Black Sesame publicly
  demonstrated an X30 carrying its silicon, but did not identify the chip SKU
  or say the customer production product uses it.
- Axera publishes an AX8910 perception + AX8850 motion/AI reference architecture,
  but no named production robot OEM deployment was found.
- AX8850 M.2 below 8 W is a valid host-connected coprocessor reference even
  though robot shipment evidence is not established.
- Fourier GR-1 is a production robot whose public manual lists an Intel host,
  but no first-party source proves a production Black Sesame accelerator.
- Leju KUAVO publicly distinguishes brain and motion-control cerebellum but does
  not disclose the exact NPU/GPU SKU or TOPS.

## Requirement consequences

- Preserve `deployment_status`: production BOM, shipping optional module,
  development kit, reference design, demo, partnership, or roadmap.
- Preserve exact SKU, precision, sparsity, document revision, power scope, and
  local memory; otherwise keep the value unknown.
- Public robot evidence supports asynchronous slow-brain/fast-policy partitioning
  and a CPU-hosted optional accelerator, not a single VLM FPS.

## EXPAND

- VENDOR GAP: Exact SesameX module precision, memory, power, and host interface.
- NDA GAP: Production BOMs for X30, KUAVO, and GR-1.
- DEAD END: No public first-party proof of a shipping named robot with the
  target A2000/C12xx/AX8850/AX8910 SKU.
