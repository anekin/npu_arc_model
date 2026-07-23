# Expansion Log: Deployed Robot Compute and Rates

## Phase 0

Core question: What first-party sources disclose about deployed humanoid, mobile, and
manipulation robots' onboard compute, power/battery, sensor-to-host boundary,
action/control rates, model latency, and CPU/accelerator division.

Axes:

1. Vendor platform specifications — Figure, Agility, Unitree, Fourier, UBTECH,
   Boston Dynamics, Apptronik, Sanctuary AI, PAL Robotics, Tesla, and 1X.
2. Research manipulation systems — Mobile ALOHA, ALOHA 2, Open X-Embodiment,
   RT-1/RT-2/RT-X, and representative deployed policies.
3. Real-time interfaces — official SDK manuals and repositories that disclose
   command/update frequencies or examples whose loop intervals are explicit.
4. Sensor-to-host boundary — location of perception compute, sensor buses,
   offboard training/inference, remote operation, and API boundaries.
5. Model execution — official papers and talks with policy frequency, inference
   latency, action chunking, and temporal aggregation.
6. Power and operating envelope — battery energy/capacity, voltage, runtime,
   charging, and hot-swap disclosures.

Codebase relevant: no. External: yes. Browsing: yes. Verification likely: source
cross-checking; no performance benchmarks will be inferred. Report requested: no.

Ground rules:

- Use official product pages, manuals, SDK repositories, first-party talks, and
  primary papers.
- Treat SDK example loop timing as an example rate, not a guaranteed platform limit.
- Mark fields as undisclosed when no first-party disclosure is found.
- Do not infer CPU/GPU partition from generic accelerator presence.

## Wave 1

- Opened: vendor product pages, SDKs, manuals, and primary research papers.
- Queries: recorded in `search-log.md`.
- Leads: Figure Helix execution stack; Digit runtime interfaces; Unitree low-level
  control timing; GR-1 SDK; Walker industrial products; Atlas/Spot compute;
  ALOHA policy timing; Open-X per-embodiment action rates.

Key boundary discovered: vendors frequently advertise accelerator partnerships
without disclosing the deployed SKU or workload split. Those mentions were not
promoted into configuration facts unless a first-party source tied the hardware to
a named robot generation.

## Wave 2

Expanded each high-value lead into implementation-level material:

- Figure: Helix, Helix 02, Figure 03 sensors and battery, logistics action chunks,
  and the walking controller.
- Agility: current NVIDIA and safety announcements, Digit sim-to-real article, and
  the 2021 real-Digit trajectory-planning paper.
- Unitree: product pages plus SHA-pinned SDK2 low-level examples for G1, H1, Go2,
  and B2.
- Fourier: GR2/GR1 specifications, Aurora SDK controller/status references, and
  the ARMOR primary paper plus first-party deployment note.
- UBTECH: legacy Walker, Walker X, Walker Tienkung developer documentation, and
  current Walker S2.
- Boston Dynamics: Atlas and Stretch brochures, Spot developer specifications,
  optional CORE I/O, image services, and Network Compute Bridge.
- Mobile ALOHA/Open-X: primary papers and SHA-pinned control/evaluation code.
- Additional products: 1X NEO and NEO Gamma, Apptronik Apollo generations, PAL
  TALOS/TIAGo Pro/KANGAROO, Sanctuary Phoenix, and Tesla Optimus.

New leads returned:

1. Product-generation drift is substantial: current Apollo 2, current NEO, Digit
   V5, and current Walker S2 must not inherit old-generation disclosures.
2. Some SDK frequencies are explicit interface/example loop values, not guaranteed
   neural-policy or actuator limits.
3. Mobile ALOHA has a primary-source battery inconsistency (paper prose versus
   Figure 2), which must be reported rather than resolved by inference.
4. Spot's battery energy differs between current developer and older support
   pages, likely because of revision; edition labels must be preserved.
5. Open X-Embodiment is a dataset/standardization effort, not one robot with a
   common onboard computer or battery.

## Convergence audit

The second wave added four distinct architecture patterns beyond the named vendor
set: a fully onboard dual-rate learned hierarchy (Figure), a fixed real-time
control PC plus optional AI accelerator (UBTECH/PAL), a network-selectable
perception worker (Spot), and research laptops/cloud serving (Mobile ALOHA/RT-X).
Follow-up searches on Sanctuary, Tesla, current Digit, current Walker S2, and stock
Fourier compute repeatedly returned marketing/deployment material without the
missing numeric configuration. No unsupported inference was used to fill those
gaps.

The evidence frontier has converged for public first-party material: further
progress on the remaining gaps requires non-public integration manuals, vendor
engineering confirmation, teardown evidence, or measured access to deployed
units.
