# Wave 1: Shipping Robot Products

## Outcome

Across 58 first-party/primary sources, published robot frequencies describe
different hierarchy levels rather than one comparable "robot Hz":

- semantic/VLA policies are commonly around 3–15 Hz;
- learned joint-target/reactive policies are around 50–250 Hz;
- state, EtherCAT, and actuator loops are around 500 Hz–2 kHz.

The last tier generally belongs to motion computers/real-time controllers, not
the high-level NPU coprocessor.

## Representative evidence

- Figure Helix: 7B S2 at 7–9 Hz, 80M S1 at 200 Hz; Helix 02 adds 10M S0
  at 1 kHz, with exact accelerator SKUs undisclosed.
- Agility Digit: separate safety PLC; historical research planner at 250 Hz,
  feedback at 1 kHz, and low-level link at 2 kHz, not a current product claim.
- Unitree SDK examples use 500 Hz low-command loops; this is example timing,
  not guaranteed neural-policy inference.
- Fourier Aurora: 500 Hz state, 50 Hz lower-body learned locomotion,
  500 Hz upper-body state management.
- UBTECH Tienkung: separate i7 motion computer and optional AGX Orin modules,
  at least 1 kHz leg and 400 Hz arm loops.
- PAL TALOS/TIAGo: 1 kHz real-time control, with optional/separate Jetson-class
  compute depending on product.
- Mobile ALOHA: 50 Hz action loop and onboard RTX laptop, but loop cadence
  does not prove sustained 50-Hz model inference.

## Requirement consequences

- Store semantic, learned-policy, action-output, state, and actuator rates as
  separate fields.
- Record exact product generation and optional-module status.
- Never inherit an older generation's compute SKU into a current product.
- Treat SDK command cadence as distinct from delivered actuator cadence and
  sustained model inference.
- Battery/system-runtime figures are system context, not NPU-only power.

## EXPAND

- VENDOR/NDA GAP: Current exact compute for Digit, Walker S2, Atlas, stock GR2,
  Sanctuary Phoenix, and Optimus.
- HARDWARE GAP: End-to-end camera-to-action latency on deployed hardware.
- VERSION GAP: Current Apollo 2 topology and product-revision-specific options.
- DEAD END: Public product material rarely discloses both neural latency and
  controller timing for the same configuration.

Full worker artifact:
`.omo/ultraresearch/20260723-134658-robot-products/SYNTHESIS.md`.
