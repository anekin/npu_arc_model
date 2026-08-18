# lift-decision-grade-fail — Problems

## 2026-07-31 — Todo 3: DRAM efficiency calibration entries

No blocking problems encountered. Notable friction points (all worked around):

1. **JEDEC standard page not machine-fetchable** — `jesd209-5b` returns 403 to automated fetch.
   Worked around by using it as the canonical `source_uri` anchor anyway and cross-checking the
   numeric claims (tRFCpb=140ns, tREFI=3900ns, tRC~48ns) against repo comments in
   `sim/contracts/hardware.py` and `sim/models/dram.py` rather than the live page.
2. **Exa web search rate-limited** on this server, so source-URI verification for the random-access
   study anchor had to fall back to direct fetches (arXiv:2012.03112 verified reachable, 5MB PDF).
3. **Notepad files were created concurrently by a parallel Wave-1 agent (Todo 2)** — Write tool
   refuses notepad paths (append-only policy), so Todo 3 findings were appended with Edit and the
   two remaining files created via shell.
