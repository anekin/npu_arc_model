# cross-node-all-engines-dse — Learnings

## 2026-07-31: Execution Complete

### What worked
- **Engine × process_node cross-product in ci-all-axes**: Added ~25 LOC to `_ci_all_axes_combinations()` that generates 7 non-block engines × 3 non-default nodes = 21 additional combos. The key insight was separating this from the existing axis iteration loop to avoid ambiguity.
- **All 10 new tests pass**: test_dse_cross_node_coverage.py covers all 4 nodes with all 8 engines, negative tests for invalid nodes, and frequency constraint application.
- **Both DSE runs succeeded**: lpddr5_3b (87 points, 87 complete, 0 failed) and onchip_7b (86 points, 86 complete, 0 failed). All 8 engines appear in both datasets.
- **Ranking matrix**: 64/64 cells filled — no constraint-filtered or missing entries. Every engine has data for every node in both scenarios.
- **Generalized standalone script**: Extended investigate-fsa-cross-node-freq.py to all 8 engines via `engine_full_ids()`. 112 combos evaluated, 32 best results. os_systolic is the top performer across all nodes.
- **Documentation updated**: model-trust-and-release.md now has full 8-engine tables. README has updated cross-node column and new insight #6.
- **F1-F4 all PASS**: 0 failures across plan compliance, code quality, release gate, and scope fidelity.

### Key finding
**os_systolic is the absolute leader across all nodes and scenarios.** This was not previously known — the earlier P0 comparison only covered block and FSA. os_systolic achieves 31.8 tok/s (lpddr5_3b) and 310.9 tok/s (onchip_7b), dominating every competitor at every process node. GMMA is competitive at high BW (203.5 tok/s @7nm onchip) but falls off at older nodes (97.7 tok/s @28nm).

### Process notes
- The v2 result schema doesn't store axis_values directly — had to regenerate DesignSpace to map design_point_ids to (process_node, engine) pairs.
- The `--result-schema v2` flag is required for --scenario mode DSE; the default "v1" raises ConfigError.
- The `--output` path with relative paths resolves under `sim/` directory, not repo root — absolute paths are safer.
- The `build_ranking_matrix.py` script is a useful artifact for future ranking analyses.

### Commits
- `6833b9c` feat(dse): add engine × process_node cross-product in ci-all-axes mode
- `91083b0` evidence(dse): complete 8-engine × 4-node DSE runs and ranking matrix
- `f27d786` evidence(dse): generalize cross-node standalone to all 8 engines
- `d7ab1f1` docs(dse): update cross-node conclusions with full 8-engine ranking
