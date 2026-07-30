"""Test: DSE engine coverage matches factory.

Verifies that `generate_configs()` enumerates every engine type
that `create_engine()` supports, and that the quick-mode list is
exactly the expected three engines.  Uses the unified registry as
the single source of truth.
"""

from pathlib import Path

from engine.registry import canonical_engine_ids, quick_engine_ids

# ── helpers ──────────────────────────────────────────────────────


def _get_dse_engine_types(quick: bool) -> set:
    """Return the set of engine types enumerated by generate_configs."""
    from design_space_explorer import generate_configs  # type: ignore[import-untyped]

    cfgs = generate_configs(quick=quick)
    return {cfg["mac_engine"]["type"] for cfg in cfgs}


# ── tests ────────────────────────────────────────────────────────


class TestDseCoverage:
    """DSE engine-coverage contract tests."""

    def test_full_engine_list_matches_registry(self):
        """generate_configs(quick=False) must enumerate every engine
        type in the canonical registry."""
        registry_types = set(canonical_engine_ids())
        dse_types = _get_dse_engine_types(quick=False)

        assert registry_types == dse_types, (
            f"Registry has {sorted(registry_types)}, but DSE generates {sorted(dse_types)}"
        )
        # Sanity: we know there are exactly 8 engines today
        assert len(dse_types) == 8

    def test_quick_mode_engine_set(self):
        """generate_configs(quick=True) must cover exactly the three
        priority engines: systolic, block, gmma."""
        dse_types = _get_dse_engine_types(quick=True)
        expected = set(quick_engine_ids())

        assert dse_types == expected, f"Quick mode generated {sorted(dse_types)}, expected {sorted(expected)}"
        assert len(dse_types) == 3

    def test_pytest_ini_exists_with_testpaths(self):
        """Verify pytest.ini exists at the repo root and configures
        testpaths = sim/tests."""
        # test file lives at  sim/tests/test_dse_coverage.py
        repo_root = Path(__file__).resolve().parent.parent.parent
        ini = repo_root / "pytest.ini"

        assert ini.exists(), f"pytest.ini not found at {ini}"
        content = ini.read_text(encoding="utf-8")
        assert "testpaths = sim/tests" in content, f"pytest.ini missing 'testpaths = sim/tests'; got:\n{content}"
