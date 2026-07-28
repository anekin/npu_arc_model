"""Test: DSE engine coverage matches factory.

Verifies that `generate_configs()` enumerates every engine type
that `create_engine()` supports, and that the quick-mode list is
exactly the expected three engines.
"""

import inspect
import re
from pathlib import Path

import pytest


# ── helpers ──────────────────────────────────────────────────────


def _get_create_engine_supported_types() -> set:
    """Dynamically extract all `engine_type == "..."` branches
    from the `create_engine` factory function."""

    from engine.mac_engine import create_engine  # type: ignore[import-untyped]

    src = inspect.getsource(create_engine)
    # matches lines like:  elif engine_type == "systolic":
    return set(re.findall(r'engine_type == "(\w+)"', src))


def _get_dse_engine_types(quick: bool) -> set:
    """Return the set of engine types enumerated by generate_configs."""
    from design_space_explorer import generate_configs  # type: ignore[import-untyped]

    cfgs = generate_configs(quick=quick)
    return {cfg["mac_engine"]["type"] for cfg in cfgs}


# ── tests ────────────────────────────────────────────────────────


class TestDseCoverage:
    """DSE engine-coverage contract tests."""

    def test_full_engine_list_matches_factory(self):
        """generate_configs(quick=False) must enumerate every engine
        type that create_engine() can instantiate."""
        factory_types = _get_create_engine_supported_types()
        dse_types = _get_dse_engine_types(quick=False)

        assert factory_types == dse_types, (
            f"Factory supports {sorted(factory_types)}, "
            f"but DSE generates {sorted(dse_types)}"
        )
        # Sanity: we know there are exactly 8 engines today
        assert len(dse_types) == 8

    def test_quick_mode_engine_set(self):
        """generate_configs(quick=True) must cover exactly the three
        priority engines: systolic, block, gmma."""
        dse_types = _get_dse_engine_types(quick=True)

        assert dse_types == {"systolic", "block", "gmma"}, (
            f"Quick mode generated {sorted(dse_types)}, "
            "expected {'systolic', 'block', 'gmma'}"
        )
        assert len(dse_types) == 3

    def test_pytest_ini_exists_with_testpaths(self):
        """Verify pytest.ini exists at the repo root and configures
        testpaths = sim/tests."""
        # test file lives at  sim/tests/test_dse_coverage.py
        repo_root = Path(__file__).resolve().parent.parent.parent
        ini = repo_root / "pytest.ini"

        assert ini.exists(), f"pytest.ini not found at {ini}"
        content = ini.read_text(encoding="utf-8")
        assert "testpaths = sim/tests" in content, (
            f"pytest.ini missing 'testpaths = sim/tests'; got:\n{content}"
        )
