# ruff: noqa: E402
"""Tests for the model-integrity verifier."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_model_integrity import (
    check_duplicate_registries,
    check_unregistered_calibration_constants,
    counterexample_unknown_op_zero_cycle,
    counterexample_utilization_bounds,
    find_diagnostic_skips,
    find_legacy_bandwidth_reads,
    find_utilization_clamps,
    verify_model_integrity,
)


class TestAstChecks:
    """AST-based static checks."""

    def test_no_utilization_clamps(self):
        files = [
            Path(__file__).resolve().parent.parent / "engine" / f
            for f in ("systolic_engine.py", "block_engine.py", "gmma_engine.py")
        ]
        clamps = find_utilization_clamps(files)
        assert not clamps, clamps

    def test_no_legacy_bandwidth_reads_in_engines(self):
        files = [
            Path(__file__).resolve().parent.parent / "engine" / f
            for f in ("systolic_engine.py", "block_engine.py", "gmma_engine.py")
        ]
        reads = find_legacy_bandwidth_reads(files)
        assert not reads, reads

    def test_no_diagnostic_skips(self):
        files = list((Path(__file__).resolve().parent).glob("test_*.py"))
        skips = find_diagnostic_skips(files)
        assert not skips, skips


class TestRegistryChecks:
    """Registry introspection checks."""

    def test_no_duplicate_registries(self):
        assert not check_duplicate_registries()

    def test_no_unregistered_calibration_constants(self):
        assert not check_unregistered_calibration_constants()


class TestCounterexamples:
    """Focused counterexample execution."""

    def test_unknown_op_raises(self):
        assert not counterexample_unknown_op_zero_cycle()

    def test_utilization_stays_within_bounds(self):
        assert not counterexample_utilization_bounds()


class TestEndToEnd:
    """End-to-end model integrity report."""

    def test_integrity_report_passes(self):
        report = verify_model_integrity(PROJECT_ROOT)
        assert report["verdict"] in ("PASS", "FAIL")
        assert isinstance(report["violations"], list)
        assert isinstance(report["utilization_clamps"], list)
        assert isinstance(report["legacy_unit_reads"], list)
        assert isinstance(report["unregistered_constants"], list)

    def test_utilization_bounds_category_present(self):
        report = verify_model_integrity(PROJECT_ROOT)
        assert "utilization_bound_violations" in report
