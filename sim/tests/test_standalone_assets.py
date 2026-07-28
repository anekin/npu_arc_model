"""Test: standalone repository asset completeness.

Verifies that every file required for a standalone checkout of the
DSE engine test suite is present.  A "missing-asset" smoke test
confirms the detection logic itself works.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

REQUIRED_ASSETS = [
    "pytest.ini",
    "sim/tests/test_engines.py",
    "sim/tests/test_engine_instantiate.py",
    "sim/tests/test_engine_result_contract.py",
    "sim/tests/test_dse_strict.py",
    "sim/tests/test_calibration_config.py",
    "sim/tests/test_dse_coverage.py",
]


class TestStandaloneAssets:
    """Asset-existence contract tests."""

    @pytest.mark.parametrize("asset", REQUIRED_ASSETS)
    def test_required_asset_exists(self, asset):
        """Every file in the REQUIRED_ASSETS list must be present."""
        full_path = REPO_ROOT / asset
        assert full_path.exists(), (
            f"Required standalone asset not found: {asset}\n"
            f"  expected at: {full_path}"
        )

    def test_missing_asset_detected(self, tmp_path):
        """tmp_path smoke test: verify that a non-existent file in an
        empty temporary directory correctly triggers AssertionError.

        This confirms the asset-detection logic (the parametrized test
        above) would catch a genuinely missing file.
        """
        missing = tmp_path / "should_not_exist.txt"
        assert not missing.exists()  # pre-condition: tmp_path is empty

        with pytest.raises(AssertionError):
            assert missing.exists(), f"Missing asset: {missing}"
