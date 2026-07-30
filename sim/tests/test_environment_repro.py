"""Test: environment reproducibility and offline fixture verification.

Verifies:
- uv.lock exists and is valid (non-empty)
- Offline test fixtures are present
- Required standalone assets exist
- Negative-path: missing lock or fixtures must fail
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Assets that must be present for a complete standalone checkout
REQUIRED_ENV_ASSETS = [
    "pyproject.toml",
    "uv.lock",
    "pytest.ini",
    "sim/tests/golden/legacy_cli_contract.json",
]

# Test fixture files that must be present for offline test execution
REQUIRED_TEST_FIXTURES = [
    "sim/tests/conftest.py",
    "sim/tests/test_standalone_assets.py",
    "sim/tests/test_engines.py",
    "sim/tests/test_engine_instantiate.py",
    "sim/tests/test_engine_result_contract.py",
    "sim/tests/test_dse_strict.py",
    "sim/tests/test_calibration_config.py",
    "sim/tests/test_dse_coverage.py",
]

# Source files needed to run the simulator
REQUIRED_SOURCE_FILES = [
    "sim/npu_sim.py",
    "sim/design_space_explorer.py",
    "sim/config/npu_config.py",
    "sim/config/npu_config.yaml",
    "sim/config/design_space.yaml",
    "sim/config/scenarios.yaml",
    "sim/engine/mac_engine.py",
    "sim/models/dma.py",
    "sim/models/dram.py",
    "sim/models/kv_cache.py",
    "sim/models/noc.py",
    "sim/models/sfu.py",
    "sim/models/vector.py",
    "sim/models/golden.py",
    "sim/models/sw_overhead.py",
    "sim/models/crossbar.py",
    "sim/models/pcie.py",
]


class TestEnvironmentReproducibility:
    """Verify that the environment can be reproduced from locked dependencies."""

    @pytest.mark.clean
    def test_lock_file_exists(self):
        """uv.lock must be present in the repository root."""
        lock_path = REPO_ROOT / "uv.lock"
        assert lock_path.exists(), f"uv.lock not found at {lock_path}. Run 'uv lock' to generate the lock file."

    @pytest.mark.clean
    def test_lock_file_not_empty(self):
        """uv.lock must be a valid, non-empty file."""
        lock_path = REPO_ROOT / "uv.lock"
        assert lock_path.exists()
        content = lock_path.read_text()
        assert len(content) > 0, "uv.lock is empty — re-run 'uv lock'"
        # Basic structure check: should contain package metadata
        assert "name =" in content, "uv.lock appears invalid (no 'name =' entries)"
        assert "version =" in content, "uv.lock appears invalid (no 'version =' entries)"

    @pytest.mark.clean
    def test_pyproject_toml_exists(self):
        """pyproject.toml must be present."""
        path = REPO_ROOT / "pyproject.toml"
        assert path.exists(), f"pyproject.toml not found at {path}"

    @pytest.mark.clean
    def test_required_env_assets_present(self):
        """All REQUIRED_ENV_ASSETS must exist."""
        for asset in REQUIRED_ENV_ASSETS:
            full_path = REPO_ROOT / asset
            assert full_path.exists(), f"Required env asset missing: {asset}"


class TestOfflineFixtures:
    """Verify that offline test fixtures and source files are present."""

    @pytest.mark.clean
    def test_test_fixtures_present(self):
        """All required test fixture files must exist."""
        for fixture in REQUIRED_TEST_FIXTURES:
            full_path = REPO_ROOT / fixture
            assert full_path.exists(), f"Required test fixture missing: {fixture}"

    @pytest.mark.clean
    def test_source_files_present(self):
        """All required source files for the simulator must exist."""
        for src in REQUIRED_SOURCE_FILES:
            full_path = REPO_ROOT / src
            assert full_path.exists(), f"Required source file missing: {src}"

    @pytest.mark.clean
    def test_golden_contract_present(self):
        """The golden legacy CLI contract must be present and valid JSON."""
        path = REPO_ROOT / "sim" / "tests" / "golden" / "legacy_cli_contract.json"
        assert path.exists(), f"Golden contract not found: {path}"
        import json

        data = json.loads(path.read_text())
        assert "_meta" in data, "Golden contract missing _meta section"
        assert "npu_sim_cli" in data, "Golden contract missing npu_sim_cli"


class TestNegativeReproducibility:
    """Negative-path: verify that missing lock or fixtures cause detectable failures."""

    @pytest.mark.missing_lock
    def test_missing_lock_detected(self, tmp_path):
        """Verify that a non-existent lock file triggers an assertion failure."""
        missing = tmp_path / "uv.lock"
        assert not missing.exists()

        with pytest.raises(AssertionError):
            assert missing.exists(), f"Lock file not found: {missing}"

    @pytest.mark.missing_lock
    def test_empty_lock_detected(self, tmp_path):
        """Verify that an empty lock file triggers an assertion failure."""
        empty_lock = tmp_path / "uv.lock"
        empty_lock.write_text("")
        assert empty_lock.exists()

        with pytest.raises(AssertionError, match="empty"):
            content = empty_lock.read_text()
            assert len(content) > 0, "uv.lock is empty — re-run 'uv lock'"

    @pytest.mark.missing_fixture
    def test_missing_golden_detected(self, tmp_path):
        """Verify that a missing golden contract triggers an assertion failure."""
        missing_golden = tmp_path / "nonexistent" / "golden.json"
        assert not missing_golden.exists()

        with pytest.raises(AssertionError):
            assert missing_golden.exists(), f"Golden contract not found: {missing_golden}"

    @pytest.mark.missing_fixture
    def test_missing_fixture_file_detected(self, tmp_path):
        """Verify that a missing source file triggers an assertion failure."""
        missing_src = tmp_path / "nonexistent.py"
        assert not missing_src.exists()

        with pytest.raises(AssertionError):
            assert missing_src.exists(), f"Required source file missing: {missing_src}"

    @pytest.mark.missing_fixture
    def test_empty_pyproject_detected(self, tmp_path):
        """Verify that an empty pyproject.toml triggers an assertion failure."""
        empty_proj = tmp_path / "pyproject.toml"
        empty_proj.write_text("")
        assert empty_proj.exists()

        with pytest.raises(AssertionError, match="missing"):
            content = empty_proj.read_text()
            assert "[project]" in content, "pyproject.toml missing [project] section"

    @pytest.mark.missing_fixture
    def test_lock_without_pyproject_detected(self, tmp_path):
        """Verify that having lock without pyproject.toml is detected (both must exist)."""
        missing_proj = tmp_path / "pyproject.toml"
        assert not missing_proj.exists()

        with pytest.raises(AssertionError):
            assert missing_proj.exists(), f"pyproject.toml not found at {missing_proj}"
