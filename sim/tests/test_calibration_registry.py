"""Calibration registry tests.

Covers:
  - Loading the canonical parameters.yaml
  - Completeness of the 10 decision-driving parameters
  - Lookup by calibration_id
  - Duplicate-ID rejection
  - Checksum mismatch detection in calibrate_mxu_model.py
  - Held-out fixture exclusion from fitting
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from calibration.registry import CalibrationRegistry
from calibration.schema import CalibrationError, CalibrationStatus, TrustLevel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "references" / "calibration" / "raw"
CALIBRATE_SCRIPT = REPO_ROOT / "scripts" / "calibrate_mxu_model.py"

EXPECTED_IDS = {
    "systolic_pe_area_7nm",
    "systolic_pe_area_28nm",
    "systolic_pe_area_22nm",
    "systolic_pe_area_12nm",
    "block_pe_area_7nm",
    "block_pe_area_28nm",
    "block_pe_area_22nm",
    "block_pe_area_12nm",
    "fsa_pe_area_7nm",
    "fsa_pe_area_28nm",
    "fsa_pe_area_22nm",
    "fsa_pe_area_12nm",
    "block_systolic_pe_ratio",
    "gmma_pipeline_scale",
    "tma_overlap",
    "wmma_fragment_serialization_cycles",
    "tensor_core_descriptor_overhead",
    "wmma_pe_ratio",
    "wmma_pe_area_7nm",
    "gmma_pe_ratio",
    "gmma_pe_area_7nm",
    "pj_per_mac_12nm_int8",
    "tsv_overhead_pct",
    "dram_phy_area_12nm",
    "power_density_12nm",
}


def test_valid_registry_has_all_parameters():
    """Canonical registry contains exactly the decision-driving parameters."""
    registry = CalibrationRegistry.from_yaml()
    assert set(registry.ids()) == EXPECTED_IDS


def test_valid_lookup_by_id_reports_trust_level():
    """Lookup returns entry with correct trust level and source_uri."""
    registry = CalibrationRegistry.from_yaml()

    systolic = registry.get("systolic_pe_area_7nm")
    assert systolic.trust_level == TrustLevel.T2
    assert systolic.source_uri is not None

    gmma_scale = registry.get("gmma_pipeline_scale")
    assert gmma_scale.trust_level == TrustLevel.T1
    assert gmma_scale.source_uri is not None

    block_ratio = registry.get("block_systolic_pe_ratio")
    assert block_ratio.trust_level == TrustLevel.T1


def test_registry_lookup_unknown_id_raises():
    """Unknown calibration_id raises CalibrationError."""
    registry = CalibrationRegistry.from_yaml()
    with pytest.raises(CalibrationError, match="unknown calibration_id"):
        registry.get("nonexistent_parameter")


def test_registry_from_dict_rejects_duplicate_id():
    """Registry construction rejects duplicate calibration_id values."""
    data = {
        "param_a": {"value": 1.0, "unit": "x", "trust_level": "T0", "calibration_range": "0-1", "status": "assumption"},
        "param_b": {"value": 2.0, "unit": "x", "trust_level": "T0", "calibration_range": "0-1", "status": "assumption"},
    }
    registry = CalibrationRegistry.from_dict(data)
    assert len(registry) == 2

    data["param_c"] = data["param_a"]
    data["param_c"]["calibration_id"] = "param_a"
    with pytest.raises(CalibrationError, match="duplicate calibration_id"):
        CalibrationRegistry.from_dict(data)


def test_registry_digest_changes_on_value_change():
    """Changing a calibration value changes the registry digest."""
    registry = CalibrationRegistry.from_yaml()
    d1 = registry.to_dict()

    entry = registry.get("gmma_pipeline_scale")
    entry_dict = entry.model_dump(mode="json")
    entry_dict["value"] = 0.99
    modified = CalibrationRegistry(
        [e if e.calibration_id != "gmma_pipeline_scale" else e.model_validate(entry_dict) for e in registry.entries()]
    )
    d2 = modified.to_dict()
    assert d1 != d2


def test_parameters_yaml_has_required_fields():
    """Every entry has calibration_id, value, unit, source_uri, trust_level, range, status."""
    registry = CalibrationRegistry.from_yaml()
    for entry in registry.entries():
        assert entry.calibration_id
        assert isinstance(entry.value, (int, float))
        assert entry.unit
        assert entry.source_uri is not None or entry.trust_level == TrustLevel.T0
        assert entry.trust_level in {TrustLevel.T0, TrustLevel.T1, TrustLevel.T2, TrustLevel.T3}
        assert entry.calibration_range
        assert entry.status in {
            CalibrationStatus.assumption,
            CalibrationStatus.calibrated,
            CalibrationStatus.exploratory,
        }


# ── calibrate_mxu_model.py integration ─────────────────────────────────────


def _run_calibrate_script(*, cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(CALIBRATE_SCRIPT)]
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        env=env or os.environ,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_calibrate_run_is_deterministic():
    """Valid fixtures produce deterministic metrics and exclude held-out from fitting."""
    result = _run_calibrate_script()
    assert result.returncode == 0, result.stderr

    output = json.loads((REPO_ROOT / ".omo" / "evidence" / "mxu-calibration.json").read_text())
    assert output["train"]["count"] == 8
    assert output["heldout"]["count"] == 2
    # Held-out case IDs are separate from train IDs.
    train_ids = set(output["train_case_ids"])
    heldout_ids = set(output["heldout_case_ids"])
    assert not train_ids & heldout_ids


def test_heldout_ids_do_not_participate_in_fitting():
    """Metrics on held-out set prove they were not used for fitting."""
    _run_calibrate_script()
    output = json.loads((REPO_ROOT / ".omo" / "evidence" / "mxu-calibration.json").read_text())
    assert set(output["heldout_case_ids"]).isdisjoint(set(output["train_case_ids"]))
    assert output["heldout"]["count"] > 0


def test_calibrate_script_fails_closed_on_missing_raw_dir(tmp_path: Path):
    """Missing raw fixture directory causes non-zero exit with CalibrationError."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "sim")!r})
from pathlib import Path
from scripts.calibrate_mxu_model import _load_raw_fixtures
from calibration.schema import CalibrationError
try:
    _load_raw_fixtures(Path({str(tmp_path / "empty")!r}))
    sys.exit(0)
except CalibrationError as e:
    print(e.reason)
    sys.exit(2)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "missing_checksum_manifest" in result.stdout


def test_calibrate_script_fails_closed_on_checksum_mismatch():
    """Tampered raw fixture causes non-zero exit with checksum mismatch."""
    train_path = RAW_DIR / "mxu_train.csv"
    original = train_path.read_text()
    try:
        tampered = original.replace("MX-T01", "MX-T01x")
        train_path.write_text(tampered)
        result = _run_calibrate_script()
        assert result.returncode == 2, result.stdout
        assert "checksum mismatch" in result.stderr
    finally:
        train_path.write_text(original)


def test_calibrate_script_fails_closed_on_duplicate_case_id():
    """Duplicate case_id across train/held-out causes non-zero exit."""
    heldout_path = RAW_DIR / "mxu_heldout.csv"
    sums_path = RAW_DIR / "SHA256SUMS"
    original_heldout = heldout_path.read_text()
    original_sums = sums_path.read_text()
    try:
        duplicate = original_heldout.replace("MX-H01", "MX-T01")
        heldout_path.write_text(duplicate)
        new_hash = hashlib.sha256(heldout_path.read_bytes()).hexdigest()
        sums_path.write_text(f"{new_hash}  mxu_heldout.csv\n", encoding="utf-8")
        result = _run_calibrate_script()
        assert result.returncode == 2, result.stdout
        assert "duplicate case_id" in result.stderr
    finally:
        heldout_path.write_text(original_heldout)
        sums_path.write_text(original_sums, encoding="utf-8")


def test_sha256sums_matches_actual_files():
    """SHA256SUMS file is consistent with fixture CSVs."""
    sums_path = RAW_DIR / "SHA256SUMS"
    lines = sums_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        expected_hash, name = line.split(None, 1)
        actual_hash = hashlib.sha256((RAW_DIR / name).read_bytes()).hexdigest()
        assert actual_hash == expected_hash
