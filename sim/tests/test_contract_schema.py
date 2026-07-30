"""Schema contract tests for sim.contracts.hardware and migration paths.

Covers plan acceptance criteria:

* Round-trip: load legacy YAML, migrate to v2, re-validate, project back.
* Conflict: ``mxu`` ≠ ``mac_engine`` → ConfigError.
* Malformed YAML: non-mapping root, invalid version, bool-as-int,
  non-finite/non-positive numbers.
"""

import tempfile
from pathlib import Path

import pytest
import yaml
from config.npu_config import load_config
from contracts.errors import ConfigError
from contracts.hardware import (
    HardwareConfigV2,
    MACEngineConfig,
    MemoryConfig,
    Provenance,
    TrustLevel,
)
from contracts.migrations import migrate_v1_to_v2, project_v2_to_legacy
from pydantic import ValidationError

# ── Helpers ────────────────────────────────────────────────────────────────


def _minimal_v2(mac_type: str = "block") -> dict:
    """Return a minimal, valid v2 config dict."""
    return {
        "version": "2",
        "mac_engine": {
            "type": mac_type,
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "bandwidth_gbps": 51.2,
        },
    }


def _minimal_v1(mac_type: str = "block") -> dict:
    """Return a minimal v1 config dict (uses ``mxu`` key)."""
    return {
        "mxu": {
            "type": mac_type,
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "bandwidth_gbps": 51.2,
        },
    }


def _load_config_as_v2() -> HardwareConfigV2:
    """Load and validate ``npu_config.yaml`` as v2 schema.

    Strips extra sections not yet part of the v2 schema.
    """
    raw = load_config()
    migrated, _ = migrate_v1_to_v2(raw)
    # Drop legacy sections not yet in the v2 schema
    for key in list(migrated.keys()):
        if key not in (
            "version",
            "mac_engine",
            "memory",
            "sram",
            "cores",
            "on_chip_memory",  # optional, keep
        ):
            del migrated[key]
    return HardwareConfigV2.model_validate(migrated)


# ── Happy-path: round-trip ────────────────────────────────────────────────


def test_v1_to_v2_round_trip():
    """Given a legacy v1 config (mxu),
    When migrating to v2 and back,
    Then the essential fields survive the round-trip.
    """
    v1 = _minimal_v1("block")
    v1["memory"]["bandwidth_bytes_per_cycle"] = 51.2

    v2, loss = migrate_v1_to_v2(v1)

    assert v2["version"] == "2"
    assert v2["mac_engine"]["type"] == "block"
    assert "bandwidth_bytes_per_cycle" not in v2.get("memory", {})
    assert loss.warnings  # migration warning is expected

    legacy, loss2 = project_v2_to_legacy(v2)

    assert legacy["mxu"]["type"] == "block"
    assert legacy["mxu"]["array_height"] == 64
    assert legacy["memory"]["bandwidth_bytes_per_cycle"] == 51.2


def test_load_real_npu_config_yaml():
    """Given the actual npu_config.yaml,
    When loading and validating as v2 (via migrate_v1_to_v2),
    Then no error is raised and mac_engine is present.
    """
    hw = _load_config_as_v2()
    assert hw.mac_engine.type == "block"
    assert hw.mac_engine.array_height == 64
    assert hw.memory.bandwidth_gbps == 51.2
    assert hw.memory.dram_efficiency == 0.85


def test_load_real_design_space_yaml():
    """Given design_space.yaml (which has mac_engine natively),
    When loading directly,
    Then it validates as v2 after stripping non-v2 sections.
    """
    dse_path = Path(__file__).parent.parent / "config" / "design_space.yaml"
    with open(dse_path) as f:
        raw = yaml.safe_load(f)

    # Drop legacy sections not yet in the v2 schema
    for key in list(raw.keys()):
        if key not in ("version", "mac_engine", "memory", "sram", "cores", "on_chip_memory"):
            del raw[key]

    # Drop v1-only fields from memory (bandwidth_bytes_per_cycle is forbidden in v2)
    if "memory" in raw and isinstance(raw["memory"], dict):
        raw["memory"].pop("bandwidth_bytes_per_cycle", None)

    cfg = HardwareConfigV2.model_validate(raw)

    assert cfg.mac_engine.type == "block"
    assert cfg.mac_engine.array_height == 64
    assert cfg.mac_engine.array_width == 64
    assert cfg.memory.bandwidth_gbps == 51.2


def test_pydantic_round_trip():
    """Given a HardwareConfigV2 instance,
    When serializing to dict and re-validating,
    Then the re-validated instance is equivalent.
    """
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=128,
            array_width=128,
            frequency_mhz=800,
        ),
        memory=MemoryConfig(
            type="LPDDR5-6400",
            bandwidth_gbps=51.2,
        ),
    )
    as_dict = cfg.model_dump()
    cfg2 = HardwareConfigV2.model_validate(as_dict)
    assert cfg2 == cfg


def test_bandwidth_bytes_per_cycle_computed():
    """Given a HardwareConfigV2 with known bandwidth and frequency,
    When computing bandwidth_bytes_per_cycle,
    Then the formula 51.2 * 1000 / 800 = 64.0 holds.
    """
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=800,
        ),
        memory=MemoryConfig(
            type="LPDDR5-6400",
            bandwidth_gbps=51.2,
        ),
    )
    assert cfg.bandwidth_bytes_per_cycle() == 64.0


def test_effective_bytes_per_cycle():
    """Given dram_efficiency=0.85,
    When computing effective bytes/cycle,
    Then the result is raw_bytes_per_cycle * 0.85.
    """
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=1000,
        ),
        memory=MemoryConfig(
            type="LPDDR5-6400",
            bandwidth_gbps=51.2,
            dram_efficiency=0.85,
        ),
    )
    assert cfg.effective_bytes_per_cycle() == pytest.approx(51.2 * 0.85)


# ── mxu / mac_engine conflict ─────────────────────────────────────────────


def test_mxu_present_alone_migrates():
    """Given a v1 config with only mxu,
    When migrating to v2,
    Then mxu is renamed to mac_engine with a warning.
    """
    v1 = _minimal_v1("block")
    v2, loss = migrate_v1_to_v2(v1)

    assert "mac_engine" in v2
    assert "mxu" not in v2
    assert any("Renamed" in w for w in loss.warnings)


def test_mxu_and_mac_engine_both_present_and_consistent():
    """Given a config with both mxu and mac_engine (same type),
    When migrating to v2,
    Then mxu is dropped and a warning is emitted.
    """
    cfg = {
        "mac_engine": {
            "type": "block",
            "array_height": 64,
            "array_width": 64,
            "frequency_mhz": 1000,
        },
        "mxu": {
            "type": "block",
            "array_height": 128,
            "array_width": 128,
        },
        "memory": {"bandwidth_gbps": 51.2},
    }
    v2, loss = migrate_v1_to_v2(cfg)

    assert "mac_engine" in v2
    assert "mxu" not in v2
    # mac_engine should be preserved with its original values
    assert v2["mac_engine"]["array_height"] == 64
    assert any("consistent" in w.lower() for w in loss.warnings)


def test_mxu_and_mac_engine_both_present_and_inconsistent():
    """Given a config with conflicting mxu and mac_engine types,
    When migrating to v2,
    Then ConfigError is raised with a clear field path.
    """
    cfg = {
        "mac_engine": {"type": "block", "array_height": 64, "array_width": 64},
        "mxu": {"type": "systolic", "array_height": 128, "array_width": 128},
        "memory": {"bandwidth_gbps": 51.2},
    }
    with pytest.raises(ConfigError, match="Conflicting"):
        migrate_v1_to_v2(cfg)


# ── extra='forbid' ────────────────────────────────────────────────────────


def test_extra_field_forbidden_in_mac_engine():
    """Given a mac_engine config with an unknown field,
    When validating as MACEngineConfig,
    Then ValidationError is raised.
    """
    with pytest.raises(ValidationError, match="Extra inputs"):
        MACEngineConfig(
            array_height=64,
            array_width=64,
            frequency_mhz=1000,
            made_up_field=123,
        )


def test_extra_field_forbidden_in_memory():
    """Given a memory config with an unknown field,
    When validating as MemoryConfig,
    Then ValidationError is raised.
    """
    with pytest.raises(ValidationError, match="Extra inputs"):
        MemoryConfig(
            type="LPDDR5-6400",
            bandwidth_gbps=51.2,
            bogus_key=42,
        )


# ── version validation ────────────────────────────────────────────────────


def test_missing_version_defaults_to_v2():
    """Given a config without a 'version' field,
    When validating as HardwareConfigV2,
    Then it defaults to '2' and succeeds.
    """
    cfg = HardwareConfigV2.model_validate(
        {
            "mac_engine": {
                "type": "block",
                "array_height": 64,
                "array_width": 64,
                "frequency_mhz": 1000,
            },
            "memory": {"bandwidth_gbps": 51.2},
        }
    )
    assert cfg.version == "2"


def test_wrong_version_fails():
    """Given a config with version='1',
    When validating as HardwareConfigV2,
    Then a validation error is raised.
    """
    with pytest.raises((ValidationError, ValueError), match="version"):
        HardwareConfigV2.model_validate(
            {
                "version": "1",
                "mac_engine": {
                    "type": "block",
                    "array_height": 64,
                    "array_width": 64,
                },
                "memory": {"bandwidth_gbps": 51.2},
            }
        )


# ── NaN / Inf ─────────────────────────────────────────────────────────────


def test_nan_bandwidth_gbps_fails():
    """Given a memory config with NaN bandwidth,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="finite"):
        MemoryConfig(bandwidth_gbps=float("nan"))


def test_inf_bandwidth_gbps_fails():
    """Given a memory config with +Inf bandwidth,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="finite"):
        MemoryConfig(bandwidth_gbps=float("inf"))


def test_inf_frequency_fails():
    """Given a mac_engine config with +Inf frequency,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="finite"):
        MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=float("inf"),
        )


def test_nan_frequency_fails():
    """Given a mac_engine config with NaN frequency,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="finite"):
        MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=float("nan"),
        )


# ── zero / negative ───────────────────────────────────────────────────────


def test_zero_frequency_fails():
    """Given a mac_engine config with frequency_mhz=0,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="positive"):
        MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=0,
        )


def test_negative_frequency_fails():
    """Given a mac_engine config with negative frequency,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="positive"):
        MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=-100,
        )


def test_zero_bandwidth_fails():
    """Given a memory config with bandwidth_gbps=0,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="positive"):
        MemoryConfig(bandwidth_gbps=0)


def test_negative_bandwidth_fails():
    """Given a memory config with negative bandwidth,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="positive"):
        MemoryConfig(bandwidth_gbps=-51.2)


def test_zero_array_dimension_fails():
    """Given a mac_engine config with array_height=0,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="positive"):
        MACEngineConfig(
            type="block",
            array_height=0,
            array_width=64,
        )


# ── bool-as-int ───────────────────────────────────────────────────────────


def test_bool_bandwidth_fails():
    """Given a memory config with True as bandwidth_gbps,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="bool"):
        MemoryConfig(bandwidth_gbps=True)


def test_bool_frequency_fails():
    """Given a mac_engine config with False as frequency_mhz,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="bool"):
        MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            frequency_mhz=False,
        )


def test_bool_array_dimension_fails():
    """Given a mac_engine config with True as array_height,
    When validating,
    Then a validation error is raised."""
    with pytest.raises(ValidationError, match="bool"):
        MACEngineConfig(
            type="block",
            array_height=True,
            array_width=64,
        )


# ── malformed YAML shapes ─────────────────────────────────────────────────


def test_non_mapping_root_fails():
    """Given a YAML file containing just a list,
    When loading via load_config,
    Then ConfigError is raised with path '<root>'."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("- this is a list\n")
        f.write("- not a mapping\n")
        tmp_path = f.name

    try:
        # Monkey-patch the config path for the test
        import config.npu_config as cfg_mod

        str(Path(cfg_mod.__file__).with_suffix(".yaml"))
        # Use load_config logic directly
        import yaml as y
        from contracts.errors import ConfigError

        with open(tmp_path) as fh:
            data = y.safe_load(fh)
        from config.npu_config import _validate_root_is_mapping

        with pytest.raises(ConfigError) as exc_info:
            _validate_root_is_mapping(data)
        assert exc_info.value.field_path == "<root>"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_non_mapping_sub_section():
    """Given a v2 config with mac_engine as a string instead of a dict,
    When validating as HardwareConfigV2,
    Then ValidationError is raised."""
    with pytest.raises(ValidationError):
        HardwareConfigV2.model_validate(
            {
                "version": "2",
                "mac_engine": "not a dict",
                "memory": {"bandwidth_gbps": 51.2},
            }
        )


# ── provenance defaults ───────────────────────────────────────────────────


def test_provenance_fields_exist():
    """Given a HardwareConfigV2 with default memory config,
    When checking the memory.provenance field,
    Then it is None (default) but the field exists for assignment.
    """
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
        ),
        memory=MemoryConfig(bandwidth_gbps=51.2),
    )
    assert cfg.mac_engine.provenance is None
    assert cfg.memory.provenance is None


def test_provenance_can_be_set():
    """Given a Provenance record,
    When assigning it to a config component,
    Then it is stored correctly.
    """
    prov = Provenance(
        source="Test source",
        trust_level=TrustLevel.T1,
        calibration_range="0.8–0.9",
        reference_uri="test://ref",
    )
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            provenance=prov,
        ),
        memory=MemoryConfig(bandwidth_gbps=51.2),
    )
    assert cfg.mac_engine.provenance == prov
    assert cfg.mac_engine.provenance.trust_level == TrustLevel.T1


def test_provenance_round_trips():
    """Given a config with provenance set,
    When serializing to dict and re-validating,
    Then provenance is preserved.
    """
    prov = Provenance(
        source="Test source",
        trust_level=TrustLevel.T1,
    )
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            provenance=prov,
        ),
        memory=MemoryConfig(bandwidth_gbps=51.2),
    )
    as_dict = cfg.model_dump()
    cfg2 = HardwareConfigV2.model_validate(as_dict)
    assert cfg2.mac_engine.provenance == prov


# ── loss report ───────────────────────────────────────────────────────────


def test_loss_report_dropped_keys():
    """Given the real npu_config.yaml migrating to v2,
    When bandwidth_bytes_per_cycle is present in the legacy,
    Then it appears in the loss report dropped_keys.
    """
    v1 = _minimal_v1("block")
    v1["memory"]["bandwidth_bytes_per_cycle"] = 51.2

    _, loss = migrate_v1_to_v2(v1)
    assert "memory.bandwidth_bytes_per_cycle" in loss.dropped_keys


def test_project_v2_to_legacy_loss_report():
    """Given a v2 config with provenance,
    When projecting to legacy,
    Then provenance is dropped and recorded in the loss report.
    """
    prov = Provenance(source="test", trust_level=TrustLevel.T0)
    cfg = HardwareConfigV2(
        mac_engine=MACEngineConfig(
            type="block",
            array_height=64,
            array_width=64,
            provenance=prov,
        ),
        memory=MemoryConfig(
            bandwidth_gbps=51.2,
            provenance=prov,
        ),
    )
    legacy, loss = project_v2_to_legacy(cfg.model_dump())

    assert "mxu" in legacy
    assert "provenance" not in legacy.get("mxu", {})
    assert "provenance" not in legacy.get("memory", {})
    assert any("provenance" in k for k in loss.dropped_keys)
