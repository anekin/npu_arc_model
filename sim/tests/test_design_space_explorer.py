"""Tests for design_space_explorer scenario resolution and cross-validation."""

import pytest
from design_space_explorer import _resolve_cv_scenario


def _make_config(memory_type: str) -> dict:
    return {"memory": {"type": memory_type, "bandwidth_gbps": 51.2}}


def _make_base_cfg(seq_len: int | None = None) -> dict:
    cfg = {
        "memory": {"type": "LPDDR5-6400", "bandwidth_gbps": 51.2},
        "area_model": {"process_node": 12},
    }
    if seq_len is not None:
        cfg["seq_len"] = seq_len
    return cfg


class TestResolveCvScenario:
    """Direct unit tests for _resolve_cv_scenario with sample configs."""

    @pytest.mark.parametrize(
        ("memory_type", "seq_len", "expected"),
        [
            # on_chip_3d_dram with seq_len > 256 -> onchip_7b
            ("on_chip_3d_dram", 1024, "onchip_7b"),
            ("on_chip_3d_dram", 4096, "onchip_7b"),
            ("on_chip_3d_dram", 257, "onchip_7b"),
            # on_chip_3d_dram with seq_len <= 256 -> onchip_7b_chat
            ("on_chip_3d_dram", 128, "onchip_7b_chat"),
            ("on_chip_3d_dram", 256, "onchip_7b_chat"),
            ("on_chip_3d_dram", 1, "onchip_7b_chat"),
            # hbm2e -> hbm2e_7b
            ("HBM2e-1024b", 128, "hbm2e_7b"),
            ("hbm2e", 1024, "hbm2e_7b"),
            # lpddr5x -> lpddr5x_7b
            ("LPDDR5X-8533", 128, "lpddr5x_7b"),
            ("lpddr5x", 1024, "lpddr5x_7b"),
            # lpddr5 -> lpddr5_3b
            ("LPDDR5-6400", 128, "lpddr5_3b"),
            ("lpddr5-6400", 1024, "lpddr5_3b"),
            ("lpddr5", 128, "lpddr5_3b"),
            # lpddr5-32b, lpddr5-256b etc
            ("LPDDR5-32b", 128, "lpddr5_3b"),
            ("LPDDR5-256b", 128, "lpddr5_3b"),
            # hbm3 -> fallback
            ("HBM3-1024b", 128, "lpddr5_3b"),
            # Unknown memory type -> fallback
            ("ddr5", 128, "lpddr5_3b"),
            ("sram", 128, "lpddr5_3b"),
            # Empty memory type -> fallback
            ("", 128, "lpddr5_3b"),
        ],
    )
    def test_resolve(self, memory_type: str, seq_len: int, expected: str) -> None:
        config = _make_config(memory_type)
        base_cfg = _make_base_cfg()
        result = _resolve_cv_scenario(config, base_cfg, seq_len_override=seq_len)
        assert result == expected, (
            f"memory_type={memory_type!r}, seq_len={seq_len}: expected {expected!r}, got {result!r}"
        )

    def test_default_seq_len_from_base_cfg(self) -> None:
        """When no seq_len_override is given, read from base_cfg seq_len."""
        config = _make_config("on_chip_3d_dram")
        base_cfg = _make_base_cfg(seq_len=1024)
        # seq_len=1024 > 256 -> onchip_7b
        assert _resolve_cv_scenario(config, base_cfg) == "onchip_7b"

    def test_default_seq_len_missing_from_base(self) -> None:
        """When base_cfg has no seq_len, default to 128."""
        config = _make_config("on_chip_3d_dram")
        base_cfg = _make_base_cfg()  # no seq_len key
        # seq_len defaults to 128, which is <= 256 -> onchip_7b_chat
        assert _resolve_cv_scenario(config, base_cfg) == "onchip_7b_chat"

    def test_workload_seq_len_from_base(self) -> None:
        """Read seq_len from workload.seq_len in base_cfg."""
        config = _make_config("on_chip_3d_dram")
        base_cfg = _make_base_cfg()
        base_cfg["workload"] = {"seq_len": 512}
        assert _resolve_cv_scenario(config, base_cfg) == "onchip_7b"

    def test_seq_len_override_wins(self) -> None:
        """seq_len_override takes precedence over base_cfg."""
        config = _make_config("on_chip_3d_dram")
        base_cfg = _make_base_cfg(seq_len=128)
        # Override with 1024 > 256 -> onchip_7b
        assert _resolve_cv_scenario(config, base_cfg, seq_len_override=1024) == "onchip_7b"

    def test_lpddr5x_before_lpddr5(self) -> None:
        """lpddr5x is checked before lpddr5, so it goes to lpddr5x_7b."""
        config = _make_config("lpddr5x-8533")
        base_cfg = _make_base_cfg()
        assert _resolve_cv_scenario(config, base_cfg) == "lpddr5x_7b"

    def test_case_insensitive(self) -> None:
        """Memory type matching is case-insensitive (lowercased)."""
        config = _make_config("HBM2E-1024B")
        base_cfg = _make_base_cfg()
        assert _resolve_cv_scenario(config, base_cfg) == "hbm2e_7b"
