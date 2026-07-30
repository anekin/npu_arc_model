"""Tests for the SRAM bitcell area lookup table.

Positive tests verify exact known-node lookups and the ``sram_area_mm2()``
convenience function.  Negative tests verify ``ConfigError`` and ``ValueError``
on invalid inputs.
"""

from __future__ import annotations

import pytest
from contracts.bitcell import BitcellTable, sram_area_mm2
from contracts.errors import ConfigError

# ============================================================================
# Positive: known-node lookups
# ============================================================================


class TestBitcellTableKnownNodes:
    """Every node with published TSMC data must return the documented value."""

    @pytest.mark.parametrize(
        ("node_nm", "expected"),
        [
            (7.0, 0.027),
            (12.0, 0.074),
            (22.0, 0.092),
            (28.0, 0.127),
        ],
    )
    def test_area_um2_per_bit(self, node_nm: float, expected: float) -> None:
        table = BitcellTable()
        assert table.area_um2_per_bit(node_nm) == pytest.approx(expected, abs=1e-4)

    def test_known_nodes_property(self) -> None:
        table = BitcellTable()
        assert 7.0 in table.known_nodes
        assert 12.0 in table.known_nodes
        assert 22.0 in table.known_nodes
        assert 28.0 in table.known_nodes
        assert len(table.known_nodes) == 4

    def test_entry_returns_bitcell_entry(self) -> None:
        table = BitcellTable()
        entry = table.entry(7.0)
        assert entry.node_nm == 7.0
        assert entry.area_um2_per_bit == 0.027
        assert "TSMC" in entry.provenance
        assert entry.source_uri.startswith("http")


# ============================================================================
# Positive: sram_area_mm2()
# ============================================================================


class TestSramAreaMm2:
    """Verify the macro-level SRAM area formula."""

    def test_4kib_l1_default(self) -> None:
        """4 KiB L1 SRAM @7nm with default overhead 1.5×."""
        # 4096 bytes * 8 * 0.027 * (1 + 1.5) / 1e6
        area = sram_area_mm2(4 * 1024, 7.0)
        expected = 4096 * 8 * 0.027 * 2.5 / 1_000_000
        assert area == pytest.approx(expected, rel=1e-6)

    def test_2mib_l2_default(self) -> None:
        """2 MiB L2 SRAM @12nm with overhead 1.3×."""
        area = sram_area_mm2(2 * 1024 * 1024, 12.0, overhead=1.3)
        bits = 2 * 1024 * 1024 * 8
        expected = bits * 0.074 * 2.3 / 1_000_000
        assert area == pytest.approx(expected, rel=1e-6)

    def test_explicit_overhead_zero(self) -> None:
        """Pure bitcell area — overhead = 0."""
        area = sram_area_mm2(1024, 7.0, overhead=0.0)
        expected = 1024 * 8 * 0.027 * 1.0 / 1_000_000
        assert area == pytest.approx(expected, rel=1e-6)

    def test_custom_table_instance(self) -> None:
        """Passing an explicit BitcellTable instance works."""
        table = BitcellTable()
        area = sram_area_mm2(512, 7.0, table=table)
        expected = 512 * 8 * 0.027 * 2.5 / 1_000_000
        assert area == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        ("size_bytes", "node_nm", "overhead"),
        [
            (4 * 1024, 7.0, 1.5),  # L1 @7nm
            (4 * 1024, 12.0, 1.5),  # L1 @12nm
            (2 * 1024 * 1024, 7.0, 1.3),  # L2 @7nm
            (2 * 1024 * 1024, 28.0, 1.3),  # L2 @28nm
        ],
    )
    def test_smoke(self, size_bytes: int, node_nm: float, overhead: float) -> None:
        """Quick smoke test: calling with valid args never raises."""
        area = sram_area_mm2(size_bytes, node_nm, overhead=overhead)
        assert area > 0.0


# ============================================================================
# Negative: unknown node
# ============================================================================


class TestBitcellTableUnknownNode:
    """Unknown process nodes must raise ConfigError."""

    @pytest.mark.parametrize("bad_node", [5.0, 3.0, 14.0, 45.0])
    def test_area_um2_per_bit_raises(self, bad_node: float) -> None:
        table = BitcellTable()
        with pytest.raises(ConfigError, match="Unknown process node"):
            table.area_um2_per_bit(bad_node)

    def test_sram_area_mm2_raises(self) -> None:
        with pytest.raises(ConfigError, match="Unknown process node"):
            sram_area_mm2(1024, 5.0)


# ============================================================================
# Negative: invalid overhead
# ============================================================================


class TestSramAreaMm2InvalidOverhead:
    """Overhead out of valid range must raise ConfigError."""

    @pytest.mark.parametrize("bad_overhead", [-0.1, -1.0, 5.1, 10.0])
    def test_negative_or_excessive(self, bad_overhead: float) -> None:
        with pytest.raises(ConfigError, match="overhead"):
            sram_area_mm2(1024, 7.0, overhead=bad_overhead)

    def test_non_numeric_overhead(self) -> None:
        with pytest.raises(ConfigError, match="overhead"):
            sram_area_mm2(1024, 7.0, overhead="1.5")  # type: ignore[arg-type]


# ============================================================================
# Negative: zero / negative size
# ============================================================================


class TestSramAreaMm2InvalidSize:
    """Non-positive size_bytes must raise ValueError."""

    @pytest.mark.parametrize("bad_size", [0, -1, -1024])
    def test_zero_or_negative(self, bad_size: int) -> None:
        with pytest.raises(ValueError, match="size_bytes must be positive"):
            sram_area_mm2(bad_size, 7.0)


# ============================================================================
# Structural: entry metadata
# ============================================================================


class TestBitcellEntryMetadata:
    """Every bitcell entry must carry source_uri and provenance."""

    def test_all_entries_have_metadata(self) -> None:
        table = BitcellTable()
        for node_nm in table.known_nodes:
            entry = table.entry(node_nm)
            assert isinstance(entry.source_uri, str) and len(entry.source_uri) > 0
            assert isinstance(entry.provenance, str) and len(entry.provenance) > 0
