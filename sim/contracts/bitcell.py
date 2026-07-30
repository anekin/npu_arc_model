"""SRAM bitcell area lookup table and SRAM area estimator.

Provides a trusted, provenance-tracked mapping from process node (nm) to
bitcell area (µm²/bit), sourced from TSMC foundry data.  Consumers compute
macro-level SRAM area via ``sram_area_mm2()``, which applies a configurable
peripheral overhead factor.

Usage::

    from contracts.bitcell import BitcellTable, sram_area_mm2

    table = BitcellTable()
    area = table.area_um2_per_bit(7.0)       # → 0.027
    sram = sram_area_mm2(4 * 1024, 7.0)      # → 4 KiB SRAM @7nm
    sram = sram_area_mm2(4 * 1024, 7.0, overhead=1.3)  # L2-style
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.errors import ConfigError

# ── Known bitcell areas (TSMC foundry data) ──────────────────────────────
# Each entry is a process node with its HD (high-density) bitcell area in
# µm²/bit.  Source documents are cited in provenance.
#
# References:
#   [TSMC-7nm]  TSMC 7nm (N7) HD bitcell: 0.027 µm²/bit
#               Source: VLSI 2017 / IEDM 2017 TSMC disclosures
#   [TSMC-12FFC] TSMC 12FFC (optical shrink of 16FFC) HD bitcell: 0.074 µm²/bit
#               Source: TSMC 12FFC product brief, 2020
#   [TSMC-22nm] TSMC 22nm (ULL) HD bitcell: ~0.092 µm²/bit
#               Source: TSMC 22nm product brief, 2023
#   [TSMC-28nm] TSMC 28nm (HPM) HD bitcell: ~0.127 µm²/bit
#               Source: TSMC 28nm product brief, foundry data
#


@dataclass(frozen=True)
class BitcellEntry:
    """A single bitcell entry with provenance metadata."""

    node_nm: float
    area_um2_per_bit: float
    source_uri: str
    provenance: str


# fmt: off
_BITCELL_TABLE: dict[float, BitcellEntry] = {
    7.0: BitcellEntry(
        node_nm=7.0,
        area_um2_per_bit=0.027,
        source_uri="https://ieeexplore.ieee.org/xpl/conhome/1000142/all-proceedings",
        provenance="TSMC N7 HD bitcell — VLSI 2017 / IEDM 2017 TSMC disclosures",
    ),
    12.0: BitcellEntry(
        node_nm=12.0,
        area_um2_per_bit=0.074,
        source_uri="https://www.tsmc.com/english/dedicatedFoundry/technology/12nm",
        provenance="TSMC 12FFC (optical shrink of 16FFC) HD bitcell — product brief 2020",
    ),
    22.0: BitcellEntry(
        node_nm=22.0,
        area_um2_per_bit=0.092,
        source_uri="https://www.tsmc.com/english/dedicatedFoundry/technology/22nm",
        provenance="TSMC 22nm ULL HD bitcell — product brief 2023",
    ),
    28.0: BitcellEntry(
        node_nm=28.0,
        area_um2_per_bit=0.127,
        source_uri="https://www.tsmc.com/english/dedicatedFoundry/technology/28nm",
        provenance="TSMC 28nm HPM HD bitcell — foundry data sheet",
    ),
}
# fmt: on

_KNOWN_NODES: frozenset[float] = frozenset(_BITCELL_TABLE.keys())

# Valid overhead range — peripheral logic typically adds 20-80% to raw
# bitcell area depending on macro size and bank count.
_OVERHEAD_MIN = 0.0
_OVERHEAD_MAX = 5.0


class BitcellTable:
    """Trusted lookup for SRAM bitcell area at a given process node.

    Only nodes with published, verified TSMC foundry data are accepted.
    Requests for an unknown node raise ``ConfigError``.
    """

    def __init__(self) -> None:
        self._entries = dict(_BITCELL_TABLE)

    @property
    def known_nodes(self) -> frozenset[float]:
        """Return the set of process nodes (nm) that have entries."""
        return _KNOWN_NODES

    def entry(self, node_nm: float) -> BitcellEntry:
        """Return the full ``BitcellEntry`` for *node_nm*.

        Raises ``ConfigError`` if the node is not in the table.
        """
        if node_nm not in self._entries:
            raise ConfigError(
                f"Unknown process node {node_nm}nm — bitcell area not in table. "
                f"Known nodes: {sorted(_KNOWN_NODES)}",
                field_path="process_node_nm",
                value=node_nm,
            )
        return self._entries[node_nm]

    def area_um2_per_bit(self, node_nm: float) -> float:
        """Return bitcell area in µm²/bit for *node_nm*.

        Raises ``ConfigError`` if the node is not in the table.
        """
        return self.entry(node_nm).area_um2_per_bit


def sram_area_mm2(
    size_bytes: int,
    node_nm: float,
    overhead: float = 1.5,
    table: BitcellTable | None = None,
) -> float:
    """Return the estimated macro-level SRAM area in mm².

    Parameters
    ----------
    size_bytes:
        SRAM capacity in bytes (e.g. ``4 * 1024`` for 4 KiB).
    node_nm:
        Process node in nm.  Must be a known node in the bitcell table.
    overhead:
        Peripheral overhead multiplier.  L1 default = 1.5×, L2 default = 1.3×.
        Must be between 0.0 and 5.0 (inclusive).
    table:
        Optional ``BitcellTable`` instance.  Defaults to a fresh table.

    Formula
    -------
    ``size_bytes * 8 * bitcell_area_um2 * (1 + overhead) / 1_000_000``

    Raises
    ------
    ConfigError
        If *node_nm* is unknown or *overhead* is out of range.
    ValueError
        If *size_bytes* is not positive.
    """
    if size_bytes <= 0:
        raise ValueError(
            f"size_bytes must be positive, got {size_bytes}"
        )

    if not isinstance(overhead, (int, float)):
        raise ConfigError(
            f"overhead must be a number, got {type(overhead).__name__}",
            field_path="overhead",
            value=overhead,
        )

    if overhead < _OVERHEAD_MIN or overhead > _OVERHEAD_MAX:
        raise ConfigError(
            f"overhead {overhead} out of range [{_OVERHEAD_MIN}, {_OVERHEAD_MAX}]",
            field_path="overhead",
            value=overhead,
        )

    tbl = table or BitcellTable()
    bitcell_area = tbl.area_um2_per_bit(node_nm)

    bits = size_bytes * 8
    raw_bitcell_mm2 = bits * bitcell_area / 1_000_000
    return raw_bitcell_mm2 * (1.0 + overhead)


__all__ = [
    "BitcellEntry",
    "BitcellTable",
    "sram_area_mm2",
]
