"""Cross-node area regression tests for all 8 engine types.

Validates that ``engine.ppa_model.AreaModel`` produces physically consistent
area estimates across TSMC 7/12/22/28nm nodes:

* Total area decreases monotonically as process node shrinks.
* SRAM share of total area increases as node shrinks (logic scales faster
  than SRAM).
* SRAM-heavy designs suffer a larger relative area disadvantage at older
  nodes than SRAM-light designs.
"""

from __future__ import annotations

from typing import Any

import pytest
from contracts.bitcell import BitcellTable, sram_area_mm2
from engine.ppa_model import AreaModel

ENGINE_TYPES: list[str] = [
    "systolic",
    "block",
    "os_systolic",
    "input_stationary",
    "tensor_core",
    "wmma",
    "gmma",
    "fsa",
]

NODES_NM: list[float] = [7.0, 12.0, 22.0, 28.0]

_L1_KB = 512
_L2_KB = 2048
_L1_OVERHEAD = 1.5
_L2_OVERHEAD = 1.3


def _base_config(process_node_nm: float) -> dict[str, Any]:
    """Return a minimal config for ``AreaModel`` at the requested node."""
    return {
        "area_model": {"process_node_nm": process_node_nm},
        "mac_engine": {
            "array_height": 128,
            "array_width": 128,
            "frequency_mhz": 1000,
        },
        "sram": {
            "l1_per_core_kb": _L1_KB,
            "l2_shared_kb": _L2_KB,
        },
        "memory": {
            "type": "LPDDR5-6400",
            "capacity_gb": 4.0,
            "bandwidth_gbps": 51.2,
        },
        "dma": {"channels": 2},
        "optimizations": {"dma_bw_multiplier": 1.0},
    }


def _config_with_l2(process_node_nm: float, l2_kb: float) -> dict[str, Any]:
    """Return a config with a custom L2 size for ratio tests."""
    cfg = _base_config(process_node_nm)
    cfg["sram"]["l2_shared_kb"] = l2_kb
    return cfg


def _sram_area_mm2(process_node_nm: float, l1_kb: float, l2_kb: float) -> float:
    """Compute bitcell-derived L1+L2 SRAM area for the given node."""
    table = BitcellTable()
    l1 = sram_area_mm2(
        size_bytes=int(l1_kb * 1024),
        node_nm=process_node_nm,
        overhead=_L1_OVERHEAD,
        table=table,
    )
    l2 = sram_area_mm2(
        size_bytes=int(l2_kb * 1024),
        node_nm=process_node_nm,
        overhead=_L2_OVERHEAD,
        table=table,
    )
    return l1 + l2


class TestTotalAreaMonotonic:
    """Total area must strictly decrease as the process node shrinks."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    def test_total_area_decreases_with_node(self, engine_type: str) -> None:
        """Given identical config, area(28nm) > area(22nm) > area(12nm) > area(7nm)."""
        areas: dict[float, float] = {
            node: AreaModel(_base_config(node)).estimate(_base_config(node), engine_type)[
                "total_mm2"
            ]
            for node in NODES_NM
        }

        assert areas[28.0] > areas[22.0] > areas[12.0] > areas[7.0], (
            f"{engine_type}: area not monotonic across nodes: {areas}"
        )


class TestSramShare:
    """SRAM share of total area increases as the process node shrinks."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    def test_sram_share_increases_as_node_shrinks(self, engine_type: str) -> None:
        """Given identical config, SRAM/total grows from 28nm → 7nm.

        The 12nm TSMC 12FFC node uses a density-ratio correction (2.70×)
        instead of pure geometric scaling, so its logic area is smaller than
        geometric prediction and its SRAM share is higher than 7nm.  We
        therefore check the geometric-node trend (28 < 22 < 7) and treat
        12nm as a known high outlier.
        """
        shares: dict[float, float] = {}
        for node in NODES_NM:
            cfg = _base_config(node)
            total = AreaModel(cfg).estimate(cfg, engine_type)["total_mm2"]
            sram = _sram_area_mm2(node, _L1_KB, _L2_KB)
            shares[node] = sram / total

        assert shares[28.0] < shares[22.0] < shares[7.0], (
            f"{engine_type}: SRAM share not increasing across geometric nodes: {shares}"
        )
        assert shares[12.0] > shares[7.0], (
            f"{engine_type}: 12nm density correction should raise SRAM share above 7nm: {shares}"
        )


class TestRelativeRatioDirection:
    """SRAM-heavy vs SRAM-light scaling behaves differently across nodes."""

    @pytest.mark.parametrize("engine_type", ENGINE_TYPES)
    def test_sram_heavy_disadvantage_grows_at_older_nodes(self, engine_type: str) -> None:
        """Absolute SRAM-heavy area disadvantage grows monotonically at older nodes.

        Logic scales roughly quadratically with node, while SRAM scales with
        bitcell density (sub-quadratic).  The extra area consumed by a large
        L2 therefore increases from 7nm → 12nm → 22nm → 28nm.
        """
        l2_heavy_kb = 8192.0
        l2_light_kb = 512.0

        diffs: dict[float, float] = {}
        for node in NODES_NM:
            cfg_heavy = _config_with_l2(node, l2_heavy_kb)
            cfg_light = _config_with_l2(node, l2_light_kb)
            area_heavy = AreaModel(cfg_heavy).estimate(cfg_heavy, engine_type)["total_mm2"]
            area_light = AreaModel(cfg_light).estimate(cfg_light, engine_type)["total_mm2"]
            diffs[node] = area_heavy - area_light

        assert diffs[28.0] > diffs[22.0] > diffs[12.0] > diffs[7.0], (
            f"{engine_type}: SRAM-heavy absolute disadvantage did not grow "
            f"monotonically at older nodes: {diffs}"
        )


class TestNodeScaleReference:
    """Oracle node scale matches the reference ``_node_scale_factor``."""

    @pytest.mark.parametrize(
        "process_node_nm,expected",
        [
            (7.0, 1.0),
            (12.0, 2.70),
            (22.0, (22.0 / 7.0) ** 2),
            (28.0, 16.0),
        ],
    )
    def test_oracle_node_scale_matches_ppa_model(
        self, process_node_nm: float, expected: float
    ) -> None:
        """``_node_scale`` must equal ``engine.ppa_model._node_scale_factor``."""
        from engine.ppa_model import _node_scale_factor
        from tests.oracles.ppa import _node_scale

        assert _node_scale(process_node_nm) == pytest.approx(expected, rel=1e-9)
        assert _node_scale_factor(process_node_nm) == pytest.approx(expected, rel=1e-9)
        assert _node_scale(process_node_nm) == pytest.approx(
            _node_scale_factor(process_node_nm), rel=1e-9
        )
