"""Cross-node area regression tests for all 8 engine types.

Validates that ``engine.ppa_model.AreaModel`` produces physically consistent
area estimates across TSMC 7/12/22/28nm nodes:

* Total area decreases monotonically as process node shrinks.
* SRAM share of total area increases as node shrinks (logic scales faster
  than SRAM).
* SRAM-heavy designs suffer a larger relative area disadvantage at older
  nodes than SRAM-light designs.
* WMMA/GMMA PE area is monotonic across nodes and keeps the physical
  ordering block < wmma < gmma (H100 SM die calibration, see plan
  wmma-gmma-pe-recalibration Todo 5/6).
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

# PE area baselines @7nm (128×128 array) — calibrated values from the
# wmma-gmma-pe-recalibration plan (Todo 5: WMMA 6.0→4.5, Todo 6: GMMA 7.0→5.5).
# block/systolic are the untouched anchors these were derived against.
_WMMA_PE_AREA_MM2 = 4.5
_GMMA_PE_AREA_MM2 = 5.5
_BLOCK_PE_AREA_MM2 = 4.0
_SYSTOLIC_PE_AREA_MM2 = 2.0


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


def _pe_area_config(process_node_nm: float) -> dict[str, Any]:
    """Base config with the calibrated WMMA/GMMA PE baselines injected.

    ``AreaModel`` reads ``*_pe_area_mm2`` from the config (falling back to the
    pre-calibration 6.0/7.0 code defaults), so the Todo 5/6 calibrated values
    must be passed explicitly to lock the current calibration in place.
    """
    cfg = _base_config(process_node_nm)
    cfg["area_model"].update(
        {
            "systolic_pe_area_mm2": _SYSTOLIC_PE_AREA_MM2,
            "block_pe_area_mm2": _BLOCK_PE_AREA_MM2,
            "wmma_pe_area_mm2": _WMMA_PE_AREA_MM2,
            "gmma_pe_area_mm2": _GMMA_PE_AREA_MM2,
        }
    )
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
            node: AreaModel(_base_config(node)).estimate(_base_config(node), engine_type)["total_mm2"]
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
            f"{engine_type}: SRAM-heavy absolute disadvantage did not grow monotonically at older nodes: {diffs}"
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
    def test_oracle_node_scale_matches_ppa_model(self, process_node_nm: float, expected: float) -> None:
        """``_node_scale`` must equal ``engine.ppa_model._node_scale_factor``."""
        from engine.ppa_model import _node_scale_factor
        from tests.oracles.ppa import _node_scale

        assert _node_scale(process_node_nm) == pytest.approx(expected, rel=1e-9)
        assert _node_scale_factor(process_node_nm) == pytest.approx(expected, rel=1e-9)
        assert _node_scale(process_node_nm) == pytest.approx(_node_scale_factor(process_node_nm), rel=1e-9)


class TestWmmaGmmaPeArea:
    """WMMA/GMMA PE area regression locks from the H100 SM die calibration.

    Todo 5/6 of the wmma-gmma-pe-recalibration plan set WMMA PE = 4.5 mm² and
    GMMA PE = 5.5 mm² @7nm, anchored against block = 4.0 mm² / systolic =
    2.0 mm².  These tests lock the calibrated ordering across every node.
    """

    def test_wmma_area_per_node(self) -> None:
        """WMMA PE area decreases monotonically: 28nm > 22nm > 12nm > 7nm."""
        areas: dict[float, float] = {
            node: AreaModel(_pe_area_config(node)).wmma_pe_baseline for node in NODES_NM
        }
        assert areas[28.0] > areas[22.0] > areas[12.0] > areas[7.0], (
            f"wmma: PE area not monotonic across nodes: {areas}"
        )

    def test_gmma_area_per_node(self) -> None:
        """GMMA PE area decreases monotonically: 28nm > 22nm > 12nm > 7nm."""
        areas: dict[float, float] = {
            node: AreaModel(_pe_area_config(node)).gmma_pe_baseline for node in NODES_NM
        }
        assert areas[28.0] > areas[22.0] > areas[12.0] > areas[7.0], (
            f"gmma: PE area not monotonic across nodes: {areas}"
        )

    @pytest.mark.parametrize("process_node_nm", NODES_NM)
    def test_gmma_ge_wmma(self, process_node_nm: float) -> None:
        """GMMA PE area exceeds WMMA PE area at every node (TMA premium)."""
        model = AreaModel(_pe_area_config(process_node_nm))
        assert model.gmma_pe_baseline > model.wmma_pe_baseline, (
            f"{process_node_nm}nm: gmma {model.gmma_pe_baseline} <= wmma {model.wmma_pe_baseline}"
        )

    @pytest.mark.parametrize("process_node_nm", NODES_NM)
    def test_wmma_gmma_area_physically_plausible(self, process_node_nm: float) -> None:
        """block < wmma < gmma at every node (WMMA sits between the anchors)."""
        model = AreaModel(_pe_area_config(process_node_nm))
        assert model.block_pe_baseline < model.wmma_pe_baseline < model.gmma_pe_baseline, (
            f"{process_node_nm}nm: block {model.block_pe_baseline}, "
            f"wmma {model.wmma_pe_baseline}, gmma {model.gmma_pe_baseline} out of order"
        )
