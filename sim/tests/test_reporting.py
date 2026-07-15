from dse.evaluator import ranking_key
from dse.reporting import (
    build_engine_comparison,
    build_engine_variant_comparison,
    print_engine_comparison,
    print_engine_variant_comparison,
)
from dse.types import DSEPoint


def _point(
    engine, tps, ttft, area, passed=True, maturity="M2", weight_cache=False,
):
    return DSEPoint(
        tok_s=tps,
        decode_tps=tps,
        aggregate_tps=tps,
        prefill_tps=1000,
        ttft_ms=ttft,
        itl_ms=1000 / max(tps, 0.01),
        area_mm2=area,
        power_w=10,
        config_label=f"{engine} test",
        config={"engine": engine, "weight_cache": weight_cache},
        constraints_passed=passed,
        maturity=maturity,
        raw_exploration_eligible=True,
        comparison_eligible=maturity in {"M2", "M3", "M4"},
        product_eligible=maturity in {"M3", "M4"},
        recommendation_eligible=maturity in {"M2", "M3", "M4"},
        failed_reasons=[] if passed else ["TPS below requirement"],
    )


def test_engine_comparison_includes_passed_and_failed_engines(capsys):
    scenario = {
        "objectives": ["-decode_tps", "ttft_ms", "area_mm2"],
        "constraints": {"decode_tps_min": 20},
    }
    rows = build_engine_comparison([
        _point("block", 20, 150, 50),
        _point("block", 30, 120, 60),
        _point("fsa", 10, 180, 45, passed=False),
    ], scenario)

    assert [row["engine"] for row in rows] == ["block", "fsa"]
    assert rows[0]["status"] == "PASS"
    assert rows[0]["metrics"]["decode_tps"] == 30
    assert rows[0]["target_status"] == "MET"
    assert rows[0]["feasible_configs"] == 2
    assert rows[1]["status"] == "FAIL"
    assert rows[1]["failed_reasons"] == ["TPS below requirement"]

    print_engine_comparison(rows)
    output = capsys.readouterr().out
    assert "Units: TPS=tok/s" in output
    assert "block" in output
    assert "fsa" in output
    assert "Failed engine details" in output


def test_weight_cache_variants_are_not_collapsed(capsys):
    scenario = {"objectives": ["area_mm2", "-decode_tps"]}
    off = _point("block", 20, 500, 40, weight_cache=False)
    on = _point("block", 25, 450, 41, weight_cache=True)

    engine_rows = build_engine_comparison([off, on], scenario)
    variant_rows = build_engine_variant_comparison([off, on], scenario)

    assert len(engine_rows) == 1
    assert len(variant_rows) == 2
    assert {row["hardware_variant"] for row in variant_rows} == {
        "WC OFF", "WC ON",
    }
    assert {row["metrics"]["area_mm2"] for row in variant_rows} == {40, 41}

    print_engine_variant_comparison(variant_rows)
    output = capsys.readouterr().out
    assert "WC ON/OFF kept separate" in output
    assert "WC OFF" in output
    assert "WC ON" in output


def test_non_weight_cache_engine_uses_not_applicable_variant():
    rows = build_engine_variant_comparison(
        [_point("os_systolic", 20, 500, 40)],
        {"objectives": ["area_mm2"]},
    )
    assert rows[0]["hardware_variant"] == "N/A"


def test_design_target_precedes_cost_but_not_hard_feasibility():
    scenario = {
        "targets": {"ttft_ms_max": 500},
        "objectives": ["area_mm2", "power_w", "-decode_tps"],
    }
    cheap_target_miss = _point("block", 25, 600, 30)
    costlier_target_met = _point("block", 25, 450, 40)
    assert ranking_key(costlier_target_met, scenario) < ranking_key(
        cheap_target_miss, scenario,
    )


def test_engine_preference_only_breaks_numeric_ties():
    scenario = {
        "objectives": ["area_mm2", "power_w", "-decode_tps"],
        "tie_breakers": {"engine_preference": ["block", "os_systolic"]},
    }
    block = _point("block", 25, 400, 40)
    output_stationary = _point("os_systolic", 25, 400, 40)
    assert ranking_key(block, scenario) < ranking_key(output_stationary, scenario)

    cheaper_os = _point("os_systolic", 25, 400, 39)
    assert ranking_key(cheaper_os, scenario) < ranking_key(block, scenario)


def test_raw_engine_ranking_does_not_penalize_lower_maturity():
    scenario = {"objectives": ["area_mm2", "-decode_tps"]}
    block = _point("block", 20, 500, 50, maturity="M2")
    fsa = _point("fsa", 30, 200, 40, maturity="M1")
    rows = build_engine_comparison([block, fsa], scenario)
    assert [row["engine"] for row in rows] == ["fsa", "block"]
    assert rows[0]["status"] == "PASS"
    assert rows[0]["eligibility_status"] == "EXPLORE"
    assert rows[0]["maturity"] == "M1"
    assert not rows[0]["comparison_eligible"]
    assert rows[1]["eligibility_status"] == "COMPARE"
