from dse.evaluator import ranking_key
from dse.reporting import build_engine_comparison, print_engine_comparison
from dse.types import DSEPoint


def _point(engine, tps, ttft, area, passed=True):
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
        config={"engine": engine},
        constraints_passed=passed,
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

def test_research_engine_is_visible_but_ranked_after_recommendable_engine():
    scenario = {"objectives": ["area_mm2", "-decode_tps"]}
    block = _point("block", 20, 500, 50)
    fsa = _point("fsa", 30, 200, 40)
    fsa.recommendation_eligible = False
    rows = build_engine_comparison([block, fsa], scenario)
    assert [row["engine"] for row in rows] == ["block", "fsa"]
    assert rows[1]["status"] == "PASS"
    assert rows[1]["eligibility_status"] == "RESEARCH"
    assert not rows[1]["recommendation_eligible"]
