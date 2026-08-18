"""DSE fail-closed error handling tests."""

import json
import sys

import design_space_explorer as dse
import pytest

_ORIG_EVALUATE = dse.evaluate_config


def _make_fail_once():
    """Return an evaluate_config wrapper that raises on the first call only."""
    calls = 0

    def wrapper(cfg, area_model, power_model, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected failure for test")
        return _ORIG_EVALUATE(cfg, area_model, power_model, **kwargs)

    return wrapper


def test_default_exit_nonzero_on_evaluation_error(monkeypatch, capsys):
    """Default mode exits with code 1 when any configuration raises."""
    monkeypatch.setattr(dse, "evaluate_config", _make_fail_once())
    monkeypatch.setattr(sys, "argv", ["design_space_explorer.py", "--quick"])

    with pytest.raises(SystemExit) as exc_info:
        dse.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR evaluating" in captured.err


def test_allow_partial_preserves_valid_results_and_reports_errors(monkeypatch, tmp_path):
    """--allow-partial keeps valid results and writes errors=1 into metadata."""
    monkeypatch.setattr(dse, "evaluate_config", _make_fail_once())
    output = tmp_path / "partial.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "design_space_explorer.py",
            "--quick",
            "--allow-partial",
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        dse.main()

    assert exc_info.value.code == 0

    data = json.loads(output.read_text())
    metadata = data.get("metadata", data)
    assert metadata["errors"] == 1
    assert metadata["evaluated"] >= 2
    assert metadata["valid_results"] > 0
    assert len(metadata["error_details"]) == 1
    err = metadata["error_details"][0]
    assert err["engine_type"] in ("systolic", "block", "gmma")
    assert "×" in err["dims"]
    assert "injected failure for test" in err["error"]
    assert data["valid_results"] > 0
