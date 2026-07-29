# ruff: noqa: E402
"""Tests for the evidence-ledger verifier."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_evidence_ledger import (
    evidence_paths_for_todo,
    inspect_evidence,
    parse_plan_todos,
    verify_ledger,
)

PLAN_PATH = PROJECT_ROOT / ".omo" / "plans" / "arc-model-scenario-driven-dse-development.md"
EVIDENCE_ROOT = PROJECT_ROOT / ".omo" / "evidence"


class TestPlanParsing:
    """Plan markdown parsing."""

    def test_parses_todo_ids(self):
        todos = parse_plan_todos(PLAN_PATH.read_text(encoding="utf-8"))
        ids = {t["id"] for t in todos}
        assert "1" in ids
        assert "18" in ids
        assert "F1" in ids
        assert "F4" in ids

    def test_todo_18_unchecked(self):
        todos = parse_plan_todos(PLAN_PATH.read_text(encoding="utf-8"))
        todo_18 = next(t for t in todos if t["id"] == "18")
        assert not todo_18["checked"]


class TestEvidenceMapping:
    """Evidence file discovery."""

    def test_todo_1_evidence_exists(self):
        paths = evidence_paths_for_todo("1", EVIDENCE_ROOT)
        assert any("task-1-" in p.name for p in paths)

    def test_f1_evidence_exists(self):
        paths = evidence_paths_for_todo("F1", EVIDENCE_ROOT)
        assert paths

    def test_nonexistent_todo_has_no_evidence(self):
        paths = evidence_paths_for_todo("9999", EVIDENCE_ROOT)
        assert not paths


class TestEvidenceInspection:
    """Evidence metadata inspection."""

    def test_inspect_json_evidence(self):
        json_path = EVIDENCE_ROOT / "final-f3-manual-qa.json"
        if json_path.exists():
            info = inspect_evidence(json_path)
            assert "digest" in info
            assert info["is_json"] is True

    def test_inspect_text_evidence(self):
        paths = evidence_paths_for_todo("1", EVIDENCE_ROOT)
        txt_paths = [p for p in paths if p.suffix == ".txt"]
        if txt_paths:
            info = inspect_evidence(txt_paths[0])
            assert "digest" in info
            assert "test_counts" in info


class TestLedgerVerification:
    """End-to-end ledger verification."""

    def test_ledger_report_structure(self):
        report = verify_ledger(PLAN_PATH, EVIDENCE_ROOT)
        assert report["verdict"] in ("PASS", "FAIL")
        assert report["plan"] == str(PLAN_PATH)
        assert report["evidence_root"] == str(EVIDENCE_ROOT)
        assert isinstance(report["missing_evidence"], list)
        assert isinstance(report["entries"], list)

    def test_ledger_detects_missing_evidence_for_fake_todo(self, tmp_path):
        plan_text = "- [ ] 99999. Fake todo\n"
        plan = tmp_path / "plan.md"
        plan.write_text(plan_text, encoding="utf-8")
        report = verify_ledger(plan, EVIDENCE_ROOT)
        assert report["verdict"] == "FAIL"
        assert any("99999" in gap for gap in report["missing_evidence"])
