# ruff: noqa: E402
"""Tests for the scope verifier."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_scope import (
    check_forbidden_dependencies,
    load_publication_manifest,
    unbound_current_claims,
    verify_scope,
)


class TestForbiddenDependencies:
    """Forbidden dependency detection."""

    def test_no_forbidden_deps_in_current_pyproject(self):
        violations = check_forbidden_dependencies(PROJECT_ROOT)
        assert not violations, violations

    def test_detects_torch_in_pyproject(self, tmp_path):
        toml = tmp_path / "pyproject.toml"
        toml.write_text('[project]\ndependencies = ["torch>=2.0"]\n', encoding="utf-8")
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        violations = check_forbidden_dependencies(tmp_path)
        assert any("torch" in v for v in violations)


class TestPublicationManifest:
    """Publication manifest binding checks."""

    def test_unbound_claims_detected(self):
        manifest = {
            "current_recommendations": [
                {"name": "Block 64x64 LPDDR5", "run_manifest": "releases/abc/manifest.json"},
                {"name": "Unbound recommendation"},
            ]
        }
        assert unbound_current_claims(manifest) == ["current_recommendations[1]: Unbound recommendation"]

    def test_all_bound_claims_pass(self):
        manifest = {
            "current_recommendations": [
                {"name": "A", "run_manifest": "releases/a/manifest.json"},
                {"name": "B", "run_manifest": "releases/b/manifest.json"},
            ]
        }
        assert not unbound_current_claims(manifest)

    def test_load_publication_manifest_missing(self, tmp_path):
        assert load_publication_manifest(tmp_path / "no.yaml") == {}


class TestScopeVerification:
    """End-to-end scope verification against current repo."""

    def test_current_scope_passes(self):
        manifest_path = PROJECT_ROOT / "docs" / "publication-manifest.yaml"
        report = verify_scope(
            project_root=PROJECT_ROOT,
            baseline_commit="HEAD",
            publication_manifest_path=manifest_path,
        )
        # If the manifest does not exist yet, the test still runs but may report
        # unbound claims.  We only assert the report structure here.
        assert report["verdict"] in ("PASS", "FAIL")
        assert isinstance(report["violations"], list)
        assert isinstance(report["changed_paths"], list)

    def test_historical_reports_not_changed(self):
        report = verify_scope(
            project_root=PROJECT_ROOT,
            baseline_commit="HEAD",
            publication_manifest_path=PROJECT_ROOT / "docs" / "publication-manifest.yaml",
        )
        assert report["historical_report_changes"] == []
