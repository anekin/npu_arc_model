#!/usr/bin/env python3
"""Scope and evidence-fidelity verifier.

Audits the working tree against the plan's Must have/Must NOT have:
  - forbidden dependencies (PyTorch, ROS, Ramulator, DRAMSim)
  - historical dated reports unchanged
  - ultraresearch sources not staged
  - current recommendations bound to a release manifest
  - changed paths relative to baseline commit

Outputs structured JSON and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

FORBIDDEN_DEPENDENCIES = ("torch", "ros", "ramulator", "dramsim")
HISTORICAL_REPORTS = (
    "reports/dse-engine-model-bugs-2026-07-27.md",
    "reports/dse-engine-model-bugs-postfix-2026-07-27.md",
)
ULTRARESEARCH_PATH = ".omo/ultraresearch/20260723-vla-models/sources/"


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def git_changed_paths(baseline_commit: str) -> list[str]:
    """Return list of paths changed since baseline commit."""
    out = _run_git("diff", "--name-only", baseline_commit, "--")
    if not out:
        return []
    return out.splitlines()


def git_status_short() -> list[str]:
    """Return git status --short lines."""
    out = _run_git("status", "--short")
    if not out:
        return []
    return out.splitlines()


def historical_report_changes() -> list[str]:
    """Return changed historical report paths."""
    changed = git_changed_paths("HEAD")
    return [p for p in changed if p in HISTORICAL_REPORTS]


def ultraresearch_staged(status_lines: list[str]) -> list[str]:
    """Return ultraresearch source paths that appear in git status."""
    return [line for line in status_lines if ULTRARESEARCH_PATH in line]


def check_forbidden_dependencies(project_root: Path) -> list[str]:
    """Check pyproject.toml and uv.lock for forbidden dependencies."""
    violations: list[str] = []
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8").lower()
        for dep in FORBIDDEN_DEPENDENCIES:
            if re.search(rf"\b{re.escape(dep)}\b", text):
                violations.append(f"pyproject.toml contains forbidden dependency: {dep}")
    lock = project_root / "uv.lock"
    if lock.exists():
        text = lock.read_text(encoding="utf-8").lower()
        for dep in FORBIDDEN_DEPENDENCIES:
            if re.search(rf"\b{re.escape(dep)}\b", text):
                violations.append(f"uv.lock contains forbidden dependency: {dep}")
    return violations


def load_publication_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def unbound_current_claims(manifest: dict[str, Any]) -> list[str]:
    """Return current recommendations lacking a run_manifest reference."""
    unbound: list[str] = []
    for idx, rec in enumerate(manifest.get("current_recommendations", [])):
        if not rec.get("run_manifest"):
            unbound.append(f"current_recommendations[{idx}]: {rec.get('name', '?')}")
    return unbound


def verify_scope(
    project_root: Path,
    baseline_commit: str,
    publication_manifest_path: Path,
) -> dict[str, Any]:
    status_lines = git_status_short()
    changed_paths = git_changed_paths(baseline_commit)

    historical_changes = historical_report_changes()
    ultraresearch_changes = ultraresearch_staged(status_lines)
    forbidden = check_forbidden_dependencies(project_root)
    manifest = load_publication_manifest(publication_manifest_path)
    unbound = unbound_current_claims(manifest)

    violations = forbidden + historical_changes + ultraresearch_changes + unbound

    # Any path outside the expected implementation set is flagged but not
    # treated as a hard violation unless it is a forbidden dependency or
    # historical-report change.
    out_of_scope_paths = [
        p
        for p in changed_paths
        if not any(
            p.startswith(prefix)
            for prefix in (
                "sim/",
                "scripts/",
                "docs/",
                "reports/README.md",
                "artifacts/releases/",
                ".omo/notepads/",
                ".omo/evidence/",
            )
        )
    ]

    verdict = "PASS" if not violations else "FAIL"

    return {
        "verdict": verdict,
        "baseline_commit": baseline_commit,
        "publication_manifest": str(publication_manifest_path),
        "violations": violations,
        "forbidden_dependencies": forbidden,
        "historical_report_changes": historical_changes,
        "ultraresearch_changes": ultraresearch_changes,
        "unbound_current_claims": unbound,
        "changed_paths": changed_paths,
        "out_of_scope_paths": out_of_scope_paths,
        "git_status": status_lines,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify scope and evidence fidelity")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--baseline-commit", default="HEAD")
    parser.add_argument("--publication-manifest", type=Path, default=Path("docs/publication-manifest.yaml"))
    parser.add_argument("--plan", type=Path, default=Path(".omo/plans/arc-model-scenario-driven-dse-development.md"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = verify_scope(
        project_root=args.project_root,
        baseline_commit=args.baseline_commit,
        publication_manifest_path=args.publication_manifest,
    )
    report["plan"] = str(args.plan)

    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
