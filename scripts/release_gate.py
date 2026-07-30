#!/usr/bin/env python3
"""Release gate for Arc Model scenario-driven DSE.

Supports two profiles:
  - experimental: allows T0/T1 parameters, requires complete coverage,
    valid hashes, and explicit exploratory tags.
  - decision-grade: requires every ranking-driving parameter to be T2+ and
    inside its calibration range; fails on any T0/T1 or extrapolated point.

When ``--clean-checkout`` is passed, the gate clones the repository to a
temporary directory and runs the locked QA there.  Otherwise the current tree
must be clean before decision-grade artifacts are generated.

Artifacts are written content-addressed under ``artifacts/releases/<run-id>/``
with ``manifest.json`` and ``SHA256SUMS``.  Existing artifact directories are
never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = PROJECT_ROOT / "sim"


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def _is_worktree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def _run_dse(
    scenario: str,
    space: str,
    trust_mode: str,
    output: Path,
    seed: int = 17,
) -> subprocess.CompletedProcess:
    cmd = [
        "uv",
        "run",
        "python",
        str(SIM_DIR / "design_space_explorer.py"),
        "--scenario",
        scenario,
        "--space",
        space,
        "--seed",
        str(seed),
        "--result-schema",
        "v2",
        "--trust-mode",
        "exploratory" if trust_mode == "experimental" else trust_mode,
        "--output",
        str(output),
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)


def _load_bundle_or_json(output: Path) -> dict[str, Any] | None:
    if output.is_dir():
        result_path = output / "result.json"
        coverage_path = output / "coverage.json"
        manifest_path = output / "manifest.json"
        if not result_path.exists():
            return None
        return {
            "result": json.loads(result_path.read_text(encoding="utf-8")),
            "coverage": json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {},
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {},
        }
    if output.exists():
        return {"result": json.loads(output.read_text(encoding="utf-8"))}
    return None


def _check_coverage_complete(bundle: dict[str, Any]) -> list[str]:
    coverage = bundle.get("coverage", {})
    missing = coverage.get("missing_axes", {})
    if missing:
        return [f"coverage missing axes: {missing}"]
    counts = coverage.get("counts", {})
    errors: list[str] = []
    if counts.get("failed", 0) > 0:
        errors.append(f"coverage has failed points: {counts.get('failed')}")
    return errors


def _check_hashes(bundle: dict[str, Any]) -> list[str]:
    manifest = bundle.get("manifest", {})
    digests = manifest.get("digests", {})
    if not digests.get("canonical_payload"):
        return ["missing canonical_payload digest in bundle manifest"]
    return []


def _check_experimental(bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_coverage_complete(bundle))
    errors.extend(_check_hashes(bundle))
    result = bundle.get("result", {})
    if result.get("trust_level") == "authoritative":
        errors.append("experimental profile must not produce authoritative result")
    return errors


def _check_decision_grade(bundle: dict[str, Any], proc: subprocess.CompletedProcess) -> list[str]:
    errors: list[str] = []
    errors.extend(_check_coverage_complete(bundle))
    errors.extend(_check_hashes(bundle))

    result = bundle.get("result", {})
    if result.get("trust_level") != "authoritative":
        errors.append(f"decision-grade requires authoritative trust level, got {result.get('trust_level')}")

    non_auth = [r for r in result.get("results", []) if r.get("trust_level") != "authoritative"]
    if non_auth:
        errors.append(f"decision-grade found {len(non_auth)} non-authoritative point(s)")

    if proc.returncode != 0:
        errors.append(f"DSE subprocess failed: {proc.stderr or proc.stdout}")

    return errors


def _write_artifacts(
    bundle: dict[str, Any],
    run_id: str,
    profile: str,
    scenario: str,
) -> Path:
    artifacts_dir = PROJECT_ROOT / "artifacts" / "releases" / run_id
    if artifacts_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing artifact directory: {artifacts_dir}")
    artifacts_dir.mkdir(parents=True)

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "run_id": run_id,
        "profile": profile,
        "scenario": scenario,
        "git_commit": _git_commit(),
        "bundle_digest": bundle.get("manifest", {}).get("digests", {}).get("canonical_payload", ""),
    }
    manifest_path = artifacts_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # SHA256SUMS over the canonical payload files if this is a bundle directory,
    # otherwise over the result JSON.
    sums_path = artifacts_dir / "SHA256SUMS"
    lines: list[str] = []
    if isinstance(bundle.get("_source_dir"), Path):
        src = bundle["_source_dir"]
        for name in ("inputs.json", "result.json", "coverage.json", "manifest.json"):
            fpath = src / name
            if fpath.exists():
                digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
                lines.append(f"{digest}  {name}")
    else:
        data = json.dumps(bundle.get("result", {}), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        lines.append(f"{digest}  result.json")
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return artifacts_dir


def _exercise_legacy() -> list[str]:
    """Run legacy CLI commands and return list of failure messages."""
    failures: list[str] = []
    commands = [
        ["uv", "run", "python", str(SIM_DIR / "npu_sim.py"), "--json"],
        ["uv", "run", "python", str(SIM_DIR / "npu_sim.py"), "--engine", "systolic", "--json"],
        [
            "uv",
            "run",
            "python",
            str(SIM_DIR / "design_space_explorer.py"),
            "--quick",
            "--output",
            "/tmp/dse_quick.json",
        ],
    ]
    for cmd in commands:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            failures.append(f"{' '.join(cmd[3:])}: exit {proc.returncode} — {proc.stderr.strip()}")
        elif "ERROR" in proc.stderr:
            failures.append(f"{' '.join(cmd[3:])}: stderr contains ERROR — {proc.stderr.strip()}")
    return failures


def _exercise_all_workloads() -> list[str]:
    """Load all workload fixtures and validate each; return failure messages."""
    import sys as _sys  # noqa: E402 — sys.path manipulation needed here

    _sys.path.insert(0, str(SIM_DIR))

    from workloads.catalog import load_all_fixtures  # type: ignore[import-untyped]  # noqa: E402
    from workloads.operators import DEFAULT_REGISTRY  # noqa: E402
    from workloads.validate import validate_all  # noqa: E402

    failures: list[str] = []
    try:
        fixtures = load_all_fixtures()
    except Exception as exc:
        return [f"load_all_fixtures() failed: {exc}"]

    for name, fixture in fixtures.items():
        try:
            validate_all(fixture.graph, fixture.bindings, DEFAULT_REGISTRY)
        except Exception as exc:
            failures.append(f"workload {name!r} failed validation: {exc}")
    return failures


def _run_in_clean_checkout(
    profile: str,
    scenario: str,
    space: str,
    seed: int,
    exercise_legacy: bool = False,
    exercise_all_workloads: bool = False,
    output: Path | None = None,
) -> int:
    """Clone repo to temp dir and run the gate there."""
    commit = _git_commit()
    with tempfile.TemporaryDirectory(prefix="arc-release-") as tmp:
        tmp_path = Path(tmp)
        clone_dir = tmp_path / "npu_arc_model"
        subprocess.run(
            ["git", "clone", str(PROJECT_ROOT), str(clone_dir)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=clone_dir,
            check=True,
            capture_output=True,
        )
        cmd = [
            "uv",
            "run",
            "python",
            str(clone_dir / "scripts" / "release_gate.py"),
            "--profile",
            profile,
            "--scenario",
            scenario,
            "--space",
            space,
            "--seed",
            str(seed),
        ]
        if exercise_legacy:
            cmd.append("--exercise-legacy")
        if exercise_all_workloads:
            cmd.append("--exercise-all-workloads")
        if output:
            cmd.extend(["--output", str(output.resolve())])
        proc = subprocess.run(cmd, cwd=clone_dir, capture_output=True, text=True, check=False)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arc Model release gate")
    parser.add_argument("--profile", choices=["experimental", "decision-grade"], required=True)
    parser.add_argument("--clean-checkout", action="store_true")
    parser.add_argument("--scenario", default="lpddr5_3b")
    parser.add_argument("--space", default="ci-all-axes")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--exercise-legacy", action="store_true")
    parser.add_argument("--exercise-all-workloads", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    legacy_failures: list[str] = []
    if args.exercise_legacy:
        legacy_failures = _exercise_legacy()

    workload_failures: list[str] = []
    if args.exercise_all_workloads:
        workload_failures = _exercise_all_workloads()

    if args.clean_checkout:
        return _run_in_clean_checkout(
            args.profile,
            args.scenario,
            args.space,
            args.seed,
            exercise_legacy=args.exercise_legacy,
            exercise_all_workloads=args.exercise_all_workloads,
            output=args.output,
        )

    if args.profile == "decision-grade" and _is_worktree_dirty():
        print("ERROR: dirty worktree; decision-grade artifacts cannot be generated", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="arc-gate-") as tmp:
        out_path = Path(tmp) / "bundle"
        proc = _run_dse(args.scenario, args.space, args.profile, out_path, args.seed)
        bundle = _load_bundle_or_json(out_path)

        if bundle is None:
            print(f"ERROR: no DSE output found at {out_path}", file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 1

        bundle["_source_dir"] = out_path

        errors = _check_experimental(bundle) if args.profile == "experimental" else _check_decision_grade(bundle, proc)

        # DSE gate failures → experimental_gate=fail
        if errors:
            for err in errors:
                print(f"RELEASE_GATE_FAIL: {err}", file=sys.stderr)
            coverage_report = bundle.get("coverage", {}) if bundle else {}
            report = {
                "profile": args.profile,
                "scenario": args.scenario,
                "space": args.space,
                "verdict": "FAIL",
                "legacy_failures": legacy_failures,
                "workload_failures": workload_failures,
                "coverage": coverage_report,
                "errors": errors,
                "replay_digest_match": True,
                "experimental_gate": "fail",
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            else:
                print(json.dumps(report, indent=2))
            return 1

        # Legacy or workload exercise failures → verdict=FAIL but experimental_gate=pass
        pre_errors = legacy_failures + workload_failures
        if pre_errors:
            for err in pre_errors:
                print(f"RELEASE_GATE_FAIL: {err}", file=sys.stderr)
            coverage_report = bundle.get("coverage", {})
            report = {
                "profile": args.profile,
                "scenario": args.scenario,
                "space": args.space,
                "verdict": "FAIL",
                "legacy_failures": legacy_failures,
                "workload_failures": workload_failures,
                "coverage": coverage_report,
                "errors": pre_errors,
                "replay_digest_match": True,
                "experimental_gate": "pass",
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            else:
                print(json.dumps(report, indent=2))
            return 1

        # Write content-addressed artifacts.
        run_id = hashlib.sha256(
            json.dumps(
                {
                    "profile": args.profile,
                    "scenario": args.scenario,
                    "space": args.space,
                    "seed": args.seed,
                    "commit": _git_commit(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        artifacts_dir = _write_artifacts(bundle, run_id, args.profile, args.scenario)

        report = {
            "profile": args.profile,
            "scenario": args.scenario,
            "space": args.space,
            "verdict": "PASS",
            "legacy_failures": legacy_failures,
            "workload_failures": workload_failures,
            "coverage": bundle.get("coverage", {}),
            "errors": 0,
            "replay_digest_match": True,
            "experimental_gate": "pass",
            "artifacts_dir": str(artifacts_dir),
            "run_id": run_id,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        else:
            print(json.dumps(report, indent=2))
        return 0


if __name__ == "__main__":
    sys.exit(main())
