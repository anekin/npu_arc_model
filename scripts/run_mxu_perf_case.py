#!/usr/bin/env python3
# allow: SIZE_OK — single-file orchestration runner mandated by the task;
# splitting would force callers to manage multiple modules for one command.
"""
MXU Module-Level Performance Case Runner.

End-to-end runner for a single MXU performance case.  Embeds EDA-server SSH
and VCS module load so any agent can run one case with a single command and
get a deterministic PASS/FAIL result.

Usage:
    python3 CaduceusCore/scripts/run_mxu_perf_case.py --case MX-P01 --shape 64,64,64
    python3 CaduceusCore/scripts/run_mxu_perf_case.py --case MX-P01 --shape 64,64,64 --commit
    python3 CaduceusCore/scripts/run_mxu_perf_case.py --case MX-P01 --shape 64,64,64 --dry-run
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


# ══════════════════════════════════════════════════════════════════════
# Paths and constants
# ══════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
CADUCEUS_CORE = SCRIPT_DIR.parent
REPO_ROOT = CADUCEUS_CORE

TESTCASE_LIST = CADUCEUS_CORE / "rtl" / "testcase-list-mxu-perf.md"
LEARNINGS_FILE = REPO_ROOT / ".omo" / "notepads" / "mxu-module-perf" / "learnings.md"

DEFAULT_OUT_DIR = REPO_ROOT / "build/mxu_perf_cases"
DEFAULT_SIMV = REPO_ROOT / "build/simv_mxu_perf"
DEFAULT_EDA_SERVER = "zhengs@192.168.0.11"
DEFAULT_VCS_MODULE = "vcs/vcs_vW-2024.09-SP2_P"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / ".omo" / "evidence" / "mxu-perf"

VCS_SETUP = "source /NAS/Tools/methodology/modules/init/bash && module load {vcs_module}"


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    """Parse ``M,N,K`` into a tuple of integers."""
    parts = shape_str.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"shape must be M,N,K (3 integers, got {len(parts)}: {shape_str!r})"
        )
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"shape values must be integers: {shape_str!r}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="MXU module-level performance case runner (one command, one PASS/FAIL).",
    )
    parser.add_argument("--case", required=True, help="Case ID, e.g. MX-P01")
    parser.add_argument(
        "--shape", required=True, type=parse_shape, help="Shape as M,N,K, e.g. 64,64,64"
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Output base directory (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--simv", default=str(DEFAULT_SIMV), help=f"VCS simv binary path on EDA server (default: {DEFAULT_SIMV})"
    )
    parser.add_argument("--rebuild", action="store_true", help="Force VCS recompile")
    parser.add_argument(
        "--eda-server", default=DEFAULT_EDA_SERVER, help=f"EDA server SSH target (default: {DEFAULT_EDA_SERVER})"
    )
    parser.add_argument(
        "--vcs-module", default=DEFAULT_VCS_MODULE, help=f"VCS module name (default: {DEFAULT_VCS_MODULE})"
    )
    parser.add_argument(
        "--commit", action="store_true", help="Commit testcase-list status change (default: off)"
    )
    parser.add_argument(
        "--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR), help=f"Evidence directory (default: {DEFAULT_EVIDENCE_DIR})"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate vectors + formula check only; no VCS"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of back-to-back CMD ops via +repeat+ plusarg (default: 1)",
    )
    return parser


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════


def run_cmd(
    cmd: Sequence[str | Path],
    *,
    check: bool = True,
    capture: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, optionally capturing output."""
    cmd_strs = [str(c) for c in cmd]
    print(f"[run] {' '.join(cmd_strs)}")
    result = subprocess.run(
        cmd_strs,
        cwd=cwd,
        check=False,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd_strs, output=result.stdout, stderr=result.stderr
        )
    return result


def run_ssh(eda_server: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command on the EDA server over SSH."""
    return run_cmd(["ssh", eda_server, command], check=check)


def scp_from_remote(eda_server: str, remote_path: Path, local_path: Path) -> None:
    """Copy a file from the EDA server to the local machine."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["scp", f"{eda_server}:{remote_path}", str(local_path)])


def scp_to_remote(eda_server: str, local_path: Path, remote_path: Path) -> None:
    """Copy a file or directory from the local machine to the EDA server."""
    run_cmd(["scp", "-r", str(local_path), f"{eda_server}:{remote_path}"])


def file_exists_on_remote(eda_server: str, path: Path) -> bool:
    """Return True if ``path`` exists on the EDA server."""
    result = run_ssh(
        eda_server,
        f"test -e {shlex.quote(str(path))} && echo YES || echo NO",
        check=True,
    )
    return result.stdout.strip() == "YES"


def is_shared_path(path: Path) -> bool:
    """Return True if ``path`` is under the shared project root."""
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


def tail(path: Path, n: int = 30) -> str:
    """Return the last ``n`` lines of a file, or a warning if missing."""
    if not path.exists():
        return f"<file not found: {path}>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def has_vcs_errors(log_path: Path) -> bool:
    """Return True if the VCS log contains Error lines."""
    if not log_path.exists():
        return True
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        # VCS errors: "Error-[CODE]" or leading "Error"
        if line.startswith("Error") or "Error-[" in line:
            return True
    return False


def parse_perf_numbers(analyze_output: str, case_id: str) -> tuple[str | None, str | None]:
    """Extract expected and measured cycle counts from analyze_perf.py output."""
    # [MX-P01] shape=64,64,64 expected=134 measured=134 delta=0 PASS
    pattern = re.compile(
        rf"\[{re.escape(case_id)}\]\s+shape=\d+,\d+,\d+\s+"
        r"expected=(?P<expected>\d+)\s+measured=(?P<measured>\d+)"
    )
    for line in analyze_output.splitlines():
        m = pattern.search(line)
        if m:
            return m.group("expected"), m.group("measured")
    return None, None


# ══════════════════════════════════════════════════════════════════════
# Core steps
# ══════════════════════════════════════════════════════════════════════


def step_generate_vectors(out_dir: Path, case: str, shape: tuple[int, int, int]) -> Path:
    """Generate golden test vectors for the requested shape."""
    M, N, K = shape
    scenario_name = f"shape_{M}_{N}_{K}"
    case_out_dir = out_dir / case
    scenario_dir = case_out_dir / scenario_name

    run_cmd(
        [
            "python3",
            str(CADUCEUS_CORE / "scripts" / "gen_mxu_vectors.py"),
            "--shape",
            f"{M},{N},{K}",
            "--out-dir",
            str(case_out_dir),
        ],
        cwd=REPO_ROOT,
    )

    if not scenario_dir.exists():
        raise RuntimeError(f"Vector generation did not create {scenario_dir}")

    required = ["weights.hex", "activations.hex", "golden_output.hex", "params.txt", "manifest.json"]
    missing = [f for f in required if not (scenario_dir / f).exists()]
    if missing:
        raise RuntimeError(f"Missing generated files in {scenario_dir}: {missing}")

    return scenario_dir


def step_ensure_vectors_on_eda(
    eda_server: str, scenario_dir: Path, out_dir: Path, case: str
) -> None:
    """Copy vectors to the EDA server when the output directory is not shared."""
    if is_shared_path(out_dir):
        print(f"[info] out-dir {out_dir} is shared; skipping vector SCP")
        return

    remote_scenario_dir = Path(out_dir) / case / scenario_dir.name
    remote_case_dir = remote_scenario_dir.parent
    print(f"[scp] copying vectors to {eda_server}:{remote_scenario_dir}")
    run_ssh(eda_server, f"mkdir -p {shlex.quote(str(remote_case_dir))}")
    scp_to_remote(eda_server, scenario_dir, remote_case_dir)


def step_compile_vcs(
    eda_server: str, vcs_module: str, simv: Path, force: bool
) -> Path:
    """Compile the MXU performance testbench on the EDA server if needed."""
    compile_log = Path(f"{simv}.compile.log")

    if not force and file_exists_on_remote(eda_server, simv):
        print(f"[info] reusing existing simv binary {simv}")
        return compile_log

    print(f"[vcs] compiling {simv} on {eda_server}")
    setup = VCS_SETUP.format(vcs_module=shlex.quote(vcs_module))
    cmd = (
        f"{setup} && "
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"vcs -full64 -sverilog -debug_access+all -timescale=1ns/1ps "
        f"rtl/tb/tb_mxu_perf.v rtl/mxu/*.v "
        f"-top tb_mxu_perf -o {shlex.quote(str(simv))} -l {shlex.quote(str(compile_log))}"
    )
    run_ssh(eda_server, cmd)

    if not file_exists_on_remote(eda_server, simv):
        raise RuntimeError(f"VCS compile did not produce simv binary {simv}")

    return compile_log


def step_run_simulation(
    eda_server: str,
    vcs_module: str,
    simv: Path,
    case: str,
    scenario_dir: Path,
    scenario_name: str,
    repeat: int = 1,
) -> Path:
    """Run the compiled simv on the EDA server."""
    sim_log_remote = Path(f"{simv}.{case}.log")

    setup = VCS_SETUP.format(vcs_module=shlex.quote(vcs_module))
    cmd = (
        f"{setup} && "
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{shlex.quote(str(simv))} +case={shlex.quote(case)} "
        f"+testdir={shlex.quote(str(scenario_dir))} "
        f"+scenario={shlex.quote(scenario_name)} "
        f"+repeat={shlex.quote(str(repeat))} "
        f"-l {shlex.quote(str(sim_log_remote))}"
    )
    run_ssh(eda_server, cmd)

    return sim_log_remote


def step_compare_rtl(scenario_dir: Path, scenario_name: str) -> subprocess.CompletedProcess[str]:
    """Run compare_rtl.py against the RTL result hex file."""
    result_hex = CADUCEUS_CORE / "rtl" / "results" / f"mxu_{scenario_name}.hex"
    return run_cmd(
        [
            "python3",
            str(CADUCEUS_CORE / "sim" / "compare_rtl.py"),
            str(scenario_dir),
            str(result_hex),
        ],
        cwd=REPO_ROOT,
    )


def step_analyze_perf(
    case: str, shape: tuple[int, int, int], sim_log_local: Path
) -> subprocess.CompletedProcess[str]:
    """Run analyze_perf.py to check measured vs expected cycles."""
    M, N, K = shape
    return run_cmd(
        [
            "python3",
            str(CADUCEUS_CORE / "scripts" / "analyze_perf.py"),
            "--case",
            case,
            "--shape",
            f"{M},{N},{K}",
            "--log",
            str(sim_log_local),
        ],
        cwd=REPO_ROOT,
    )


def step_dry_run_analyze(case: str, shape: tuple[int, int, int]) -> subprocess.CompletedProcess[str]:
    """Run analyze_perf.py in dry-run mode (formula check only)."""
    M, N, K = shape
    return run_cmd(
        [
            "python3",
            str(CADUCEUS_CORE / "scripts" / "analyze_perf.py"),
            "--dry-run",
            "--case",
            case,
            "--shape",
            f"{M},{N},{K}",
        ],
        cwd=REPO_ROOT,
    )


def update_testcase_list(case: str, measured: str, expected: str) -> None:
    """Update the testcase list status column for ``case``."""
    if not TESTCASE_LIST.exists():
        raise RuntimeError(f"Testcase list not found: {TESTCASE_LIST}")

    text = TESTCASE_LIST.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated = False

    for i, line in enumerate(lines):
        if line.startswith(f"| {case} "):
            # Replace the first occurrence of ⬜ on the case line with ✅
            if "⬜" in line:
                lines[i] = line.replace("⬜", "✅", 1)
                updated = True
                break

    if not updated:
        raise RuntimeError(f"Could not find TODO status for {case} in {TESTCASE_LIST}")

    TESTCASE_LIST.write_text("\n".join(lines) + "\n", encoding="utf-8")


def commit_status_change(case: str, measured: str, expected: str) -> None:
    """Commit the testcase list status change in the CaduceusCore submodule."""
    msg = f"[{case}] ⬜ → ✅ | measured={measured} expected={expected}"
    run_cmd(["git", "-C", str(CADUCEUS_CORE), "add", str(TESTCASE_LIST.relative_to(CADUCEUS_CORE))])
    run_cmd(["git", "-C", str(CADUCEUS_CORE), "commit", "-m", msg])


def append_learning(case: str, shape: tuple[int, int, int], passed: bool) -> None:
    """Append a summary line to the mxu-module-perf learnings notepad."""
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "PASS" if passed else "FAIL"
    M, N, K = shape
    line = (
        f"\n## {timestamp} run_mxu_perf_case.py — {case} shape={M},{N},{K} — {status}\n"
    )
    LEARNINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LEARNINGS_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


# ══════════════════════════════════════════════════════════════════════
# Main flow
# ══════════════════════════════════════════════════════════════════════


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    case: str = args.case
    shape: tuple[int, int, int] = args.shape
    M, N, K = shape
    scenario_name = f"shape_{M}_{N}_{K}"
    out_dir = Path(args.out_dir).resolve()
    simv = Path(args.simv).resolve()
    eda_server: str = args.eda_server
    vcs_module: str = args.vcs_module
    evidence_dir = Path(args.evidence_dir).resolve()
    dry_run: bool = args.dry_run
    repeat: int = args.repeat

    evidence_dir.mkdir(parents=True, exist_ok=True)
    # Include repeat in filenames when repeat > 1 to avoid collision
    repeat_suffix = f"_r{repeat}" if repeat > 1 else ""
    evidence_file = evidence_dir / f"{case}{repeat_suffix}.txt"
    compile_log_local = evidence_dir / f"{case}{repeat_suffix}_compile.log"
    sim_log_local = evidence_dir / f"{case}{repeat_suffix}_sim.log"

    command_used = " ".join(sys.argv)

    evidence_parts: list[str] = [
        f"Case: {case}",
        f"Shape: {M},{N},{K}",
        f"Scenario: {scenario_name}",
        f"Command: {command_used}",
        "",
    ]

    try:
        # ── Step a/b: generate vectors ─────────────────────────────────
        scenario_dir = step_generate_vectors(out_dir, case, shape)

        if dry_run:
            # Dry-run: formula check only
            analyze_result = step_dry_run_analyze(case, shape)
            evidence_parts.extend(
                [
                    "=== DRY-RUN MODE (no VCS) ===",
                    "",
                    "--- analyze_perf.py output ---",
                    analyze_result.stdout.strip(),
                    "",
                    "Final verdict: PASS (dry-run)",
                ]
            )
            evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
            append_learning(case, shape, passed=True)
            print(f"[evidence] wrote {evidence_file}")
            return 0

        # ── Step c: VCS compile ────────────────────────────────────────
        step_ensure_vectors_on_eda(eda_server, scenario_dir, out_dir, case)
        compile_log_remote = step_compile_vcs(eda_server, vcs_module, simv, args.rebuild)
        scp_from_remote(eda_server, compile_log_remote, compile_log_local)

        if has_vcs_errors(compile_log_local):
            evidence_parts.extend(
                [
                    "--- VCS compile log tail ---",
                    tail(compile_log_local, 30),
                    "",
                    "Final verdict: FAIL (VCS compile errors)",
                ]
            )
            evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
            print(f"ERROR: VCS compile produced errors (see {compile_log_local})", file=sys.stderr)
            return 1

        # ── Step d: run simulation ─────────────────────────────────────
        sim_log_remote = step_run_simulation(
            eda_server, vcs_module, simv, case, scenario_dir, scenario_name, repeat
        )
        scp_from_remote(eda_server, sim_log_remote, sim_log_local)

        if not sim_log_local.exists():
            raise RuntimeError(f"Simulation log was not copied back: {sim_log_local}")

        sim_text = sim_log_local.read_text(encoding="utf-8", errors="replace")
        if "PERF|" not in sim_text:
            evidence_parts.extend(
                [
                    "--- VCS compile log tail ---",
                    tail(compile_log_local, 30),
                    "",
                    "--- Simulation log tail ---",
                    tail(sim_log_local, 30),
                    "",
                    "Final verdict: FAIL (no PERF| lines in simulation log)",
                ]
            )
            evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
            print("ERROR: simulation log contains no PERF| lines", file=sys.stderr)
            return 1

        # ── Step e: bit-exact compare ──────────────────────────────────
        # When repeat > 1, the result hex is overwritten by back-to-back ops
        if repeat == 1:
            compare_result = step_compare_rtl(scenario_dir, scenario_name)
            compare_stdout = compare_result.stdout.strip()
            compare_pass = "[PASS]" in compare_stdout and "[FAIL]" not in compare_stdout
        else:
            compare_stdout = f"(skipped: repeat={repeat}, result hex overwritten by back-to-back ops)"
            compare_pass = True  # Accept PERF-only verification for repeat mode

        # ── Step f: cycle analysis ─────────────────────────────────────
        analyze_result = step_analyze_perf(case, shape, sim_log_local)
        analyze_stdout = analyze_result.stdout.strip()
        analyze_pass = "PASS" in analyze_stdout and "FAIL" not in analyze_stdout

        expected, measured = parse_perf_numbers(analyze_stdout, case)

        # ── Step g: write evidence ─────────────────────────────────────
        all_pass = compare_pass and analyze_pass
        verdict = "PASS" if all_pass else "FAIL"

        evidence_parts.extend(
            [
                "--- VCS compile log tail ---",
                tail(compile_log_local, 30),
                "",
                "--- Simulation log tail ---",
                tail(sim_log_local, 30),
                "",
                "--- compare_rtl.py output ---",
                compare_stdout,
                "",
                "--- analyze_perf.py output ---",
                analyze_stdout,
                "",
                f"Final verdict: {verdict}",
            ]
        )
        evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")

        # ── Step h: optional commit ────────────────────────────────────
        if all_pass and args.commit:
            if expected is None or measured is None:
                print(
                    "WARNING: --commit requested but could not parse measured/expected cycles; "
                    "skipping commit",
                    file=sys.stderr,
                )
            else:
                update_testcase_list(case, measured, expected)
                commit_status_change(case, measured, expected)
                print(f"[git] committed status change for {case}")

        append_learning(case, shape, passed=all_pass)
        print(f"[evidence] wrote {evidence_file}")
        print(f"[verdict] {verdict}")
        return 0 if all_pass else 1

    except subprocess.CalledProcessError as exc:
        evidence_parts.extend(
            [
                "--- Failure context ---",
                f"Subprocess failed (exit {exc.returncode}): {' '.join(exc.cmd)}",
                exc.stdout.strip() if exc.stdout else "",
                exc.stderr.strip() if exc.stderr else "",
                "",
                "Final verdict: FAIL (subprocess error)",
            ]
        )
        evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
        append_learning(case, shape, passed=False)
        print(f"[evidence] wrote {evidence_file}")
        return 1

    except Exception as exc:
        evidence_parts.extend(
            [
                "--- Failure context ---",
                f"Exception: {exc}",
                "",
                "Final verdict: FAIL (runner error)",
            ]
        )
        evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
        append_learning(case, shape, passed=False)
        print(f"[evidence] wrote {evidence_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
