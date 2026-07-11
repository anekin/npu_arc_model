#!/usr/bin/env python3
"""SFU Module-Level Performance Case Runner.

End-to-end runner for a single SFU performance case.  Compiles the perf
testbench on the EDA server, runs it with plusargs, downloads the log,
and runs analyze_sfu_perf.py for PASS/FAIL verdict.

Usage:
    python3 scripts/run_sfu_perf_case.py --case SFV-P01 --op softmax --dim 64
    python3 scripts/run_sfu_perf_case.py --case SFV-P01 --op softmax --dim 64 --commit
    python3 scripts/run_sfu_perf_case.py --case SFV-P01 --op softmax --dim 64 --dry-run
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

TESTCASE_LIST = CADUCEUS_CORE / "rtl" / "testcase-list-sfu-vector-perf.md"
LEARNINGS_FILE = REPO_ROOT / ".omo" / "notepads" / "soc-verification-gaps-phase5" / "learnings.md"

DEFAULT_EDA_SERVER = "zhengs@192.168.0.11"
DEFAULT_VCS_MODULE = "vcs/vcs_2023.12sp2"  # SFU/Vector require 2023.12sp2
DEFAULT_SIMV = REPO_ROOT / "build" / "simv_tb_sfu_perf"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "build" / "evidence"

VCS_SETUP = "source /NAS/Tools/methodology/modules/init/bash && module load {vcs_module}"

# Op name → numeric code mapping (from rtl/sfu/sfu_top.v OP_* constants)
OP_CODE_MAP: dict[str, int] = {
    "softmax":   0,
    "layernorm": 1,
    "gelu":      2,
    "silu":      4,
    "rope":      5,
    "rmsnorm":   6,
}


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="SFU module-level performance case runner (one command, one PASS/FAIL).",
    )
    parser.add_argument("--case", required=True, help="Case ID, e.g. SFV-P01")
    parser.add_argument("--op", required=True, help="SFU operation: softmax, layernorm, gelu, silu, rope, rmsnorm")
    parser.add_argument("--dim", type=int, required=True, help="Element dimension")
    parser.add_argument("--pos", type=int, default=0, help="Position for RoPE (default: 0)")
    parser.add_argument("--repeat", type=int, default=1, help="Back-to-back CMD loop count (default: 1)")
    parser.add_argument("--simv", default=str(DEFAULT_SIMV), help=f"VCS simv binary path (default: {DEFAULT_SIMV})")
    parser.add_argument("--rebuild", action="store_true", help="Force VCS recompile")
    parser.add_argument("--eda-server", default=DEFAULT_EDA_SERVER, help=f"EDA server SSH target (default: {DEFAULT_EDA_SERVER})")
    parser.add_argument("--vcs-module", default=DEFAULT_VCS_MODULE, help=f"VCS module name (default: {DEFAULT_VCS_MODULE})")
    parser.add_argument("--commit", action="store_true", help="Commit testcase-list status change")
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR), help=f"Evidence directory (default: {DEFAULT_EVIDENCE_DIR})")
    parser.add_argument("--dry-run", action="store_true", help="Formula check only; no VCS")
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


def file_exists_on_remote(eda_server: str, path: Path) -> bool:
    """Return True if path exists on the EDA server."""
    result = run_ssh(eda_server, f"test -e {shlex.quote(str(path))} && echo YES || echo NO", check=True)
    return result.stdout.strip() == "YES"


def tail(path: Path, n: int = 30) -> str:
    """Return the last n lines of a file."""
    if not path.exists():
        return f"<file not found: {path}>"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def has_vcs_errors(log_path: Path) -> bool:
    """Return True if the VCS log contains Error lines."""
    if not log_path.exists():
        return True
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Error") or "Error-[" in line:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Core steps
# ══════════════════════════════════════════════════════════════════════


def step_compile_vcs(
    eda_server: str, vcs_module: str, simv: Path, force: bool
) -> Path:
    """Compile the SFU performance testbench on the EDA server."""
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
        f"-top tb_sfu_perf "
        f"rtl/tb/tb_sfu_perf.v rtl/sfu/*.v "
        f"-o {shlex.quote(str(simv))} -l {shlex.quote(str(compile_log))}"
    )
    run_ssh(eda_server, cmd)

    if not file_exists_on_remote(eda_server, simv):
        raise RuntimeError(f"VCS compile did not produce simv binary {simv}")

    return compile_log


def step_run_simulation(
    eda_server: str,
    vcs_module: str,
    simv: Path,
    case_id: str,
    op: str,
    dim: int,
    pos: int,
    repeat: int,
) -> Path:
    """Run the compiled simv on the EDA server."""
    sim_log_remote = Path(f"{simv}.{case_id}.log")

    setup = VCS_SETUP.format(vcs_module=shlex.quote(vcs_module))
    op_code = OP_CODE_MAP.get(op.lower(), 0)
    plusargs = f"+case={case_id} +op_code={op_code} +dim={dim}"
    if op.lower() == "rope":
        plusargs += f" +pos={pos}"
    if repeat > 1:
        plusargs += f" +repeat={repeat}"

    cmd = (
        f"{setup} && "
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"{shlex.quote(str(simv))} {plusargs} "
        f"-l {shlex.quote(str(sim_log_remote))}"
    )
    run_ssh(eda_server, cmd)

    return sim_log_remote


def step_analyze_perf(
    case_id: str, op: str, dim: int, sim_log_local: Path
) -> subprocess.CompletedProcess[str]:
    """Run analyze_sfu_perf.py to check measured vs expected cycles."""
    return run_cmd(
        [
            "python3",
            str(CADUCEUS_CORE / "scripts" / "analyze_sfu_perf.py"),
            "--case", case_id,
            "--op", op,
            "--dim", str(dim),
            "--log", str(sim_log_local),
        ],
        cwd=REPO_ROOT,
    )


def step_dry_run_analyze(case_id: str, op: str, dim: int) -> subprocess.CompletedProcess[str]:
    """Run analyze_sfu_perf.py in dry-run mode."""
    return run_cmd(
        [
            "python3",
            str(CADUCEUS_CORE / "scripts" / "analyze_sfu_perf.py"),
            "--dry-run",
            "--case", case_id,
            "--op", op,
            "--dim", str(dim),
        ],
        cwd=REPO_ROOT,
    )


def append_learning(case_id: str, op: str, dim: int, passed: bool) -> None:
    """Append a summary line to the learnings notepad."""
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "PASS" if passed else "FAIL"
    line = (
        f"\n## {timestamp} run_sfu_perf_case.py — {case_id} op={op} dim={dim} — {status}\n"
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

    case_id: str = args.case
    op: str = args.op
    dim: int = args.dim
    pos: int = args.pos
    repeat: int = args.repeat
    simv = Path(args.simv).resolve()
    eda_server: str = args.eda_server
    vcs_module: str = args.vcs_module
    evidence_dir = Path(args.evidence_dir).resolve()
    dry_run: bool = args.dry_run

    evidence_dir.mkdir(parents=True, exist_ok=True)
    repeat_suffix = f"_r{repeat}" if repeat > 1 else ""
    evidence_file = evidence_dir / f"sfv-{case_id}{repeat_suffix}-summary.md"
    compile_log_local = evidence_dir / f"sfv-{case_id}{repeat_suffix}_compile.log"
    sim_log_local = evidence_dir / f"sfv-{case_id}{repeat_suffix}_sim.log"

    command_used = " ".join(sys.argv)

    evidence_parts: list[str] = [
        f"# SFU Perf Case: {case_id}",
        f"",
        f"- **Op**: {op}",
        f"- **Dim**: {dim}",
        f"- **Pos**: {pos}",
        f"- **Command**: `{command_used}`",
        f"",
    ]

    try:
        if dry_run:
            analyze_result = step_dry_run_analyze(case_id, op, dim)
            evidence_parts.extend([
                "## Dry-Run Formula Check",
                "",
                "```",
                analyze_result.stdout.strip(),
                "```",
                "",
                "**Final verdict: PASS (dry-run, formula check only)**",
            ])
            evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
            append_learning(case_id, op, dim, passed=True)
            print(f"[evidence] wrote {evidence_file}")
            return 0

        # ── VCS compile ──────────────────────────────────────────────────
        compile_log_remote = step_compile_vcs(eda_server, vcs_module, simv, args.rebuild)
        scp_from_remote(eda_server, compile_log_remote, compile_log_local)

        if has_vcs_errors(compile_log_local):
            evidence_parts.extend([
                "## VCS Compile — FAIL",
                "",
                "```",
                tail(compile_log_local, 30),
                "```",
                "",
                "**Final verdict: FAIL (VCS compile errors)**",
            ])
            evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
            print(f"ERROR: VCS compile produced errors (see {compile_log_local})", file=sys.stderr)
            return 1

        # ── Run simulation ───────────────────────────────────────────────
        sim_log_remote = step_run_simulation(
            eda_server, vcs_module, simv, case_id, op, dim, pos, repeat
        )
        scp_from_remote(eda_server, sim_log_remote, sim_log_local)

        if not sim_log_local.exists():
            raise RuntimeError(f"Simulation log was not copied back: {sim_log_local}")

        sim_text = sim_log_local.read_text(encoding="utf-8", errors="replace")
        if "PERF|" not in sim_text:
            evidence_parts.extend([
                "## Simulation — FAIL (no PERF data)",
                "",
                "```",
                tail(sim_log_local, 30),
                "```",
                "",
                "**Final verdict: FAIL (no PERF| lines in simulation log)**",
            ])
            evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
            print("ERROR: simulation log contains no PERF| lines", file=sys.stderr)
            return 1

        # ── Cycle analysis ───────────────────────────────────────────────
        analyze_result = step_analyze_perf(case_id, op, dim, sim_log_local)
        analyze_stdout = analyze_result.stdout.strip()
        analyze_pass = "PASS" in analyze_stdout and "FAIL" not in analyze_stdout

        evidence_parts.extend([
            "## Compile Log",
            "",
            "```",
            tail(compile_log_local, 20),
            "```",
            "",
            "## Simulation Log (tail)",
            "",
            "```",
            tail(sim_log_local, 40),
            "```",
            "",
            "## Cycle Analysis",
            "",
            "```",
            analyze_stdout,
            "```",
            "",
            f"**Final verdict: {'PASS' if analyze_pass else 'FAIL'}**",
        ])
        evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")

        append_learning(case_id, op, dim, passed=analyze_pass)
        print(f"[evidence] wrote {evidence_file}")
        print(f"[verdict] {'PASS' if analyze_pass else 'FAIL'}")
        return 0 if analyze_pass else 1

    except subprocess.CalledProcessError as exc:
        evidence_parts.extend([
            "## Failure Context",
            "",
            f"Subprocess failed (exit {exc.returncode}): {' '.join(exc.cmd)}",
            exc.stdout.strip() if exc.stdout else "",
            exc.stderr.strip() if exc.stderr else "",
            "",
            "**Final verdict: FAIL (subprocess error)**",
        ])
        evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
        append_learning(case_id, op, dim, passed=False)
        print(f"[evidence] wrote {evidence_file}")
        return 1

    except Exception as exc:
        evidence_parts.extend([
            "## Failure Context",
            "",
            f"Exception: {exc}",
            "",
            "**Final verdict: FAIL (runner error)**",
        ])
        evidence_file.write_text("\n".join(evidence_parts) + "\n", encoding="utf-8")
        append_learning(case_id, op, dim, passed=False)
        print(f"[evidence] wrote {evidence_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
