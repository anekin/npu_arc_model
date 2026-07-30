#!/usr/bin/env python3
"""Verify that every Todo/F1-F4 in the plan has matching evidence.

Parses the plan markdown, finds all todo/checkbox items, maps them to evidence
files under ``--evidence-root``, and reports gaps.  Produces a structured JSON
report and exits non-zero when required evidence is missing or a recorded
command exit code is non-zero.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

EXPECTED_RED_PATTERNS = (
    "task-3-physical-red*",
    "task-5-bandwidth-detail*",
    "task-5-verify*",
)


def _is_expected_red(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pat) for pat in EXPECTED_RED_PATTERNS)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def parse_plan_todos(plan_text: str) -> list[dict[str, Any]]:
    """Return list of todos from markdown checkbox lines.

    Matches lines like:
      - [x] 18. Todo title
      - [ ] F1. Final verification name
    """
    pattern = re.compile(r"^\s*-\s*\[([xX\s])\]\s*([0-9]+|F[1-4])\.\s*(.*)$", re.MULTILINE)
    todos = []
    for match in pattern.finditer(plan_text):
        checked = match.group(1).strip().lower() == "x"
        todo_id = match.group(2)
        title = match.group(3).strip()
        todos.append(
            {
                "id": todo_id,
                "checked": checked,
                "title": title,
            }
        )
    return todos


def evidence_paths_for_todo(todo_id: str, evidence_root: Path) -> list[Path]:
    """Find evidence files matching a todo ID.

    Todo 18 matches ``task-18-*``; F1 matches ``final-f1-*`` (case insensitive).
    """
    if todo_id.upper().startswith("F"):
        prefix = todo_id.lower()
        pattern = f"final-{prefix}-*"
    else:
        pattern = f"task-{todo_id}-*"
    return sorted(evidence_root.glob(pattern))


def _extract_test_counts(text: str) -> dict[str, Any]:
    """Best-effort parse of pytest collected/passed/failed/skipped counts."""
    result: dict[str, Any] = {}
    # pytest -q final line: "X passed, Y failed, Z skipped in ..."
    m = re.search(r"(\d+)\s+passed", text)
    if m:
        result["passed"] = int(m.group(1))
    # Negative lookahead rejects key=value patterns like "complete=66 failed=0"
    m = re.search(r"(\d+)\s+failed(?!\s*=\s*\d)", text)
    if m:
        result["failed"] = int(m.group(1))
    m = re.search(r"(\d+)\s+skipped", text)
    if m:
        result["skipped"] = int(m.group(1))
    m = re.search(r"(\d+)\s+collected", text)
    if m:
        result["collected"] = int(m.group(1))
    # Generic exit code marker
    m = re.search(r"exit_code[:=]\s*(-?\d+)", text)
    if m:
        result["exit_code"] = int(m.group(1))
    return result


def inspect_evidence(path: Path) -> dict[str, Any]:
    """Return structured metadata for an evidence file."""
    digest = _sha256_file(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    info: dict[str, Any] = {
        "path": str(path),
        "digest": digest,
        "size_bytes": path.stat().st_size,
    }
    try:
        data = json.loads(text)
        info["is_json"] = True
        info["exit_code"] = data.get("exit_code", 0) if isinstance(data, dict) else 0
        info["test_counts"] = data.get("test_counts") if isinstance(data, dict) else None
    except json.JSONDecodeError:
        info["is_json"] = False
        counts = _extract_test_counts(text)
        info["test_counts"] = counts
        # If no explicit exit code, infer from failure text.
        info["exit_code"] = counts.get("exit_code", 1 if counts.get("failed", 0) > 0 else 0)
    return info


def verify_ledger(
    plan_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    """Build the ledger verification report."""
    plan_text = plan_path.read_text(encoding="utf-8")
    todos = parse_plan_todos(plan_text)

    entries: list[dict[str, Any]] = []
    gaps: list[str] = []
    non_zero_exits: list[str] = []
    expected_red_evidence: list[str] = []

    for todo in todos:
        todo_id = todo["id"]
        paths = evidence_paths_for_todo(todo_id, evidence_root)
        inspected = [inspect_evidence(p) for p in paths]
        has_evidence = bool(paths)

        for ev in inspected:
            ev_path = Path(ev["path"])
            ev["expected_red"] = _is_expected_red(ev_path)

        entry = {
            "id": todo_id,
            "title": todo["title"],
            "checked": todo["checked"],
            "evidence_files": inspected,
            "evidence_count": len(paths),
        }
        entries.append(entry)

        if not has_evidence:
            gaps.append(f"{todo_id}: {todo['title']}")
        else:
            for ev in inspected:
                if ev.get("exit_code", 0) != 0:
                    marker = f"{todo_id}: {ev['path']} exit={ev['exit_code']}"
                    if ev.get("expected_red"):
                        expected_red_evidence.append(marker)
                    else:
                        non_zero_exits.append(marker)

    verdict = "PASS" if not gaps and not non_zero_exits else "FAIL"

    return {
        "verdict": verdict,
        "plan": str(plan_path),
        "evidence_root": str(evidence_root),
        "todos_checked": len(todos),
        "missing_evidence": gaps,
        "non_zero_exit_evidence": non_zero_exits,
        "expected_red_evidence": expected_red_evidence,
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify evidence ledger against plan")
    parser.add_argument("--plan", type=Path, default=Path(".omo/plans/arc-model-scenario-driven-dse-development.md"))
    parser.add_argument("--evidence-root", type=Path, default=Path(".omo/evidence"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = verify_ledger(args.plan, args.evidence_root)

    output_text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
