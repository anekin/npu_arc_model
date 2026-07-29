#!/usr/bin/env python3
"""
MXU Model Calibration: RTL measured cycles vs MXUModel.estimate().

Reads evidence from references/calibration/raw/ as train/held-out CSV fixtures,
validates SHA256SUMS, checks for duplicate case IDs, fits a per-shape correction
factor on the training set, and reports deterministic metrics on the held-out
set.

Fail-closed behavior:
    - Missing raw fixture files or SHA256SUMS -> CalibrationError, exit 2
    - Duplicate case_id across train/held-out -> CalibrationError, exit 2
    - Checksum mismatch -> CalibrationError, exit 2
    - No analytic fallback when raw RTL is absent

Usage:
    PYTHONPATH=sim python3 scripts/calibrate_mxu_model.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = REPO_ROOT / "sim"
RAW_DIR = REPO_ROOT / "references" / "calibration" / "raw"
OUTPUT_FILE = REPO_ROOT / ".omo" / "evidence" / "mxu-calibration.json"

# Add sim to path for MXUModel and calibration modules.
sys.path.insert(0, str(SIM_DIR))

from calibration.schema import CalibrationError
from models.mxu import MXUModel


MODEL_CONFIG = {
    "mxu": {
        "array_height": 64,
        "array_width": 64,
        "frequency_mhz": 1000,
        "weight_precision_bits": 4,
        "activation_precision_bits": 8,
        "ops_per_mac": 2,
        "double_buffer": True,
    },
    "memory": {
        "bandwidth_bytes_per_cycle": 51.2,
        "dram_efficiency": 0.85,
    },
    "optimizations": {
        "dma_bw_multiplier": 1.0,
    },
}


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksums(raw_dir: Path) -> None:
    """Validate SHA256SUMS against actual files; fail closed on mismatch."""
    sums_path = raw_dir / "SHA256SUMS"
    if not sums_path.exists():
        raise CalibrationError(
            f"checksum manifest not found: {sums_path}",
            reason="missing_checksum_manifest",
            details={"path": str(sums_path)},
        )

    expected: Dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise CalibrationError(
                f"invalid SHA256SUMS line: {line!r}",
                reason="malformed_checksum_manifest",
            )
        expected[Path(parts[1]).name] = parts[0]

    for name, expected_hash in expected.items():
        fpath = raw_dir / name
        if not fpath.exists():
            raise CalibrationError(
                f"raw file listed in SHA256SUMS is missing: {name}",
                reason="missing_raw_file",
                details={"file": name},
            )
        actual = _sha256_file(fpath)
        if actual != expected_hash:
            raise CalibrationError(
                f"checksum mismatch for {name}: expected {expected_hash}, got {actual}",
                reason="checksum_mismatch",
                details={"file": name, "expected": expected_hash, "actual": actual},
            )


def _load_csv(path: Path) -> List[dict[str, Any]]:
    """Load a calibration CSV file."""
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "case_id": row["case_id"].strip(),
                "M": int(row["M"]),
                "N": int(row["N"]),
                "K": int(row["K"]),
                "measured_cycles": int(row["measured_cycles"]),
            })
    return rows


def _load_raw_fixtures(raw_dir: Path) -> Tuple[List[dict[str, Any]], List[dict[str, Any]]]:
    """Return (train_rows, heldout_rows) after checksum and duplicate checks."""
    _verify_checksums(raw_dir)

    train_path = raw_dir / "mxu_train.csv"
    heldout_path = raw_dir / "mxu_heldout.csv"

    if not train_path.exists():
        raise CalibrationError(
            f"training fixture not found: {train_path}",
            reason="missing_train_fixture",
            details={"path": str(train_path)},
        )
    if not heldout_path.exists():
        raise CalibrationError(
            f"held-out fixture not found: {heldout_path}",
            reason="missing_heldout_fixture",
            details={"path": str(heldout_path)},
        )

    train_rows = _load_csv(train_path)
    heldout_rows = _load_csv(heldout_path)

    seen: Dict[str, str] = {}
    for row in train_rows:
        cid = row["case_id"]
        if cid in seen:
            raise CalibrationError(
                f"duplicate case_id in training fixture: {cid!r}",
                reason="duplicate_case_id",
                details={"case_id": cid, "partition": "train"},
            )
        seen[cid] = "train"
    for row in heldout_rows:
        cid = row["case_id"]
        if cid in seen:
            raise CalibrationError(
                f"duplicate case_id across train/held-out: {cid!r} "
                f"(already in {seen[cid]})",
                reason="duplicate_case_id",
                details={"case_id": cid, "partition": "heldout", "existing_partition": seen[cid]},
            )
        seen[cid] = "heldout"

    return train_rows, heldout_rows


def _model_cycles(model: MXUModel, M: int, K: int, N: int) -> int:
    """Return MXUModel total cycles for (M,K,N)."""
    return model.estimate(M, K, N).total_cycles


def _fit_correction(train_rows: List[dict[str, Any]]) -> float:
    """Fit a single multiplicative correction factor on training data.

    Held-out rows must not participate in fitting.
    """
    model = MXUModel(MODEL_CONFIG)
    ratios: List[float] = []
    for row in train_rows:
        measured = int(row["measured_cycles"])
        predicted = _model_cycles(model, row["M"], row["K"], row["N"])
        if predicted > 0 and measured > 0:
            ratios.append(measured / predicted)
    if not ratios:
        raise CalibrationError(
            "no valid training rows to fit correction",
            reason="empty_training_set",
        )
    # Deterministic mean.
    return sum(ratios) / len(ratios)


def _evaluate(
    rows: List[dict[str, Any]],
    model: MXUModel,
    correction: float,
) -> dict[str, Any]:
    """Compute deterministic calibration metrics for a set of cases."""
    abs_errors: List[float] = []
    rel_errors: List[float] = []
    entries: List[dict[str, Any]] = []

    for row in rows:
        M, K, N = row["M"], row["K"], row["N"]
        measured = int(row["measured_cycles"])
        predicted = int(round(_model_cycles(model, M, K, N) * correction))
        abs_err = abs(measured - predicted)
        rel_err = abs_err / max(measured, 1)
        abs_errors.append(abs_err)
        rel_errors.append(rel_err)
        entries.append({
            "case_id": row["case_id"],
            "M": M,
            "N": N,
            "K": K,
            "measured_cycles": measured,
            "predicted_cycles": predicted,
            "absolute_error": abs_err,
            "relative_error": round(rel_err, 6),
        })

    return {
        "count": len(rows),
        "correction_factor": round(correction, 6),
        "mean_absolute_error": round(sum(abs_errors) / len(abs_errors), 3) if abs_errors else 0.0,
        "max_absolute_error": max(abs_errors) if abs_errors else 0,
        "mean_relative_error": round(sum(rel_errors) / len(rel_errors), 6) if rel_errors else 0.0,
        "max_relative_error": round(max(rel_errors), 6) if rel_errors else 0.0,
        "cases": entries,
    }


def main() -> int:
    train_rows, heldout_rows = _load_raw_fixtures(RAW_DIR)

    if not train_rows:
        raise CalibrationError(
            "training fixture is empty",
            reason="empty_training_set",
        )

    correction = _fit_correction(train_rows)
    model = MXUModel(MODEL_CONFIG)

    train_metrics = _evaluate(train_rows, model, correction)
    heldout_metrics = _evaluate(heldout_rows, model, correction)

    output = {
        "model_config": MODEL_CONFIG,
        "raw_dir": str(RAW_DIR),
        "train_case_ids": [r["case_id"] for r in train_rows],
        "heldout_case_ids": [r["case_id"] for r in heldout_rows],
        "train": train_metrics,
        "heldout": heldout_metrics,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    print(f"MXU calibration complete: {len(train_rows)} train, {len(heldout_rows)} held-out")
    print(f"  correction_factor={correction:.4f}")
    print(f"  held-out MAE={heldout_metrics['mean_absolute_error']}")
    print(f"  output written to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CalibrationError as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        sys.exit(2)
