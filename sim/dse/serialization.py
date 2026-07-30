"""Replay bundle serialization for reproducible scenario-driven DSE.

A replay bundle contains:

* ``inputs.json`` — normalized input spec (scenario, axes, seed, run config).
* ``result.json`` — the schema-v2 ``DesignSpaceResultV2``.
* ``coverage.json`` — the ``CoverageManifest``.
* ``manifest.json`` — bundle-level manifest linking the above.
* ``SHA256SUMS`` — checksums over the canonical payload files.
* ``metadata.json`` — non-deterministic metadata (timestamp, command, host).

The canonical payload (inputs + result + coverage + manifest) must be
byte-identical across runs when the same commit/input/seed is used.  Timestamps
and host information live only in ``metadata.json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contracts.errors import ConfigError
from contracts.identity import canonical_json_bytes
from contracts.result import DesignSpaceResultV2
from dse.manifest import CoverageManifest
from dse.models import DesignPoint
from dse.space import GenerationResult

_CANONICAL_PAYLOAD_FILES = ("inputs.json", "result.json", "coverage.json", "manifest.json")


@dataclass(frozen=True)
class ReplayBundlePaths:
    """Paths inside a replay bundle directory."""

    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs.json"

    @property
    def result(self) -> Path:
        return self.root / "result.json"

    @property
    def coverage(self) -> Path:
        return self.root / "coverage.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def sha256sums(self) -> Path:
        return self.root / "SHA256SUMS"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"


def _canonical_dump(obj: dict[str, Any]) -> bytes:
    """Dump a dict to canonical JSON bytes (sorted keys, compact)."""
    return canonical_json_bytes(obj)


def _write_if_not_exists(path: Path, data: bytes) -> None:
    """Write *data* to *path* only if *path* does not already exist."""
    if path.exists():
        raise ConfigError(
            f"refusing to overwrite existing replay bundle file: {path}",
            field_path=str(path),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _build_inputs_dict(
    *,
    scenario_dict: dict[str, Any],
    axes_dict: dict[str, Any],
    seed: int,
    run_config: dict[str, Any],
    design_points: list[DesignPoint],
) -> dict[str, Any]:
    """Build normalized inputs dictionary."""
    return {
        "schema_version": "1",
        "scenario": scenario_dict,
        "axes": axes_dict,
        "seed": seed,
        "run_config": run_config,
        "design_points": [
            {
                "design_point_id": p.design_point_id,
                "scenario_ref": p.scenario_ref,
                "workload_ref": p.workload_ref,
                "axis_values": p.axis_values,
                "hardware_config": p.hardware_config,
            }
            for p in design_points
        ],
    }


def _build_manifest_dict(
    *,
    input_digest: str,
    result_digest: str,
    coverage_digest: str,
    payload_digest: str,
    design_point_count: int,
) -> dict[str, Any]:
    """Build bundle manifest linking payload files."""
    return {
        "schema_version": "1",
        "payload_files": list(_CANONICAL_PAYLOAD_FILES),
        "digests": {
            "inputs": input_digest,
            "result": result_digest,
            "coverage": coverage_digest,
            "canonical_payload": payload_digest,
        },
        "design_point_count": design_point_count,
    }


def write_replay_bundle(
    output_path: Path | str,
    *,
    result_set: DesignSpaceResultV2,
    manifest: CoverageManifest,
    scenario_dict: dict[str, Any],
    axes_dict: dict[str, Any],
    seed: int,
    run_config: dict[str, Any],
    generation_result: GenerationResult,
    command: str = "",
    git_commit: str = "",
    allow_overwrite: bool = False,
) -> ReplayBundlePaths:
    """Write a replay bundle to *output_path*.

    Args:
        output_path: Directory path for the bundle.  Must not exist unless
            ``allow_overwrite`` is True.
        result_set: The v2 DSE result.
        manifest: Coverage manifest.
        scenario_dict: Normalized scenario dict (from ``Scenario.model_dump``).
        axes_dict: Axis configuration dict.
        seed: Deterministic seed.
        run_config: Run-control parameters.
        generation_result: Generated design points and exclusions.
        command: CLI command used to produce the bundle (metadata only).
        git_commit: Git commit hash (metadata only).
        allow_overwrite: If False, raise if any payload file already exists.

    Returns:
        ``ReplayBundlePaths`` with the written file paths.
    """
    root = Path(output_path)
    paths = ReplayBundlePaths(root=root)

    if root.exists() and not allow_overwrite and any((root / f).exists() for f in _CANONICAL_PAYLOAD_FILES):
        raise ConfigError(
            f"refusing to overwrite existing release/replay bundle: {root}",
            field_path=str(root),
        )

    design_points = list(generation_result.points)

    # Canonical payload files.
    inputs_dict = _build_inputs_dict(
        scenario_dict=scenario_dict,
        axes_dict=axes_dict,
        seed=seed,
        run_config=run_config,
        design_points=design_points,
    )
    inputs_bytes = _canonical_dump(inputs_dict)

    result_dict = result_set.model_dump(mode="json")
    result_bytes = _canonical_dump(result_dict)

    coverage_dict = manifest.to_dict()
    coverage_bytes = _canonical_dump(coverage_dict)

    input_digest = hashlib.sha256(inputs_bytes).hexdigest()
    result_digest = hashlib.sha256(result_bytes).hexdigest()
    coverage_digest = hashlib.sha256(coverage_bytes).hexdigest()

    payload_digest = hashlib.sha256(inputs_bytes + result_bytes + coverage_bytes).hexdigest()

    manifest_dict = _build_manifest_dict(
        input_digest=input_digest,
        result_digest=result_digest,
        coverage_digest=coverage_digest,
        payload_digest=payload_digest,
        design_point_count=len(design_points),
    )
    manifest_bytes = _canonical_dump(manifest_dict)

    # Write payload files.
    writers = [
        (paths.inputs, inputs_bytes),
        (paths.result, result_bytes),
        (paths.coverage, coverage_bytes),
        (paths.manifest, manifest_bytes),
    ]
    for path, data in writers:
        if not allow_overwrite:
            _write_if_not_exists(path, data)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    # SHA256SUMS over canonical payload.
    sha_lines: list[str] = []
    for name in _CANONICAL_PAYLOAD_FILES:
        digest = _sha256_file(root / name)
        sha_lines.append(f"{digest}  {name}")
    sha_content = "\n".join(sha_lines) + "\n"
    if not allow_overwrite:
        _write_if_not_exists(paths.sha256sums, sha_content.encode("utf-8"))
    else:
        paths.sha256sums.write_text(sha_content, encoding="utf-8")

    # Non-deterministic metadata.
    metadata = {
        "schema_version": "1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "git_commit": git_commit,
        "canonical_payload_digest": payload_digest,
    }
    if not allow_overwrite:
        _write_if_not_exists(paths.metadata, _canonical_dump(metadata))
    else:
        paths.metadata.write_bytes(_canonical_dump(metadata))

    return paths


def read_replay_bundle(output_path: Path | str) -> dict[str, Any]:
    """Read a replay bundle directory and return its canonical payload dicts."""
    root = Path(output_path)
    paths = ReplayBundlePaths(root=root)
    if not root.is_dir():
        raise ConfigError(f"replay bundle is not a directory: {root}", field_path=str(root))

    missing = [f for f in _CANONICAL_PAYLOAD_FILES + ("SHA256SUMS",) if not (root / f).exists()]
    if missing:
        raise ConfigError(
            f"replay bundle missing files: {missing}",
            field_path=str(root),
        )

    inputs = json.loads(paths.inputs.read_text(encoding="utf-8"))
    result = json.loads(paths.result.read_text(encoding="utf-8"))
    coverage = json.loads(paths.coverage.read_text(encoding="utf-8"))
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

    # Verify SHA256SUMS.
    expected_sums = {}
    for line in paths.sha256sums.read_text(encoding="utf-8").strip().splitlines():
        digest, name = line.split("  ", 1)
        expected_sums[name] = digest

    for name in _CANONICAL_PAYLOAD_FILES:
        actual = _sha256_file(root / name)
        expected = expected_sums.get(name)
        if expected is None or actual != expected:
            raise ConfigError(
                f"checksum mismatch for {name}: expected {expected}, got {actual}",
                field_path=str(root / name),
            )

    # Verify manifest payload digest.
    payload_bytes = paths.inputs.read_bytes() + paths.result.read_bytes() + paths.coverage.read_bytes()
    actual_payload_digest = hashlib.sha256(payload_bytes).hexdigest()
    expected_payload_digest = manifest.get("digests", {}).get("canonical_payload")
    if expected_payload_digest != actual_payload_digest:
        raise ConfigError(
            f"canonical payload digest mismatch: expected {expected_payload_digest}, got {actual_payload_digest}",
            field_path=str(paths.manifest),
        )

    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    return {
        "inputs": inputs,
        "result": result,
        "coverage": coverage,
        "manifest": manifest,
        "metadata": metadata,
    }


def replay_bundle_canonical_digest(output_path: Path | str) -> str:
    """Return the canonical payload digest for a bundle directory."""
    bundle = read_replay_bundle(output_path)
    return bundle["manifest"]["digests"]["canonical_payload"]


__all__ = [
    "ReplayBundlePaths",
    "write_replay_bundle",
    "read_replay_bundle",
    "replay_bundle_canonical_digest",
]
