"""Calibration registry — load and lookup decision-driving parameters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from calibration.schema import CalibrationEntry, CalibrationError

DEFAULT_PARAMETERS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "references" / "calibration" / "parameters.yaml"
)


class CalibrationRegistry:
    """In-memory registry of calibration entries keyed by calibration_id."""

    def __init__(self, entries: Iterable[CalibrationEntry]) -> None:
        self._entries: dict[str, CalibrationEntry] = {}
        for entry in entries:
            if entry.calibration_id in self._entries:
                raise CalibrationError(
                    f"duplicate calibration_id: {entry.calibration_id!r}",
                    calibration_id=entry.calibration_id,
                    reason="duplicate_id",
                )
            self._entries[entry.calibration_id] = entry

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> CalibrationRegistry:
        """Load registry from the canonical parameters.yaml file."""
        if path is None:
            path = DEFAULT_PARAMETERS_PATH
        path = Path(path)
        if not path.exists():
            raise CalibrationError(
                f"calibration parameters file not found: {path}",
                reason="missing_parameters_file",
                details={"path": str(path)},
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise CalibrationError(
                "calibration parameters file must contain a mapping",
                reason="malformed_parameters",
                details={"path": str(path)},
            )
        raw_entries = data.get("parameters", [])
        if not isinstance(raw_entries, list):
            raise CalibrationError(
                "calibration 'parameters' key must be a list",
                reason="malformed_parameters",
                details={"path": str(path)},
            )
        entries = [CalibrationEntry.model_validate(item) for item in raw_entries]
        return cls(entries)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationRegistry:
        """Build a registry from a dict mapping IDs to entry dicts."""
        entries = []
        for calibration_id, item in data.items():
            if not isinstance(item, dict):
                raise CalibrationError(
                    f"entry for {calibration_id!r} must be a dict",
                    reason="malformed_entry",
                    details={"calibration_id": calibration_id},
                )
            item = dict(item)
            item.setdefault("calibration_id", calibration_id)
            entries.append(CalibrationEntry.model_validate(item))
        return cls(entries)

    def get(self, calibration_id: str) -> CalibrationEntry:
        """Return the entry for *calibration_id*.

        Raises :class:`CalibrationError` if the ID is unknown.
        """
        try:
            return self._entries[calibration_id]
        except KeyError as exc:
            raise CalibrationError(
                f"unknown calibration_id: {calibration_id!r}",
                calibration_id=calibration_id,
                reason="unknown_id",
            ) from exc

    def lookup(self, calibration_id: str) -> CalibrationEntry | None:
        """Return the entry or ``None`` if unknown."""
        return self._entries.get(calibration_id)

    def __contains__(self, calibration_id: str) -> bool:
        return calibration_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def ids(self) -> list[str]:
        """Return all registered calibration IDs in sorted order."""
        return sorted(self._entries.keys())

    def entries(self) -> list[CalibrationEntry]:
        """Return all entries sorted by calibration_id."""
        return [self._entries[k] for k in self.ids()]

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical dict suitable for hashing."""
        return {
            "schema_version": "1",
            "parameters": {cid: entry.model_dump(mode="json") for cid, entry in sorted(self._entries.items())},
        }


__all__ = ["DEFAULT_PARAMETERS_PATH", "CalibrationRegistry"]
