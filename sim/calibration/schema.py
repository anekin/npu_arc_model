"""Calibration schema and typed errors.

Defines :class:`CalibrationEntry`, the provenance/status enums, and the
 typed :class:`CalibrationError` used for fail-closed calibration handling.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from contracts.hardware import TrustLevel


class CalibrationStatus(str, Enum):
    """Lifecycle status of a calibration entry."""

    assumption = "assumption"
    calibrated = "calibrated"
    exploratory = "exploratory"
    superseded = "superseded"


class CalibrationEntry(BaseModel):
    """One decision-driving calibration parameter.

    Every entry carries a stable ``calibration_id``, the current ``value``,
    provenance metadata, a trust level, and an explicit calibration range.
    ``range_min`` and ``range_max`` are optional numeric bounds parsed from
    ``calibration_range``; when present they enable programmatic
    extrapolation checks.
    """

    model_config = ConfigDict(extra="forbid")

    calibration_id: str = Field(..., description="Stable registry key")
    value: float = Field(..., description="Current best value")
    unit: str = Field(..., description="Physical unit")
    source_uri: Optional[str] = Field(default=None, description="URI/path to source")
    source_hash: Optional[str] = Field(default=None, description="SHA-256 of raw source data when applicable")
    trust_level: TrustLevel = Field(..., description="T0-T3 as defined in contracts.hardware")
    calibration_range: str = Field(..., description="Human-readable calibrated domain")
    range_min: Optional[float] = Field(default=None, description="Programmatic lower bound")
    range_max: Optional[float] = Field(default=None, description="Programmatic upper bound")
    status: CalibrationStatus = Field(..., description="assumption | calibrated | exploratory | superseded")
    description: str = Field(default="")

    @field_validator("trust_level", mode="before")
    @classmethod
    def _accept_string_trust(cls, v: Any) -> Any:
        if isinstance(v, str):
            return TrustLevel(v)
        return v

    @field_validator("status", mode="before")
    @classmethod
    def _accept_string_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            return CalibrationStatus(v)
        return v

    def is_in_range(self, actual_value: float) -> bool:
        """Return True if *actual_value* lies inside [range_min, range_max]."""
        if self.range_min is not None and actual_value < self.range_min:
            return False
        if self.range_max is not None and actual_value > self.range_max:
            return False
        return True


class CalibrationError(ValueError):
    """Raised when calibration evidence is missing, duplicated, or corrupted.

    Carries structured context so callers can classify failures without
    parsing error text.
    """

    def __init__(
        self,
        message: str,
        *,
        calibration_id: str = "",
        reason: str = "",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.calibration_id = calibration_id
        self.reason = reason
        self.details = details or {}

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.args[0]!r}, "
            f"calibration_id={self.calibration_id!r}, reason={self.reason!r})"
        )


__all__ = [
    "CalibrationEntry",
    "CalibrationError",
    "CalibrationStatus",
]
