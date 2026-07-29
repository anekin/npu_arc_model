"""Calibration provenance, registry, and trust gate for the Arc Model."""

from calibration.evaluate import TrustGate, calibration_digest, calibration_ids_for_design_point, result_digest
from calibration.registry import CalibrationRegistry
from calibration.schema import CalibrationEntry, CalibrationError, CalibrationStatus

__all__ = [
    "CalibrationEntry",
    "CalibrationError",
    "CalibrationRegistry",
    "CalibrationStatus",
    "TrustGate",
    "calibration_digest",
    "calibration_ids_for_design_point",
    "result_digest",
]
