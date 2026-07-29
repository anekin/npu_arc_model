"""Validation package for full acceptance matrix and release gates."""

from validation.scenario_matrix import (
    AcceptanceMatrix,
    MatrixCategory,
    MatrixEntry,
    build_matrix,
)

__all__ = [
    "AcceptanceMatrix",
    "MatrixCategory",
    "MatrixEntry",
    "build_matrix",
]
