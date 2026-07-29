"""Typed error hierarchy for the Arc Model contracts.

All errors carry a message, and most carry additional structured context
(e.g. field path, offending value, expected constraint) so that downstream
tools can classify failures without regex-parsing error strings.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "SchemaVersionError",
    "DimensionBindingError",
    "UnsupportedOperatorError",
    "CoverageError",
    "NonAuthoritativeRunError",
]


class ConfigError(ValueError):
    """Raised when a configuration file is structurally or semantically invalid.

    Covers: non-mapping YAML root, missing required sections, invalid version
    key, bool-as-int, non-finite/non-positive numeric values, and unknown
    fields when extra='forbid' is active.
    """

    def __init__(self, message: str, field_path: str = "", value: object = None) -> None:
        super().__init__(message)
        self.field_path = field_path
        self.value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.args[0]!r}, field_path={self.field_path!r})"


class SchemaVersionError(ConfigError):
    """Raised when an input carries an unsupported or missing schema version."""


class DimensionBindingError(ValueError):
    """Raised when a symbolic dimension is unbound or bound to an invalid value."""

    def __init__(self, message: str, dimension: str = "") -> None:
        super().__init__(message)
        self.dimension = dimension


class UnsupportedOperatorError(ValueError):
    """Raised when a workload graph contains an operator that is not in the
    registry (neither modeled, free/fused, nor explicitly unsupported).
    """

    def __init__(self, message: str, op_type: str = "", node_id: str = "") -> None:
        super().__init__(message)
        self.op_type = op_type
        self.node_id = node_id


class CoverageError(RuntimeError):
    """Raised when a DSE or validation run has missing coverage on a required
    axis — e.g. a scenario dimension was not enumerated, or a required test
    point was skipped.
    """

    def __init__(self, message: str, missing_axes: list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_axes = missing_axes or []


class NonAuthoritativeRunError(RuntimeError):
    """Raised when a caller attempts to treat a partial/exploratory run as
    authoritative — e.g. requesting a release recommendation from a set of
    results that contains failures or uncovered critical axes.
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason
