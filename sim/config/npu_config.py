"""NPU configuration loader.

Reads the canonical YAML spec into a plain dict so other modules can deep-copy
and override it without mutating the global configuration.

All errors are typed ``ConfigError`` instances with a field path,
never bare ``ValueError`` or ``TypeError``.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from sim.contracts.errors import ConfigError


def _validate_root_is_mapping(data: object) -> None:
    """Reject non-mapping YAML roots."""
    if not isinstance(data, dict):
        raise ConfigError(
            f"YAML root must be a mapping, got {type(data).__name__}",
            field_path="<root>",
            value=data,
        )


def _validate_numeric_fields(
    data: dict,
    path: str = "",
    required_positive: tuple[str, ...] = (),
    required_non_negative: tuple[str, ...] = (),
) -> None:
    """Recursively validate numeric fields.

    Rejects:
      - bool values for numeric fields
      - non-finite floats (NaN, Inf)
      - non-positive values where positive is required
      - negative values where non-negative is required
    """
    # Check leaf numeric values
    for key, value in data.items():
        field = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            _validate_numeric_fields(
                value,
                path=field,
                required_positive=required_positive,
                required_non_negative=required_non_negative,
            )
            continue

        if isinstance(value, bool):
            # bool is subclass of int, reject it for numeric fields
            if key in required_positive or key in required_non_negative:
                raise ConfigError(
                    f"bool value for numeric field '{field}': got {value!r}",
                    field_path=field,
                    value=value,
                )
            continue

        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ConfigError(
                    f"non-finite value for '{field}': {value}",
                    field_path=field,
                    value=value,
                )
            if key in required_positive and value <= 0:
                raise ConfigError(
                    f"'{field}' must be positive, got {value}",
                    field_path=field,
                    value=value,
                )
            if key in required_non_negative and value < 0:
                raise ConfigError(
                    f"'{field}' must be non-negative, got {value}",
                    field_path=field,
                    value=value,
                )


def load_config() -> dict:
    """Load ``sim/config/npu_config.yaml`` and return it as a plain dict.

    Raises ``ConfigError`` for invalid YAML shape or values.
    """
    config_path = Path(__file__).with_suffix(".yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(
            f"Failed to parse YAML: {exc}",
            field_path=str(config_path),
        ) from exc

    _validate_root_is_mapping(data)

    # Version check
    version = data.get("version")
    if version is not None and version != "2" and version != 2:
        raise ConfigError(
            f"Unsupported schema version: {version!r} (expected '2')",
            field_path="version",
            value=version,
        )

    # Numeric validation: frequency, bandwidth, dimensions, etc.
    numeric_positive = (
        "frequency_mhz",
        "bandwidth_gbps",
        "array_height",
        "array_width",
        "weight_precision_bits",
        "activation_precision_bits",
        "accumulate_precision_bits",
        "ops_per_mac",
    )
    numeric_non_negative = (
        "refresh_overhead_percent",
        "dram_efficiency",
    )

    _validate_numeric_fields(
        data,
        required_positive=numeric_positive,
        required_non_negative=numeric_non_negative,
    )

    return data


def load_config_v2() -> dict:
    """Load and validate as v2 hardware schema.

    Returns ``HardwareConfigV2.model_dump()``.
    Raises ``ConfigError`` on any validation failure.
    """
    from sim.contracts.hardware import HardwareConfigV2
    from sim.contracts.migrations import migrate_v1_to_v2

    data = load_config()

    try:
        migrated, loss = migrate_v1_to_v2(data)
        validated = HardwareConfigV2.model_validate(migrated)
        return validated.model_dump()
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"v2 schema validation failed: {exc}",
            field_path="<root>",
        ) from exc
