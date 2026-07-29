"""Arc Model — canonical contracts.

Public API surface for the contracts package:
  units      — canonical unit conversions
  errors     — typed error hierarchy
  hardware   — versioned hardware schema (Pydantic v2)
  migrations — v1 ↔ v2 migration and legacy projection
"""

from contracts.errors import (  # noqa: F401
    ConfigError,
    CoverageError,
    DimensionBindingError,
    NonAuthoritativeRunError,
    SchemaVersionError,
    UnsupportedOperatorError,
)
from contracts.hardware import (  # noqa: F401
    DEFAULT_DRAM_EFFICIENCY_PROVENANCE,
    DEFAULT_GMMA_PIPELINE_PROVENANCE,
    DEFAULT_NODE_SCALE_PROVENANCE,
    DEFAULT_PE_AREA_RATIO_PROVENANCE,
    DEFAULT_TSV_PROVENANCE,
    HardwareConfigV2,
    MACEngineConfig,
    MemoryConfig,
    Provenance,
    SRAMConfig,
    TrustLevel,
)
from contracts.migrations import (  # noqa: F401
    migrate_v1_to_v2,
    project_v2_to_legacy,
)
from contracts.units import (  # noqa: F401
    bandwidth_gbps_to_bytes_per_cycle,
    bytes_per_cycle_to_bandwidth_gbps,
    bytes_to_gib,
    cycles_to_microseconds,
    cycles_to_seconds,
    gib_to_bytes,
    microseconds_to_cycles,
    seconds_to_cycles,
)
