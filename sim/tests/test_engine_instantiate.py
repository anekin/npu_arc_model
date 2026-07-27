"""Test all engine types instantiate via create_engine() without error.

Every engine subclass must be constructable through the factory and
report the correct engine_type identifier.
"""

from engine.mac_engine import create_engine

# Minimal config: all engines share the same mac_engine params
# Only the "type" field varies.
_BASE_CONFIG = {
    "mac_engine": {
        "array_height": 64,
        "array_width": 64,
        "frequency_mhz": 1000,
        "weight_precision_bits": 4,
        "activation_precision_bits": 8,
    },
    "memory": {
        "bandwidth_bytes_per_cycle": 51.2,
        "dram_efficiency": 0.85,
    },
}

ENGINE_TYPES = [
    "systolic",
    "block",
    "os_systolic",
    "input_stationary",
    "tensor_core",
    "wmma",
    "gmma",
]


def test_all_engines_instantiate():
    """create_engine must succeed for every engine type."""
    for etype in ENGINE_TYPES:
        cfg = dict(_BASE_CONFIG)
        cfg["mac_engine"]["type"] = etype
        e = create_engine(cfg)
        assert e.engine_type == etype, (
            f"Expected engine_type={etype!r}, got {e.engine_type!r}"
        )
