"""Physical memory-unit conversion and scenario memory coupling."""

from typing import Any, Dict, Optional


def bandwidth_bytes_per_cycle(memory: Dict[str, Any], frequency_mhz: float) -> float:
    """Convert physical GB/s to bytes/cycle at the configured clock."""
    frequency_mhz = float(frequency_mhz)
    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")
    if "bandwidth_gbps" in memory:
        return float(memory["bandwidth_gbps"]) * 1000.0 / frequency_mhz
    return float(memory.get("bandwidth_bytes_per_cycle", 0.0))


def effective_bandwidth_bytes_per_cycle(
    memory: Dict[str, Any], frequency_mhz: float, multiplier: float = 1.0,
) -> float:
    raw = bandwidth_bytes_per_cycle(memory, frequency_mhz)
    efficiency = float(memory.get("dram_efficiency", 1.0))
    return raw * efficiency * float(multiplier)


def couple_on_chip_bandwidth(
    config: Dict[str, Any], logic_area_mm2: Optional[float],
) -> Optional[float]:
    """Apply the scenario's bandwidth/area coupling and return GB/s."""
    onchip = config.get("on_chip_memory", {})
    if not onchip or float(onchip.get("capacity_gb", 0.0)) <= 0:
        return None

    bw_per_mm2 = float(onchip.get("bw_per_mm2_gbps", 0.0))
    if bw_per_mm2 > 0 and logic_area_mm2 is not None:
        bandwidth_gbps = float(logic_area_mm2) * bw_per_mm2
    else:
        bandwidth_gbps = float(onchip.get("bandwidth_gbps", 0.0))

    onchip["bandwidth_gbps"] = bandwidth_gbps
    memory = config.setdefault("memory", {})
    memory["bandwidth_gbps"] = bandwidth_gbps
    memory["dram_efficiency"] = float(memory.get("dram_efficiency", 1.0))
    return bandwidth_gbps
