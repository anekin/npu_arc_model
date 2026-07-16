"""Physical memory-unit conversion, coupling and capacity accounting."""

from typing import Any, Dict, Optional

from dse.types import MemoryFootprint, WorkloadSpec


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


def estimate_memory_footprint(
    config: Dict[str, Any], workload: WorkloadSpec,
) -> MemoryFootprint:
    """Estimate persistent capacity needed by one deployment workload.

    Capacity is decimal GB, matching memory product specifications. KV cache is
    provisioned for the maximum prompt + output context for every concurrent
    request. Runtime reserve is explicit so scenarios can replace the default.
    """
    weights = workload.parameters_b * 1e9 * workload.weight_bits / 8.0
    kv_per_token = (
        2 * workload.layers * workload.kv_heads * workload.head_dim
        * workload.kv_bits / 8.0
    )
    kv_cache = (
        kv_per_token * workload.max_context_tokens
        * workload.concurrent_requests
    )
    activation_elems = (
        max(workload.hidden, workload.intermediate)
        * workload.decode_batch_size * 3
    )
    activations = activation_elems * workload.activation_bits / 8.0
    runtime = workload.runtime_reserve_mb * 1e6
    required = weights + kv_cache + activations + runtime

    memory = config.get("memory", {})
    onchip = config.get("on_chip_memory", {})
    installed_gb = float(
        memory.get("capacity_gb", onchip.get("capacity_gb", 0.0))
    )
    capacity_specified = installed_gb > 0
    usable_fraction = float(memory.get("capacity_usable_fraction", 0.90))
    if not 0 < usable_fraction <= 1:
        raise ValueError("capacity_usable_fraction must be in (0, 1]")
    usable_gb = installed_gb * usable_fraction if capacity_specified else 0.0
    required_gb = required / 1e9
    margin_gb = usable_gb - required_gb if capacity_specified else 0.0

    return MemoryFootprint(
        weights_gb=round(weights / 1e9, 4),
        kv_cache_gb=round(kv_cache / 1e9, 4),
        activations_gb=round(activations / 1e9, 4),
        runtime_reserve_gb=round(runtime / 1e9, 4),
        required_gb=round(required_gb, 4),
        installed_gb=round(installed_gb, 4),
        usable_gb=round(usable_gb, 4),
        margin_gb=round(margin_gb, 4),
        fits=(not capacity_specified or required_gb <= usable_gb),
        capacity_specified=capacity_specified,
    )


def couple_on_chip_bandwidth(
    config: Dict[str, Any], logic_area_mm2: Optional[float],
) -> Optional[float]:
    """Apply bandwidth/area coupling without exceeding the rated interface."""
    onchip = config.get("on_chip_memory", {})
    if not onchip or float(onchip.get("capacity_gb", 0.0)) <= 0:
        return None

    rated_bandwidth = float(onchip.get(
        "rated_bandwidth_gbps", onchip.get("bandwidth_gbps", 0.0),
    ))
    onchip["rated_bandwidth_gbps"] = rated_bandwidth
    bw_per_mm2 = float(onchip.get("bw_per_mm2_gbps", 0.0))
    if bw_per_mm2 > 0 and logic_area_mm2 is not None:
        area_limited_bandwidth = float(logic_area_mm2) * bw_per_mm2
        bandwidth_gbps = (
            min(area_limited_bandwidth, rated_bandwidth)
            if rated_bandwidth > 0 else area_limited_bandwidth
        )
    else:
        bandwidth_gbps = rated_bandwidth

    onchip["bandwidth_gbps"] = bandwidth_gbps
    memory = config.setdefault("memory", {})
    memory["bandwidth_gbps"] = bandwidth_gbps
    memory["dram_efficiency"] = float(memory.get("dram_efficiency", 1.0))
    return bandwidth_gbps
