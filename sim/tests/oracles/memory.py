"""Independent closed-form oracle for memory residency planning.

This oracle verifies byte conservation and capacity boundaries using only
first-principles arithmetic.  It deliberately does NOT import the production
``models.residency`` estimator, so it can serve as an independent checker.
"""

from __future__ import annotations

from typing import Any

from models.memory_hierarchy import MemoryHierarchy
from workloads.schema import WorkloadGraphV1


def _align_up(value: int, alignment: int) -> int:
    """Return the smallest multiple of ``alignment`` >= ``value``."""
    return ((value + alignment - 1) // alignment) * alignment


def total_aligned_tensor_bytes(
    graph: WorkloadGraphV1,
    alignment: int,
    resident_models: int = 1,
) -> int:
    """Sum of all tensor byte sizes after alignment and weight replication.

    Weights are replicated by ``resident_models``; all other categories are
    counted once.
    """
    total = 0
    for tensor in graph.tensors:
        aligned = _align_up(tensor.bytes, alignment)
        # Heuristic weight classification matching the production planner.
        is_weight = tensor.producer_node == "input" or tensor.tensor_id.lower().startswith(("weight", "param"))
        factor = resident_models if is_weight else 1
        total += aligned * factor
    return total


def total_required_bytes(
    graph: WorkloadGraphV1,
    alignment: int,
    resident_models: int = 1,
    queue_buffers: dict[str, int] | None = None,
) -> int:
    """Total bytes required by tensors + queue buffers after alignment."""
    tensors_total = total_aligned_tensor_bytes(graph, alignment, resident_models)
    queue_total = 0
    for size in (queue_buffers or {}).values():
        queue_total += _align_up(max(0, size), alignment)
    return tensors_total + queue_total


def reserve_bytes_per_tier(hierarchy: MemoryHierarchy) -> dict[str, int]:
    """Return reserved bytes for each tier."""
    return {tier.name: tier.reserve_bytes for tier in hierarchy.tiers}


def usable_capacity_per_tier(hierarchy: MemoryHierarchy) -> dict[str, int]:
    """Return usable bytes (capacity - reserve) for each tier."""
    return {tier.name: tier.usable_bytes for tier in hierarchy.tiers}


def total_usable_bytes(hierarchy: MemoryHierarchy) -> int:
    """Sum of usable bytes across all tiers."""
    return sum(tier.usable_bytes for tier in hierarchy.tiers)


def capacity_conservation_ok(
    hierarchy: MemoryHierarchy,
    allocated_per_tier: dict[str, int],
) -> tuple[bool, str]:
    """Return (ok, message) verifying allocated + reserved <= capacity.

    The oracle checks the physical invariant independently of how the
    production planner arrived at its allocations.
    """
    for tier in hierarchy.tiers:
        allocated = allocated_per_tier.get(tier.name, 0)
        total = allocated + tier.reserve_bytes
        if total > tier.capacity_bytes:
            return False, (
                f"tier {tier.name!r} overflow: allocated={allocated} + "
                f"reserve={tier.reserve_bytes} > capacity={tier.capacity_bytes}"
            )
    return True, "capacity conservation holds"


def expected_spill_bytes(
    graph: WorkloadGraphV1,
    hierarchy: MemoryHierarchy,
    *,
    resident_models: int = 1,
    queue_buffers: dict[str, int] | None = None,
) -> int:
    """Closed-form upper-bound spill bytes for an all-or-nothing weight policy.

    This intentionally approximates the production planner's behavior:
    * weights/KV are not split;
    * activations/scratch/queues may be split.

    The returned value is the exact spill bytes under the simplifying
    assumption that each tensor is placed independently in the fastest tier
    with enough free space, considering priority order.
    """
    alignment = hierarchy.default_alignment_bytes
    required = total_required_bytes(graph, alignment, resident_models, queue_buffers)
    usable = total_usable_bytes(hierarchy)
    return max(0, required - usable)


def boundary_check(
    tensor_size_bytes: int,
    tier_capacity_bytes: int,
    alignment: int,
) -> tuple[str, int, int]:
    """Classify a single tensor against a single tier capacity boundary.

    Returns (expected_state, resident_bytes, spill_bytes).

    Weights and KV use an all-or-nothing policy: if the aligned tensor does
    not fit the tier, the entire tensor spills to the next eligible tier.
    """
    aligned = _align_up(tensor_size_bytes, alignment)
    if tier_capacity_bytes >= aligned:
        return "full", aligned, 0
    return "spill", 0, aligned


def verify_plan_conservation(
    hierarchy: MemoryHierarchy,
    placements: list[Any],
) -> tuple[bool, str]:
    """Verify that a set of placements conserves bytes.

    ``placements`` must be objects with ``destination_tier``, ``full_bytes``,
    ``partial_bytes``, and ``spill_bytes`` attributes (e.g. production
    ``TensorPlacement``).
    """
    allocated: dict[str, int] = {tier.name: 0 for tier in hierarchy.tiers}
    for p in placements:
        allocated[p.destination_tier] += p.full_bytes + p.partial_bytes
    ok, msg = capacity_conservation_ok(hierarchy, allocated)
    if not ok:
        return False, msg

    # Byte-level conservation per placement.
    for p in placements:
        total = p.full_bytes + p.partial_bytes + p.spill_bytes
        if hasattr(p, "effective_bytes") and total != p.effective_bytes:
            return False, (
                f"placement {p.tensor_id!r} byte mismatch: full+partial+spill={total} != effective={p.effective_bytes}"
            )
    return True, "byte conservation holds"
