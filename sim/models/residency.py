"""Unified memory residency and spill planning.

Builds an immutable ``MemoryAccessPlan`` from a workload graph, a memory
hierarchy, and runtime parameters (resident model count, queue buffers).

Placement priority:
  1. persistent weights
  2. KV cache
  3. live activations / scratch
  4. queue buffers

Weights and KV are placed all-or-nothing into the fastest eligible tier that
can hold them; activations and scratch may be split across tiers (partial).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from contracts.errors import ConfigError, CoverageError
from contracts.identity import digest_sha256
from models.memory_hierarchy import MemoryHierarchy, MemoryTier, MemoryTierName
from workloads.schema import Precision, WorkloadGraphV1


def _precision_bits(precision: Precision) -> int:
    """Return element size in bits for a canonical precision."""
    mapping = {
        Precision.FP32: 32,
        Precision.FP16: 16,
        Precision.BF16: 16,
        Precision.INT8: 8,
        Precision.INT4: 4,
        Precision.INT2: 2,
        Precision.MIXED_INT8_INT4: 4,
        Precision.MIXED_INT8_INT8: 8,
    }
    return mapping.get(precision, 8)


def _tensor_footprint_bytes(tensor: Any) -> int:
    """Return tensor byte footprint, computing from shape if bytes is unset."""
    if tensor.bytes > 0:
        return tensor.bytes

    if all(isinstance(d, int) for d in tensor.shape):
        elements = 1
        for d in tensor.shape:
            elements *= d
        bits = _precision_bits(tensor.precision)
        return elements * bits // 8

    return 0


# Category priority: lower number = higher priority.
_WEIGHT_PRIORITY = 0
_KV_PRIORITY = 1
_ACTIVATION_PRIORITY = 2
_QUEUE_PRIORITY = 3

_CATEGORY_PRIORITY: Dict[str, int] = {
    "weight": _WEIGHT_PRIORITY,
    "kv": _KV_PRIORITY,
    "activation": _ACTIVATION_PRIORITY,
    "scratch": _ACTIVATION_PRIORITY,
    "queue": _QUEUE_PRIORITY,
}

# Categories that are never split across tiers (weights, KV).
_ATOMIC_CATEGORIES = frozenset({"weight", "kv"})

# Categories that may be partially resident in a faster tier and spilled to a
# slower tier (activations, scratch, queues).
_SPLITABLE_CATEGORIES = frozenset({"activation", "scratch", "queue"})


def _align_up(value: int, alignment: int) -> int:
    """Return the smallest multiple of ``alignment`` >= ``value``."""
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


class TensorPlacement(BaseModel):
    """Residency decision for a single tensor or queue buffer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tensor_id: str = Field(..., description="Stable tensor or buffer identifier")
    category: str = Field(..., description="weight | kv | activation | scratch | queue")
    size_bytes: int = Field(..., ge=0, description="Original tensor size in bytes")
    alignment_bytes: int = Field(..., ge=1)
    aligned_bytes: int = Field(..., ge=0, description="Size after alignment")
    replication_factor: int = Field(default=1, ge=1, description="Resident-model replication count")
    effective_bytes: int = Field(..., ge=0, description="aligned_bytes × replication_factor")

    state: str = Field(..., pattern=r"^(full|partial|spill)$")
    destination_tier: str = Field(..., description="Tier holding the resident portion")
    spill_tier: Optional[str] = Field(default=None, description="Tier/host receiving the spilled portion")

    full_bytes: int = Field(default=0, ge=0)
    partial_bytes: int = Field(default=0, ge=0)
    spill_bytes: int = Field(default=0, ge=0)

    evict_bytes: int = Field(default=0, ge=0, description="Bytes evicted from faster tiers")
    reload_bytes: int = Field(default=0, ge=0, description="Bytes reloaded from slower tiers")
    access_bytes: int = Field(default=0, ge=0, description="Total bytes accessed for this tensor")


class TierSummary(BaseModel):
    """Capacity accounting for one tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    capacity_bytes: int
    reserve_bytes: int
    allocated_bytes: int
    free_bytes: int
    spill_in_bytes: int = 0
    spill_out_bytes: int = 0


class AccessStream(BaseModel):
    """A directed byte stream between two tiers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stream_id: str
    source_tier: str
    destination_tier: str
    category: str
    bytes_: int = Field(..., ge=0, alias="bytes")


class MemoryAccessPlan(BaseModel):
    """Immutable output of the residency planner.

    The ``digest`` field is a deterministic SHA-256 over the canonical plan
    dict.  Two plans built from equivalent inputs always produce the same
    digest, regardless of dict-insertion order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hierarchy: MemoryHierarchy
    placements: Tuple[TensorPlacement, ...]
    tier_summaries: Tuple[TierSummary, ...]
    access_streams: Tuple[AccessStream, ...]
    total_access_bytes: Dict[str, int]
    spill_bytes_total: int
    resident_bytes_total: int
    digest: str

    def placements_for_category(self, category: str) -> Tuple[TensorPlacement, ...]:
        """Return all placements for a given category."""
        return tuple(p for p in self.placements if p.category == category)

    def allocated_bytes_in_tier(self, tier_name: str) -> int:
        """Return total allocated bytes in a tier."""
        for summary in self.tier_summaries:
            if summary.name == tier_name:
                return summary.allocated_bytes
        return 0

    def spill_bytes_for_category(self, category: str) -> int:
        """Return total spill bytes for a category."""
        return sum(p.spill_bytes for p in self.placements if p.category == category)

    def resident_bytes_for_category(self, category: str) -> int:
        """Return total resident bytes for a category."""
        return sum(p.full_bytes + p.partial_bytes for p in self.placements if p.category == category)

    def fastest_allocated_tier(self, category: str) -> Optional[str]:
        """Return the fastest tier that holds any bytes of ``category``."""
        tier_names = {p.destination_tier for p in self.placements if p.category == category}
        for tier in self.hierarchy.tiers:
            if tier.name in tier_names:
                return tier.name
        return None

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Return a deterministic dict suitable for hashing."""
        return {
            "hierarchy": self.hierarchy.model_dump(mode="json"),
            "placements": [p.model_dump(mode="json") for p in self.placements],
            "tier_summaries": [t.model_dump(mode="json") for t in self.tier_summaries],
            "access_streams": [s.model_dump(mode="json") for s in self.access_streams],
            "total_access_bytes": dict(sorted(self.total_access_bytes.items())),
            "spill_bytes_total": self.spill_bytes_total,
            "resident_bytes_total": self.resident_bytes_total,
        }


def _category_priority(category: str) -> int:
    """Return placement priority for a category."""
    return _CATEGORY_PRIORITY.get(category, _ACTIVATION_PRIORITY)


def _classify_tensor(graph: WorkloadGraphV1, tensor_id: str) -> str:
    """Heuristic classification of a tensor into a residency category.

    Rules (in order):
      * producer == "input" or tensor_id starts with ``weight``/``param`` → weight
      * ``kv`` appears in tensor_id or producer node op_type → kv
      * ``queue``/``fifo`` appears in tensor_id or producer node op_type → queue
      * otherwise → activation/scratch
    """
    tensor = graph.get_tensor(tensor_id)
    t_id_lower = tensor_id.lower()
    producer_id = tensor.producer_node or ""

    if t_id_lower.startswith(("weight", "param")):
        return "weight"

    try:
        producer_node = graph.get_node(producer_id)
        producer_op = producer_node.op_type.lower()
    except KeyError:
        producer_op = ""

    if "kv" in t_id_lower or producer_op in {"kv_cache", "kv_load", "kv_store"}:
        return "kv"
    if "queue" in t_id_lower or "fifo" in t_id_lower or producer_op in {"queue", "fifo"}:
        return "queue"
    if producer_op in {"constant", "parameter", "weight"}:
        return "weight"

    return "activation"


def _eligible_tiers(category: str, hierarchy: MemoryHierarchy) -> Tuple[MemoryTier, ...]:
    """Return tiers eligible for ``category``, ordered fastest to slowest.

    Weights skip SRAM: their fastest eligible tier is on-chip 3D DRAM if
    present, otherwise external DRAM.  All other categories may use SRAM.
    """
    tiers = hierarchy.tiers
    if category == "weight":
        for idx, tier in enumerate(tiers):
            if tier.name == MemoryTierName.ON_CHIP_3D_DRAM.value:
                return tiers[idx:]
        return tiers
    return tiers


def _validate_alias(graph: WorkloadGraphV1, tensor_id: str) -> None:
    """Raise ConfigError for an alias with mismatched size or missing target."""
    tensor = graph.get_tensor(tensor_id)
    alias_of = tensor.alias_of
    if not alias_of:
        return
    try:
        target = graph.get_tensor(alias_of)
    except KeyError as exc:
        raise ConfigError(
            f"tensor {tensor_id!r} aliases non-existent tensor {alias_of!r}",
            field_path=f"tensors.{tensor_id}.alias_of",
        ) from exc

    target_bytes = _tensor_footprint_bytes(target)
    source_bytes = _tensor_footprint_bytes(tensor)
    if target_bytes != source_bytes:
        raise ConfigError(
            f"alias {tensor_id!r} ({source_bytes} B) must match "
            f"target {alias_of!r} ({target_bytes} B)",
            field_path=f"tensors.{tensor_id}.bytes",
        )


def _placement_for_tensor(
    graph: WorkloadGraphV1,
    tensor_id: str,
    category: str,
    hierarchy: MemoryHierarchy,
    allocations: Dict[str, int],
    reserves: Dict[str, int],
    resident_models: int,
) -> TensorPlacement:
    """Place one tensor and return its immutable placement."""
    tensor = graph.get_tensor(tensor_id)
    alignment = hierarchy.default_alignment_bytes
    aligned = _align_up(_tensor_footprint_bytes(tensor), alignment)
    replication = resident_models if category == "weight" else 1
    effective = aligned * replication

    eligible = _eligible_tiers(category, hierarchy)
    if not eligible:
        raise CoverageError(
            f"no eligible tier for {category} tensor {tensor_id!r}",
            missing_axes=[f"tensors.{tensor_id}"],
        )

    atomic = category in _ATOMIC_CATEGORIES
    parts: List[Tuple[str, int]] = []
    remaining = effective

    if atomic:
        # All-or-nothing: choose the fastest eligible tier that can hold the
        # entire replicated tensor.
        for tier in eligible:
            usable = tier.capacity_bytes - allocations[tier.name] - reserves[tier.name]
            if usable >= effective:
                parts.append((tier.name, effective))
                allocations[tier.name] += effective
                remaining = 0
                break
    else:
        # Splittable: fill each eligible tier greedily.
        for tier in eligible:
            if remaining <= 0:
                break
            usable = tier.capacity_bytes - allocations[tier.name] - reserves[tier.name]
            if usable <= 0:
                continue
            place = min(remaining, usable)
            parts.append((tier.name, place))
            allocations[tier.name] += place
            remaining -= place

    first_eligible = eligible[0]

    if not parts:
        state = "spill"
        destination = eligible[-1].name
        spill_tier: Optional[str] = None
        full_bytes = 0
        partial_bytes = 0
        spill_bytes = effective
    elif remaining == 0 and len(parts) == 1 and parts[0][0] == first_eligible.name:
        state = "full"
        destination = parts[0][0]
        spill_tier = None
        full_bytes = effective
        partial_bytes = 0
        spill_bytes = 0
    elif parts[0][0] == first_eligible.name:
        state = "partial"
        destination = parts[0][0]
        spill_tier = parts[1][0] if len(parts) > 1 else None
        full_bytes = 0
        partial_bytes = parts[0][1]
        spill_bytes = effective - partial_bytes
    else:
        # Fully resident in a slower tier than the first eligible one.
        state = "spill"
        destination = parts[0][0]
        spill_tier = None
        full_bytes = 0
        partial_bytes = 0
        spill_bytes = effective

    evict = spill_bytes if state in ("partial", "spill") else 0
    reload_ = spill_bytes if state in ("partial", "spill") else 0

    return TensorPlacement(
        tensor_id=tensor_id,
        category=category,
        size_bytes=tensor.bytes,
        alignment_bytes=alignment,
        aligned_bytes=aligned,
        replication_factor=replication,
        effective_bytes=effective,
        state=state,
        destination_tier=destination,
        spill_tier=spill_tier,
        full_bytes=full_bytes,
        partial_bytes=partial_bytes,
        spill_bytes=spill_bytes,
        evict_bytes=evict,
        reload_bytes=reload_,
        access_bytes=effective,
    )


def _queue_placements(
    queue_buffers: Dict[str, int],
    hierarchy: MemoryHierarchy,
    allocations: Dict[str, int],
    reserves: Dict[str, int],
) -> List[TensorPlacement]:
    """Create splittable placements for explicit queue buffers."""
    placements: List[TensorPlacement] = []
    alignment = hierarchy.default_alignment_bytes
    eligible = _eligible_tiers("queue", hierarchy)
    first_eligible = eligible[0] if eligible else None

    for buffer_id, size_bytes in sorted(queue_buffers.items()):
        if size_bytes < 0:
            raise ConfigError(
                f"queue buffer {buffer_id!r} has negative size {size_bytes}",
                field_path=f"queue_buffers.{buffer_id}",
            )
        aligned = _align_up(size_bytes, alignment)
        effective = aligned
        parts: List[Tuple[str, int]] = []
        remaining = effective
        for tier in eligible:
            if remaining <= 0:
                break
            usable = tier.capacity_bytes - allocations[tier.name] - reserves[tier.name]
            if usable <= 0:
                continue
            place = min(remaining, usable)
            parts.append((tier.name, place))
            allocations[tier.name] += place
            remaining -= place

        if not parts:
            state = "spill"
            destination = eligible[-1].name if eligible else "host"
            spill_tier = None
            full_bytes = 0
            partial_bytes = 0
            spill_bytes = effective
        elif remaining == 0 and len(parts) == 1 and first_eligible and parts[0][0] == first_eligible.name:
            state = "full"
            destination = parts[0][0]
            spill_tier = None
            full_bytes = effective
            partial_bytes = 0
            spill_bytes = 0
        elif first_eligible and parts[0][0] == first_eligible.name:
            state = "partial"
            destination = parts[0][0]
            spill_tier = parts[1][0] if len(parts) > 1 else None
            full_bytes = 0
            partial_bytes = parts[0][1]
            spill_bytes = effective - partial_bytes
        else:
            state = "spill"
            destination = parts[0][0]
            spill_tier = None
            full_bytes = 0
            partial_bytes = 0
            spill_bytes = effective

        evict = spill_bytes if state in ("partial", "spill") else 0
        reload_ = spill_bytes if state in ("partial", "spill") else 0

        placements.append(
            TensorPlacement(
                tensor_id=buffer_id,
                category="queue",
                size_bytes=size_bytes,
                alignment_bytes=alignment,
                aligned_bytes=aligned,
                replication_factor=1,
                effective_bytes=effective,
                state=state,
                destination_tier=destination,
                spill_tier=spill_tier,
                full_bytes=full_bytes,
                partial_bytes=partial_bytes,
                spill_bytes=spill_bytes,
                evict_bytes=evict,
                reload_bytes=reload_,
                access_bytes=effective,
            )
        )
    return placements


def _build_tier_summaries(
    hierarchy: MemoryHierarchy,
    allocations: Dict[str, int],
    spill_in: Dict[str, int],
    spill_out: Dict[str, int],
) -> Tuple[TierSummary, ...]:
    """Return immutable tier summaries sorted by tier name."""
    summaries: List[TierSummary] = []
    for tier in hierarchy.tiers:
        reserve = tier.reserve_bytes
        allocated = allocations[tier.name]
        summaries.append(
            TierSummary(
                name=tier.name,
                capacity_bytes=tier.capacity_bytes,
                reserve_bytes=reserve,
                allocated_bytes=allocated,
                free_bytes=max(0, tier.capacity_bytes - reserve - allocated),
                spill_in_bytes=spill_in.get(tier.name, 0),
                spill_out_bytes=spill_out.get(tier.name, 0),
            )
        )
    return tuple(sorted(summaries, key=lambda s: s.name))


def build_memory_access_plan(
    graph: WorkloadGraphV1,
    hierarchy: MemoryHierarchy,
    *,
    resident_models: int = 1,
    queue_buffers: Optional[Dict[str, int]] = None,
    allow_spill: bool = True,
) -> MemoryAccessPlan:
    """Build an immutable memory access plan from a workload graph.

    Args:
        graph: The workload graph whose tensors will be placed.
        hierarchy: Ordered memory hierarchy (fastest tier first).
        resident_models: Number of model weight sets resident simultaneously.
        queue_buffers: Optional mapping of queue buffer id → size in bytes.
        allow_spill: If False, raise ``CoverageError`` when any bytes spill
            beyond the slowest tier.

    Returns:
        ``MemoryAccessPlan`` with deterministic digest.

    Raises:
        ConfigError: For invalid reserve/alignment/capacity or alias mismatch.
        CoverageError: When ``allow_spill=False`` and capacity is exceeded.
    """
    if resident_models <= 0:
        raise ConfigError(
            f"resident_models must be positive, got {resident_models}",
            field_path="resident_models",
        )

    # Validate tiers.
    for tier in hierarchy.tiers:
        if tier.reserve_bytes > tier.capacity_bytes:
            raise ConfigError(
                f"tier {tier.name!r} reserve {tier.reserve_bytes} B exceeds "
                f"capacity {tier.capacity_bytes} B",
                field_path=f"hierarchy.{tier.name}.reserve_fraction",
            )

    allocations: Dict[str, int] = {tier.name: 0 for tier in hierarchy.tiers}
    reserves: Dict[str, int] = {tier.name: tier.reserve_bytes for tier in hierarchy.tiers}

    # Classify and sort tensors by priority, then by size descending.
    classified: List[Tuple[str, str, int]] = []
    for tensor in graph.tensors:
        _validate_alias(graph, tensor.tensor_id)
        category = _classify_tensor(graph, tensor.tensor_id)
        priority = _category_priority(category)
        classified.append((tensor.tensor_id, category, priority))

    classified.sort(key=lambda item: (item[2], -graph.get_tensor(item[0]).bytes))

    placements: List[TensorPlacement] = []
    for tensor_id, category, _ in classified:
        placement = _placement_for_tensor(
            graph=graph,
            tensor_id=tensor_id,
            category=category,
            hierarchy=hierarchy,
            allocations=allocations,
            reserves=reserves,
            resident_models=resident_models,
        )
        placements.append(placement)

    queue_bufs = queue_buffers or {}
    queue_placements = _queue_placements(
        queue_buffers=queue_bufs,
        hierarchy=hierarchy,
        allocations=allocations,
        reserves=reserves,
    )
    placements.extend(queue_placements)

    # Re-sort placements by tensor_id for deterministic output.
    placements.sort(key=lambda p: p.tensor_id)

    spill_total = sum(p.spill_bytes for p in placements)
    if not allow_spill and spill_total > 0:
        raise CoverageError(
            f"memory hierarchy cannot hold all tensors: {spill_total} bytes spill "
            "beyond the slowest tier",
            missing_axes=["hierarchy.capacity_bytes"],
        )

    # Build access streams and per-tier access bytes.
    total_access: Dict[str, int] = {}
    spill_in: Dict[str, int] = {tier.name: 0 for tier in hierarchy.tiers}
    spill_out: Dict[str, int] = {tier.name: 0 for tier in hierarchy.tiers}
    streams: List[AccessStream] = []

    for placement in placements:
        # Resident portion is read from its destination tier.
        resident = placement.full_bytes + placement.partial_bytes
        if resident > 0:
            total_access[placement.destination_tier] = (
                total_access.get(placement.destination_tier, 0) + resident
            )
        if placement.spill_bytes > 0:
            # Spilled bytes are read from the spill tier (host or next tier).
            spill_tier = placement.spill_tier or (
                hierarchy.tiers[-1].name if hierarchy.tiers else "host"
            )
            total_access[spill_tier] = total_access.get(spill_tier, 0) + placement.spill_bytes
            spill_in[spill_tier] = spill_in.get(spill_tier, 0) + placement.spill_bytes
            spill_out[placement.destination_tier] = (
                spill_out.get(placement.destination_tier, 0) + placement.spill_bytes
            )
            streams.append(
                AccessStream(
                    stream_id=f"{placement.tensor_id}_spill",
                    source_tier=placement.destination_tier,
                    destination_tier=spill_tier,
                    category=placement.category,
                    bytes=placement.spill_bytes,
                )
            )

    tier_summaries = _build_tier_summaries(hierarchy, allocations, spill_in, spill_out)

    resident_total = sum(p.full_bytes + p.partial_bytes for p in placements)

    plan = MemoryAccessPlan(
        hierarchy=hierarchy,
        placements=tuple(placements),
        tier_summaries=tier_summaries,
        access_streams=tuple(streams),
        total_access_bytes=dict(sorted(total_access.items())),
        spill_bytes_total=spill_total,
        resident_bytes_total=resident_total,
        digest="",  # placeholder; replaced below
    )

    digest = digest_sha256(plan.to_canonical_dict())
    # Pydantic frozen model: rebuild with digest.
    plan = plan.model_copy(update={"digest": digest})
    return plan
