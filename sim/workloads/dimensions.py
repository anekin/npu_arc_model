"""Canonical dimension fields for workload graph dimension-binding.

Each dimension is an independent, orthogonal concept.  Changing one dimension
must never implicitly substitute or auto-change another.  Dimensions bind to
named symbolic axes used in graph tensor shapes.

Dimensions (from the plan):
- ``request_batch``: number of simultaneous requests in a batch
- ``active_sequences``: number of concurrently active sequences (decode)
- ``token_block``: token block size (prefill chunk)
- ``image_count``: number of images per inference
- ``action_horizon``: VLA action horizon (future steps predicted)
- ``flow_steps``: continuous flow model steps
- ``resident_models``: number of models resident in memory
- ``inflight_jobs``: number of concurrently executing jobs
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts.errors import DimensionBindingError
from workloads.schema import WorkloadGraphV1

# ── Named symbolic axis ──────────────────────────────────────────────────────

# Canonical symbolic axis names that dimensions bind to
AXIS_BATCH = "batch"
AXIS_SEQUENCES = "active_sequences"
AXIS_TOKEN_BLOCK = "token_block"
AXIS_IMAGE_COUNT = "image_count"
AXIS_ACTION_HORIZON = "action_horizon"
AXIS_FLOW_STEPS = "flow_steps"
AXIS_RESIDENT_MODELS = "resident_models"
AXIS_INFLIGHT_JOBS = "inflight_jobs"

ALL_AXES: set[str] = frozenset(
    {
        AXIS_BATCH,
        AXIS_SEQUENCES,
        AXIS_TOKEN_BLOCK,
        AXIS_IMAGE_COUNT,
        AXIS_ACTION_HORIZON,
        AXIS_FLOW_STEPS,
        AXIS_RESIDENT_MODELS,
        AXIS_INFLIGHT_JOBS,
    }
)
"""Frozen set of all canonical symbolic axis names."""


# ── Edge batch values (acceptance criteria) ──────────────────────────────────

STANDARD_BATCH_EDGES: dict[str, set[int]] = {
    AXIS_BATCH: {1, 2, 4, 8},
    AXIS_SEQUENCES: {1, 2, 4, 8},
}
"""Standard batch edges from the plan."""

STRESS_BATCH_EDGES: dict[str, set[int]] = {
    AXIS_BATCH: {16},
    AXIS_SEQUENCES: {16},
}
"""Stress batch edges from the plan."""

TOKEN_BLOCK_EDGES: set[int] = {16, 32, 64, 128, 256}
"""Standard token block sizes."""

TOKEN_BLOCK_VLM_VLA_EXT: set[int] = {512, 1024}
"""Extended token block sizes for VLM/VLA scenarios."""

IMAGE_COUNT_EDGES: set[int] = {1, 2, 3, 4}
"""Image count edges."""

ACTION_HORIZON_EDGES: set[int] = {8, 10, 25, 50}
"""Action horizon edges."""

FLOW_STEPS_EDGES: set[int] = {4, 8, 10}
"""Flow steps edges."""

RESIDENT_MODELS_EDGES: set[int] = {4, 8}
"""Resident model count edges."""

INFLIGHT_JOBS_EDGES: set[int] = {4, 8, 16}
"""Inflight jobs edges."""


# ── Dimension bindings ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DimensionBindings:
    """Immutable dimension bindings mapping symbolic axes to concrete values.

    Each field is optional — unset dimensions are ``None``.  Before lowering
    a workload graph to cost estimation, all referenced symbolic axes must be
    bound.  ``None`` means "not applicable for this workload type".

    Orthogonality: setting ``request_batch=8`` does NOT auto-set
    ``active_sequences`` or ``token_block``.  Each dimension is independent.
    """

    request_batch: int | None = None
    """Number of simultaneous requests in a batch.  Maps to axis ``batch``."""

    active_sequences: int | None = None
    """Number of concurrently active sequences (decode).  Maps to axis ``active_sequences``."""

    token_block: int | None = None
    """Token block size for prefill chunking.  Maps to axis ``token_block``."""

    image_count: int | None = None
    """Number of images per inference (VLM/VLA).  Maps to axis ``image_count``."""

    action_horizon: int | None = None
    """VLA action horizon — number of future steps predicted.  Maps to axis ``action_horizon``."""

    flow_steps: int | None = None
    """Number of continuous flow model steps.  Maps to axis ``flow_steps``."""

    resident_models: int | None = None
    """Number of models resident in memory.  Maps to axis ``resident_models``."""

    inflight_jobs: int | None = None
    """Number of concurrently executing jobs.  Maps to axis ``inflight_jobs``."""

    # Arbitrary additional bindings for schema symbols not in canonical list
    extra: dict[str, int] = field(default_factory=dict)
    """Additional symbolic dimension bindings beyond the canonical set."""

    def __post_init__(self):
        """Validate positive integer values and canonical axis overlap."""
        canonical_fields = [
            ("request_batch", self.request_batch),
            ("active_sequences", self.active_sequences),
            ("token_block", self.token_block),
            ("image_count", self.image_count),
            ("action_horizon", self.action_horizon),
            ("flow_steps", self.flow_steps),
            ("resident_models", self.resident_models),
            ("inflight_jobs", self.inflight_jobs),
        ]
        for name, value in canonical_fields:
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        for name, value in canonical_fields:
            if value is not None and not isinstance(value, int):
                raise ValueError(f"{name} must be int or None, got {type(value).__name__}")

        # Warn about extra keys that overlap canonical names
        for key in self.extra:
            if key in {
                "request_batch",
                "active_sequences",
                "token_block",
                "image_count",
                "action_horizon",
                "flow_steps",
                "resident_models",
                "inflight_jobs",
            }:
                raise ValueError(f"extra dimension {key!r} shadows canonical field — use the dedicated field instead")

    def to_dict(self) -> dict[str, int]:
        """Return all bound dimensions as a flat dict (name → int).

        Unset dimensions are excluded.
        """
        result: dict[str, int] = {}
        axis_map = [
            (AXIS_BATCH, self.request_batch),
            (AXIS_SEQUENCES, self.active_sequences),
            (AXIS_TOKEN_BLOCK, self.token_block),
            (AXIS_IMAGE_COUNT, self.image_count),
            (AXIS_ACTION_HORIZON, self.action_horizon),
            (AXIS_FLOW_STEPS, self.flow_steps),
            (AXIS_RESIDENT_MODELS, self.resident_models),
            (AXIS_INFLIGHT_JOBS, self.inflight_jobs),
        ]
        for key, value in axis_map:
            if value is not None:
                result[key] = value
        result.update(self.extra)
        return result

    def unbound(self, required_axes: set[str]) -> set[str]:
        """Return the set of required axes that are not bound.

        Args:
            required_axes: The symbolic axis names the workload needs.
        """
        bound = set(self.to_dict().keys())
        return required_axes - bound


def apply_bindings(
    graph: WorkloadGraphV1,
    bindings: DimensionBindings,
) -> WorkloadGraphV1:
    """Return a new graph with all symbolic dimensions replaced by bound values.

    Args:
        graph: The workload graph to bind.
        bindings: Concrete dimension bindings.

    Raises:
        DimensionBindingError: if a symbolic dimension is unbound.
    """
    from workloads.schema import TensorSpec  # local imports avoid circular dependency

    binding_dict = bindings.to_dict()
    new_tensors: list[TensorSpec] = []

    for tensor in graph.tensors:
        new_shape: list[int | str] = []
        for dim in tensor.shape:
            if isinstance(dim, str):
                if dim not in binding_dict:
                    raise DimensionBindingError(
                        f"symbolic dimension {dim!r} is not bound",
                        dimension=dim,
                    )
                new_shape.append(binding_dict[dim])
            else:
                new_shape.append(dim)
        new_tensors.append(tensor.model_copy(update={"shape": new_shape}))

    return graph.model_copy(update={"tensors": new_tensors})
