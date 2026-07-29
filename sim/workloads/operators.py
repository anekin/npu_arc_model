"""Fail-closed operator registry for workload graph lowering.

Every operator type referenced in a ``WorkloadGraphV1`` must be registered
here with one of three dispositions:

``modeled``
    The operator has a known cost model (cycle, bandwidth, footprint).
    Examples: ``gemm``, ``softmax``, ``layernorm``, ``pointwise_conv``.

``explicitly_free_or_fused``
    The operator is free (0 cycle) ONLY because it is fused into another
    operator.  Must carry ``fused_into`` (target node or op) and provenance.
    Examples: ``reshape`` fused into preceding ``gemm``, ``gelu`` fused as
    activation function of ``gemm``.

``unsupported``
    The operator is explicitly not supported.  Import and execution MUST fail
    when encountering these.  Examples: unknown ONNX custom ops, ops that
    have been explicitly excluded.

The default disposition for any unregistered operator is ``unsupported`` —
there is no ``profile_required`` or ``unknown`` fallback.  An unregistered
op in the graph triggers ``UnsupportedOperatorError``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional

from contracts.errors import UnsupportedOperatorError
from workloads.schema import WorkloadProvenance


class OperatorDisposition(str, Enum):
    """What the simulator knows about this operator type."""

    MODELED = "modeled"
    """The operator has a validated cost model."""

    EXPLICITLY_FREE_OR_FUSED = "explicitly_free_or_fused"
    """The operator is 0-cycle only because it is fused (must carry fused_into)."""

    UNSUPPORTED = "unsupported"
    """The operator is explicitly not supported — fail on encounter."""


@dataclass(frozen=True)
class OperatorEntry:
    """A single registry entry for an operator type.

    For ``EXPLICITLY_FREE_OR_FUSED`` ops, ``fused_into`` and ``provenance``
    are required.  An op without these fields defaults to ``UNSUPPORTED``
    if the disposition is set that way, but it is recommended to supply them
    for auditability.
    """

    op_type: str
    """Operator type key (e.g. ``gemm``, ``softmax``)."""

    disposition: OperatorDisposition
    """How the simulator treats this operator."""

    description: str = ""
    """Human-readable description."""

    fused_into: Optional[str] = None
    """For ``EXPLICITLY_FREE_OR_FUSED``: what operation this is fused into."""

    provenance: Optional[WorkloadProvenance] = None
    """Source of the registration decision."""

    def __post_init__(self):
        if not self.op_type or not self.op_type.strip():
            raise ValueError("op_type must not be empty")
        if self.disposition == OperatorDisposition.EXPLICITLY_FREE_OR_FUSED:
            if not self.fused_into:
                raise ValueError(
                    f"free/fused op {self.op_type!r} must record fused_into"
                )


# ── Built-in registry ────────────────────────────────────────────────────────

# Modeled ops: compute-bound operations with known cost models
_MODELED_OPS: Dict[str, OperatorEntry] = {
    "gemm": OperatorEntry(
        op_type="gemm",
        disposition=OperatorDisposition.MODELED,
        description="General matrix multiply (M×K×N)",
    ),
    "pointwise_conv": OperatorEntry(
        op_type="pointwise_conv",
        disposition=OperatorDisposition.MODELED,
        description="1×1 convolution (im2col → GEMM)",
    ),
    "depthwise_conv": OperatorEntry(
        op_type="depthwise_conv",
        disposition=OperatorDisposition.MODELED,
        description="Per-channel depthwise convolution",
    ),
    "softmax": OperatorEntry(
        op_type="softmax",
        disposition=OperatorDisposition.MODELED,
        description="Softmax activation (SFU)",
    ),
    "layernorm": OperatorEntry(
        op_type="layernorm",
        disposition=OperatorDisposition.MODELED,
        description="Layer normalization",
    ),
    "rms_norm": OperatorEntry(
        op_type="rms_norm",
        disposition=OperatorDisposition.MODELED,
        description="RMS normalization",
    ),
    "gelu": OperatorEntry(
        op_type="gelu",
        disposition=OperatorDisposition.MODELED,
        description="GELU activation (SFU)",
    ),
    "relu": OperatorEntry(
        op_type="relu",
        disposition=OperatorDisposition.MODELED,
        description="ReLU activation (SFU/elementwise)",
    ),
    "hard_swish": OperatorEntry(
        op_type="hard_swish",
        disposition=OperatorDisposition.MODELED,
        description="HardSwish activation (SFU)",
    ),
    "hard_sigmoid": OperatorEntry(
        op_type="hard_sigmoid",
        disposition=OperatorDisposition.MODELED,
        description="HardSigmoid activation (SFU)",
    ),
    "global_avg_pool": OperatorEntry(
        op_type="global_avg_pool",
        disposition=OperatorDisposition.MODELED,
        description="Global average pooling (SFU)",
    ),
    "max_pool": OperatorEntry(
        op_type="max_pool",
        disposition=OperatorDisposition.MODELED,
        description="Max pooling",
    ),
    "add": OperatorEntry(
        op_type="add",
        disposition=OperatorDisposition.MODELED,
        description="Element-wise addition (vector/SIMD)",
    ),
    "mul": OperatorEntry(
        op_type="mul",
        disposition=OperatorDisposition.MODELED,
        description="Element-wise multiplication (vector/SIMD)",
    ),
    "matmul": OperatorEntry(
        op_type="matmul",
        disposition=OperatorDisposition.MODELED,
        description="Generic matrix multiply (alias for gemm)",
    ),
    "conv": OperatorEntry(
        op_type="conv",
        disposition=OperatorDisposition.MODELED,
        description="General convolution (im2col → GEMM)",
    ),
}

# Explicitly free or fused ops: zero-cycle only when fused into another op
_FREE_FUSED_OPS: Dict[str, OperatorEntry] = {
    "reshape": OperatorEntry(
        op_type="reshape",
        disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
        description="Reshape/view operation — zero cost when fused into preceding op",
        fused_into="preceding_compute_op",
    ),
    "reduce_mean": OperatorEntry(
        op_type="reduce_mean",
        disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
        description="Mean reduction — metadata only, fused into preceding op",
        fused_into="preceding_compute_op",
    ),
    "shape": OperatorEntry(
        op_type="shape",
        disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
        description="Shape inference — metadata only, fused into preceding op",
        fused_into="preceding_compute_op",
    ),
    "concat": OperatorEntry(
        op_type="concat",
        disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
        description="Tensor concatenation — zero cost when fused into preceding buffer manipulation",
        fused_into="preceding_buffer_op",
    ),
    "transpose": OperatorEntry(
        op_type="transpose",
        disposition=OperatorDisposition.EXPLICITLY_FREE_OR_FUSED,
        description="Transpose — metadata rewrite, zero cost when fused",
        fused_into="preceding_compute_op",
    ),
}

# Explicitly unsupported ops — must fail-closed
_UNSUPPORTED_OPS: Dict[str, OperatorEntry] = {
    "gather": OperatorEntry(
        op_type="gather",
        disposition=OperatorDisposition.UNSUPPORTED,
        description="Scatter/gather operations not in scope for arc model",
    ),
    "scatter": OperatorEntry(
        op_type="scatter",
        disposition=OperatorDisposition.UNSUPPORTED,
        description="Scatter/gather operations not in scope for arc model",
    ),
    "instance_norm": OperatorEntry(
        op_type="instance_norm",
        disposition=OperatorDisposition.UNSUPPORTED,
        description="Instance normalization not yet cost-modeled",
    ),
    "group_norm": OperatorEntry(
        op_type="group_norm",
        disposition=OperatorDisposition.UNSUPPORTED,
        description="Group normalization not yet cost-modeled",
    ),
    "batch_norm": OperatorEntry(
        op_type="batch_norm",
        disposition=OperatorDisposition.UNSUPPORTED,
        description="Batch normalization not yet cost-modeled",
    ),
    "upsample": OperatorEntry(
        op_type="upsample",
        disposition=OperatorDisposition.UNSUPPORTED,
        description="Upsampling not yet cost-modeled",
    ),
}


# ── Master registry ──────────────────────────────────────────────────────────


class OperatorRegistry:
    """Fail-closed operator registry.

    Queries that return ``EXPLICITLY_FREE_OR_FUSED`` or ``UNSUPPORTED``
    require the caller to explicitly handle those cases.  The default
    disposition for any unregistered operator is ``UNSUPPORTED``.
    """

    def __init__(self):
        self._entries: Dict[str, OperatorEntry] = {}
        self._entries.update(_MODELED_OPS)
        self._entries.update(_FREE_FUSED_OPS)
        self._entries.update(_UNSUPPORTED_OPS)

    def lookup(self, op_type: str) -> OperatorEntry:
        """Look up an operator type by key.

        Returns the registered ``OperatorEntry``, or a synthetic
        ``UNSUPPORTED`` entry for unregistered ops.

        Raises:
            UnsupportedOperatorError: if the operator is explicitly unsupported
                or not registered (caller should not proceed to execution).
        """
        entry = self._entries.get(op_type)
        if entry is None:
            # Unregistered: synthetic unsupported
            entry = OperatorEntry(
                op_type=op_type,
                disposition=OperatorDisposition.UNSUPPORTED,
                description=f"unregistered operator: {op_type}",
            )

        if entry.disposition == OperatorDisposition.UNSUPPORTED:
            raise UnsupportedOperatorError(
                f"operator {op_type!r} is not supported",
                op_type=op_type,
            )
        return entry

    def lookup_or_none(self, op_type: str) -> Optional[OperatorEntry]:
        """Look up without raising — returns None for unregistered ops.

        This is used for introspection; callers that plan to execute should
        use ``lookup()`` instead.
        """
        return self._entries.get(op_type)

    def check(self, op_type: str) -> OperatorDisposition:
        """Return the disposition for an operator type.

        Unregistered ops return ``UNSUPPORTED``.
        """
        entry = self._entries.get(op_type)
        if entry is None:
            return OperatorDisposition.UNSUPPORTED
        return entry.disposition

    def is_modeled(self, op_type: str) -> bool:
        """Return True if the operator is registered as modeled."""
        return self.check(op_type) == OperatorDisposition.MODELED

    def is_free_or_fused(self, op_type: str) -> bool:
        """Return True if the operator is registered as explicitly free/fused."""
        return self.check(op_type) == OperatorDisposition.EXPLICITLY_FREE_OR_FUSED

    def is_unsupported(self, op_type: str) -> bool:
        """Return True if the operator is unsupported or unregistered."""
        return self.check(op_type) == OperatorDisposition.UNSUPPORTED

    @property
    def modeled_ops(self) -> FrozenSet[str]:
        """Frozen set of modeled operator type keys."""
        return frozenset(
            k for k, v in self._entries.items()
            if v.disposition == OperatorDisposition.MODELED
        )

    @property
    def free_fused_ops(self) -> FrozenSet[str]:
        """Frozen set of explicitly free/fused operator type keys."""
        return frozenset(
            k for k, v in self._entries.items()
            if v.disposition == OperatorDisposition.EXPLICITLY_FREE_OR_FUSED
        )

    @property
    def unsupported_ops(self) -> FrozenSet[str]:
        """Frozen set of explicitly unsupported operator type keys."""
        return frozenset(
            k for k, v in self._entries.items()
            if v.disposition == OperatorDisposition.UNSUPPORTED
        )

    def register(self, entry: OperatorEntry) -> None:
        """Register or override an operator entry.

        Use this to add custom modeled ops or to mark ops as unsupported.
        """
        if not entry.op_type or not entry.op_type.strip():
            raise ValueError("op_type must not be empty")
        self._entries[entry.op_type] = entry

    def __contains__(self, op_type: str) -> bool:
        return op_type in self._entries

    def __len__(self) -> int:
        return len(self._entries)


# ── Singleton ────────────────────────────────────────────────────────────────

DEFAULT_REGISTRY = OperatorRegistry()
"""Default operator registry including all built-in entries.

All lowering adapters and executors should use this singleton unless
a custom registry is needed for testing.
"""
