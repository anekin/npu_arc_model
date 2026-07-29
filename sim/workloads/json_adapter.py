"""Canonical JSON serialization and deterministic digest for WorkloadGraphV1.

Provides:
- ``graph_to_json`` / ``graph_to_bytes``: canonical, sorted-key, stable
  float/enum representation of a ``WorkloadGraphV1``.
- ``graph_digest``: deterministic SHA-256 digest of the canonical JSON.
- ``json_to_graph``: deserialize canonical JSON back into a validated graph.

The canonical form is suitable for:
- Stable graph instance IDs (digest changes only when the graph changes).
- Round-trip regression tests.
- Equivalence comparisons between lowering adapters (JSON, ONNX, legacy).
"""

from __future__ import annotations

from contracts.identity import canonical_json_bytes, digest_sha256
from workloads.schema import WorkloadGraphV1


def graph_to_dict(graph: WorkloadGraphV1) -> dict:
    """Convert a workload graph to a plain, JSON-serializable dict.

    Uses Pydantic's ``mode='json'`` so enums become string values and no
    Python-only objects leak into the dict.
    """
    return graph.model_dump(mode="json")


def graph_to_bytes(graph: WorkloadGraphV1, *, indent: int | str | None = None) -> bytes:
    """Return canonical JSON bytes for *graph*.

    Guarantees:
    - Sorted keys at every nesting level.
    - Stable ``repr`` for floats.
    - Enums serialized as their string value.
    - ``ensure_ascii=True`` for cross-platform stability.
    """
    return canonical_json_bytes(graph_to_dict(graph), indent=indent)


def graph_to_json(graph: WorkloadGraphV1, *, indent: int | str | None = None) -> str:
    """Return canonical JSON string for *graph*."""
    return graph_to_bytes(graph, indent=indent).decode("utf-8")


def graph_digest(graph: WorkloadGraphV1) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of *graph*."""
    return digest_sha256(graph_to_dict(graph))


def json_to_graph(json_str: str | bytes) -> WorkloadGraphV1:
    """Deserialize canonical JSON back into a validated ``WorkloadGraphV1``.

    Raises:
        ValueError / pydantic.ValidationError: if the JSON is malformed or
            violates the workload graph schema.
    """
    if isinstance(json_str, bytes):
        json_str = json_str.decode("utf-8")
    return WorkloadGraphV1.model_validate_json(json_str)
