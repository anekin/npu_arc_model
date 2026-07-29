"""Arc Model — versioned declarative workload graph and operator contracts.

Public API surface:
    schema      — ``WorkloadGraphV1``, ``TensorSpec``, ``NodeSpec``, ``SymbolicDim``, ``WorkloadProvenance``
    dimensions  — ``DimensionBindings`` and canonical dimension fields
    operators   — ``OperatorRegistry``, ``OperatorEntry``, ``OperatorDisposition``
    validate    — pre-lowering validation gates
"""

from workloads.dimensions import (  # noqa: F401
    AXIS_ACTION_HORIZON,
    AXIS_BATCH,
    AXIS_FLOW_STEPS,
    AXIS_IMAGE_COUNT,
    AXIS_INFLIGHT_JOBS,
    AXIS_RESIDENT_MODELS,
    AXIS_SEQUENCES,
    AXIS_TOKEN_BLOCK,
    DimensionBindings,
)
from workloads.operators import (  # noqa: F401
    DEFAULT_REGISTRY,
    OperatorDisposition,
    OperatorEntry,
    OperatorRegistry,
)
from workloads.schema import (  # noqa: F401
    Layout,
    NodeSpec,
    Precision,
    ShapeElement,
    SymbolicDim,
    TensorSpec,
    WorkloadGraphV1,
    WorkloadProvenance,
)
from workloads.validate import (  # noqa: F401
    validate_all,
    validate_dimensions,
    validate_graph_dag,
    validate_operators,
    validate_tensor_lifetime,
)
