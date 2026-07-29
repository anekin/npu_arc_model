"""Arc Model — versioned declarative workload graph and operator contracts.

Public API surface:
    schema      — ``WorkloadGraphV1``, ``TensorSpec``, ``NodeSpec``, ``SymbolicDim``, ``WorkloadProvenance``
    dimensions  — ``DimensionBindings`` and canonical dimension fields
    operators   — ``OperatorRegistry``, ``OperatorEntry``, ``OperatorDisposition``
    validate    — pre-lowering validation gates
    json_adapter    — canonical serialization / deserialization
    onnx_adapter    — ONNX → ``WorkloadGraphV1`` lowering
    legacy_adapter  — legacy tuple/dict traces → ``WorkloadGraphV1`` lowering
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
    apply_bindings,
)
from workloads.json_adapter import (  # noqa: F401
    graph_digest,
    graph_to_bytes,
    graph_to_dict,
    graph_to_json,
    json_to_graph,
)
from workloads.legacy_adapter import (  # noqa: F401
    apply_legacy_batch_m,
    lower_cv_dict_trace,
    lower_llm_tuple_trace,
)
from workloads.onnx_adapter import (  # noqa: F401
    lower_onnx_model_to_graph,
    lower_onnx_to_graph,
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
