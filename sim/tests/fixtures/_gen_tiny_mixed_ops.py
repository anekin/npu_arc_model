"""Generate sim/tests/fixtures/tiny_mixed_ops.onnx programmatically.

Model topology: input[batch,3,8,8] -> Conv -> Add(bias) -> Relu -> output[batch,4,6,6]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def build_model() -> onnx.ModelProto:
    # Inputs
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, ["batch", 3, 8, 8])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, ["batch", 4, 6, 6])

    # Weights: Conv 3x3, 3 -> 4 channels
    weight = np.random.randn(4, 3, 3, 3).astype(np.float32) * 0.1
    weight_init = numpy_helper.from_array(weight, name="conv_weight")

    # Bias for Add: 4 channels broadcast over H/W
    bias = np.random.randn(4, 1, 1).astype(np.float32) * 0.1
    bias_init = numpy_helper.from_array(bias, name="bias")

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "conv_weight"],
            outputs=["conv_out"],
            name="conv_node",
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node(
            "Add",
            inputs=["conv_out", "bias"],
            outputs=["add_out"],
            name="add_node",
        ),
        helper.make_node(
            "Relu",
            inputs=["add_out"],
            outputs=["output"],
            name="relu_node",
        ),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="tiny_mixed_ops",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_init, bias_init],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model


def main() -> None:
    out_path = Path(__file__).with_name("tiny_mixed_ops.onnx")
    model = build_model()
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
