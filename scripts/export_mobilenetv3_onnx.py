#!/usr/bin/env python3
"""Export MobileNetV3-Small from torchvision to ONNX with opset>=14."""
import torch
import torchvision
import onnx
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "mobilenetv3_small.onnx")
EVIDENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "evidence", "cv-task-1-onnx.txt")

def main():
    # Load pretrained model, eval mode
    model = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    model.eval()

    # BN folding: torch.quantization.fuse_modules is not needed — torch.onnx.export
    # handles BN folding via torch.jit.script/trace or the built-in optimizer passes.
    # We use torch.onnx.export with do_constant_folding=True (default).

    dummy_input = torch.randn(1, 3, 224, 224)
    opset_version = 17  # >=14, 17 supports native HardSwish/HardSigmoid

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_PATH,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Validate
    onnx_model = onnx.load(MODEL_PATH)
    onnx.checker.check_model(onnx_model)

    # Collect ops info
    nodes = list(onnx_model.graph.node)
    op_counts = {}
    for n in nodes:
        op = n.op_type
        op_counts[op] = op_counts.get(op, 0) + 1

    lines = []
    lines.append("=" * 60)
    lines.append("CV Task 1 — MobileNetV3-Small ONNX Export")
    lines.append("=" * 60)
    lines.append(f"PyTorch:   {torch.__version__}")
    lines.append(f"TorchVision: {torchvision.__version__}")
    lines.append(f"ONNX:      {onnx.__version__}")
    lines.append(f"Opset:     {opset_version}")
    lines.append(f"Model:     MobileNetV3-Small (pretrained)")
    lines.append(f"Input:     (1, 3, 224, 224)")
    lines.append(f"Output:    {MODEL_PATH}")
    lines.append(f"ONNX check: PASSED")
    lines.append(f"Total nodes: {len(nodes)}")
    lines.append("")
    lines.append("Operator list (sorted by count):")
    for op, cnt in sorted(op_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {op}: {cnt}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    # Save evidence
    os.makedirs(os.path.dirname(EVIDENCE_PATH), exist_ok=True)
    with open(EVIDENCE_PATH, "w") as f:
        f.write(report)

    print(f"\nEvidence saved to {EVIDENCE_PATH}")


if __name__ == "__main__":
    main()
