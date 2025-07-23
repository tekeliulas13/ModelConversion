import torch
import torchvision

# 1. Load and modify model
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load("/home/syntonym/workspace_ulas/ModelConversion/models/best_resnet50_flowers102.pth"))
model.eval()

# 2. Trace with dummy input
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "/home/syntonym/workspace_ulas/ModelConversion/tensorrt_export/resnet50_flowers102.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("✅ Exported model to ONNX: resnet50_flowers102.onnx")
