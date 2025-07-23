import torch
import torchvision
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# Load ResNet50 and modify final layer for 102 classes
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load("/home/syntonym/workspace_ulas/ModelConversion/models/best_resnet50_flowers102.pth"))
model.eval()

input_shape = [1, 3, 224, 224]
input_name = "input0"
# Trace model for TorchScript
scripted_model = torch.jit.trace(model, torch.randn(input_shape)).eval()
# Convert to TVM Relay
mod, params = relay.frontend.from_pytorch(scripted_model, [(input_name, input_shape)])

target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
# Compile with TVM
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Create TVM runtime module
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input(input_name, tvm.nd.array(torch.randn(input_shape).numpy()))
module.run()
output = module.get_output(0)
print("Output shape:", output.shape)

# Export compiled module
export_prefix = "./tvm_export/resnet50_flowers102"
lib.export_library(export_prefix + ".so")
with open(export_prefix + ".json", "w") as f_graph:
    f_graph.write(lib.get_graph_json())
with open(export_prefix + ".params", "wb") as f_params:
    f_params.write(relay.save_param_dict(lib.get_params()))
print("✅ TVM module saved.")
