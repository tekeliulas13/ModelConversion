import os
from ultralytics import YOLO
import tensorrt as trt
import torch
import torchvision.models as models
from yolov7.models.experimental import attempt_load

# === CONFIGURATION ===
yolo = "yolo8"
if yolo == "yolo8":
    model_path = "models/yolov8n-face.pt"    # Your model file
    img_size = (640, 640)        # Input size (H, W)
    onnx_output = "models/yolov8n-face.onnx" # Name for the intermediate ONNX file
    trt_output = "models/yolov8n-face.engine"   # Name for the final TensorRT engine
    dynamic = False              # Set to True for dynamic batch size/input size
    workspace_mb = 8192          # Max workspace size in megabytes
    model = YOLO(model_path).cuda()
    model.eval()
elif yolo == "yolov7":
    model_path = "models/yolov7-face.pt"    # Your model file
    img_size = (1920, 1088)        # Input size (H, W)
    onnx_output = "models/yolov7-face.onnx" # Name for the intermediate ONNX file
    trt_output = "models/yolov7-face.engine"   # Name for the final TensorRT engine
    dynamic = False              # Set to True for dynamic batch size/input size
    workspace_mb = 8192          # Max workspace size in megabytes
    
    # model = models.resnet18()
    # state_dict = torch.load(model_path, map_location="cuda")
    # model.load_state_dict(state_dict)
    # model = model.cuda()
    # model.eval()
    from models.yolo import Model
    # 1. Load checkpoint
    ckpt = torch.load(
        "/home/syntonym/workspace_ulas/ModelConversion/models/yolov7-face.pt",
        map_location=torch.device("cpu"),
    )
    model_obj = ckpt.get("model", None)
    if isinstance(model_obj, Model):
        model = model_obj
    else:
        state_dict = model_obj if model_obj is not None else ckpt
        model = Model(cfg="/home/syntonym/workspace_ulas/ModelConversion/yolov7-face/cfg/yolov7-face.yaml")
        model.load_state_dict(state_dict)

    torch.save(model.state_dict(), "models/yolov7-face.pth")
    model = model.float()
    model.eval()
else:
    model_path = "models/resnet18-f37072fd.pth"    # Your model file
    img_size = (224, 224)        # Input size (H, W)
    onnx_output = "models/resnet18.onnx" # Name for the intermediate ONNX file
    trt_output = "models/resnet18.engine"   # Name for the final TensorRT engine
    dynamic = False              # Set to True for dynamic batch size/input size
    workspace_mb = 8192          # Max workspace size in megabytes
    model = models.resnet18()
    state_dict = torch.load(model_path, map_location="cuda")
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    

# === STEP 1: Export to ONNX ===
print(f"🚀 [1/3] Exporting '{model_path}' to ONNX...")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

if yolo == "yolo8":
    model.export(
        format="onnx",
        dynamic=dynamic,
        opset=12,
        imgsz=img_size,
        simplify=True,
        verbose=False  # Set to True for more detailed export logs
    )
else:
    # Use torch.onnx.export() for standard PyTorch models
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device='cuda')
    torch.onnx.export(
        model,                          # The model to export
        dummy_input,                    # A sample input tensor for tracing
        onnx_output,                    # Where to save the ONNX file
        export_params=True,             # Store the trained weights in the model file
        opset_version=12,               # The ONNX version to use
        do_constant_folding=True,       # For optimization
        input_names=['input'],          # Define a name for the model's input
        output_names=['output'],        # Define a name for the model's output
        dynamic_axes={'input' : {0 : 'batch_size'},    # Optional: for dynamic batch size
                    'output' : {0 : 'batch_size'}}
                    if dynamic else None
    )

assert os.path.exists(onnx_output), "ONNX export failed."
print(f"✅ ONNX export successful: '{onnx_output}'")

# === STEP 2: Define the TensorRT Build Function ===
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path, input_shape=(1, 3, 640, 640), is_dynamic=False):
    """Builds a TensorRT engine from an ONNX file."""
    if not os.path.exists(onnx_file_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_file_path}")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"🔎 [2/3] Parsing ONNX file: '{onnx_file_path}'")
    with open(onnx_file_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("❌ ONNX parsing failed.")
            for error in range(parser.num_errors):
                print(f"[ERROR] {parser.get_error(error)}")
            return None

    config = builder.create_builder_config()
    # Set memory pool limit (replaces max_workspace_size in newer TensorRT)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    if is_dynamic:
        print("    Building with dynamic input shapes.")
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        # Define min, optimal, and max shapes for the dynamic input
        min_shape = (1, 3, 320, 320)
        opt_shape = input_shape
        max_shape = (1, 3, 1280, 1280)
        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)
    else:
        print(f"    Building with static input shape: {input_shape}")
        network.get_input(0).shape = input_shape

    print(f"⚙️ [3/3] Building TensorRT engine. This may take a few minutes...")

    # --- THE FIX IS HERE ---
    # `build_engine` is deprecated. Use `build_serialized_network` instead.
    # It directly returns the serialized engine ready for saving.
    serialized_engine = builder.build_serialized_network(network, config)
    # -----------------------

    if serialized_engine is None:
        raise RuntimeError("❌ TensorRT engine build failed.")

    print(f"💾 Saving engine to '{engine_file_path}'")
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    print(f"✅ Engine built and saved successfully!")
    return serialized_engine

# === STEP 3: Build the Engine ===
build_engine(
    onnx_output,
    trt_output,
    input_shape=(1, 3, img_size[0], img_size[1]),
    is_dynamic=dynamic
)