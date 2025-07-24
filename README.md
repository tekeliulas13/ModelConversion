# ModelConversion

**This project demonstrates a complete, end-to-end workflow for converting a PyTorch computer vision model for high-performance deployment and running inference in a C++ application using NVIDIA TensorRT.**

**The example uses a ResNet50 model trained on the Flowers102 dataset, but you can use any compatible PyTorch model.**

---

## 🚀 Project Workflow

The project follows these distinct steps:

1. **ONNX Export**: A Python script (`tensorrt_example.py`) loads trained PyTorch weights and exports the model to the universal ONNX (`.onnx`) format.
2. **TensorRT Engine Build**: The C++ application loads the ONNX file and uses TensorRT to build a highly optimized inference engine, which it caches to disk (`.engine`) for subsequent runs.
3. **C++ Inference**: The final C++ executable (`tensorrt_infer`) performs fast inference on a user-provided image using the optimized TensorRT engine.

---

## 📋 Requirements

### Python Environment

- Python 3.8+
- Dependencies are listed in `requirements.txt`. Install them with:
    ```bash
    pip install -r requirements.txt
    ```
    - `torch` and `torchvision`
    - `numpy`
    - `opencv-python`
    - `onnx`

### C++ Environment

- C++17 compliant compiler (e.g., GCC, Clang)
- CMake (version 3.14 or higher)
- NVIDIA CUDA Toolkit
- NVIDIA TensorRT [cite: 3]
- OpenCV [cite: 5]

---

## 🛠️ How to Use

Follow these steps to run the pipeline from ONNX export to C++ inference.

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd ModelConversion
```

### Step 2: Set Up Python

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 3: Convert the Model to ONNX

Run the export script. This will load your trained `.pth` file and export the model to an `.onnx` file:

```bash
python tensorrt_example.py
```

> **Note:** You may need to update the hardcoded file paths in `tensorrt_example.py` and `main.cpp` to match your directory structure.

### Step 4: Build the C++ Inference Application

Use CMake to build the C++ executable:

```bash
# Create a build directory
mkdir build && cd build

# Configure the project with CMake.
# You MUST point this to your TensorRT installation directory. [cite: 3]
cmake .. -DTENSORRT_DIR=/path/to/your/tensorrt

# Build the project
cmake --build .
```

This will create an executable named `tensorrt_infer` inside the `build` directory.

### Step 5: Run C++ Inference

Execute the compiled program, providing a path to an image as an argument:

```bash
./tensorrt_infer /path/to/your/flower_image.jpg
```

The program will:

- Load the ONNX model and build/cache the TensorRT engine on the first run.
- Perform a warmup and measure the average inference time.
- Print the predicted class index to the console.
- Save the result to `prediction_result_trt.txt`.
