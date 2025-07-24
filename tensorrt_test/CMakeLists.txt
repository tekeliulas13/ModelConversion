cmake_minimum_required(VERSION 3.14)
project(TensorRT_Inference_Example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- Find Dependencies ---
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# --- Configure TensorRT ---
# Set this to your TensorRT installation directory.
# You can override this from the command line: cmake .. -DTENSORRT_DIR=/path/to/tensorrt
set(TENSORRT_DIR "/usr/src/tensorrt" CACHE PATH "Path to the TensorRT installation directory")
message(STATUS "Using TensorRT from: ${TENSORRT_DIR}")

# --- Create Executable ---
# **CORRECTION**: Added main.cpp to the executable source files.
add_executable(tensorrt_infer main.cpp inference.cpp)

# --- Link Dependencies to the Target ---
# This is a more modern way to manage dependencies per target.

# Link OpenCV
target_include_directories(tensorrt_infer PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tensorrt_infer PRIVATE ${OpenCV_LIBS})

# Link TensorRT
target_include_directories(tensorrt_infer PRIVATE ${TENSORRT_DIR}/include)
target_link_directories(tensorrt_infer PRIVATE ${TENSORRT_DIR}/lib)
target_link_libraries(tensorrt_infer PRIVATE nvinfer nvonnxparser)

# Link CUDA (already handled by project(LANGUAGES CUDA))
target_link_libraries(tensorrt_infer PRIVATE ${CUDA_LIBRARIES})

# --- Final Message ---
message(STATUS "Project configured. Build with 'cmake --build .'")