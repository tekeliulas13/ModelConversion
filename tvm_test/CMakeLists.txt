cmake_minimum_required(VERSION 3.10)
project(TVMInferenceExample)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV ---
find_package(OpenCV REQUIRED)

# --- Specify TVM include and library paths ---
set(TVM_DIR "/home/syntonym/workspace_ulas/libs/tvm/build")

# Add all necessary include directories for TVM and its dependencies
include_directories(
    ${TVM_DIR}/../include
    ${TVM_DIR}/../3rdparty/dmlc-core/include
    ${TVM_DIR}/../3rdparty/dlpack/include
    ${TVM_DIR}/../3rdparty/vta/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${TVM_DIR})

# Add the executable with all necessary source files
add_executable(tvm_infer main.cpp inference.cpp)

# Link libraries
target_link_libraries(tvm_infer
    ${OpenCV_LIBS}
    tvm_runtime
)