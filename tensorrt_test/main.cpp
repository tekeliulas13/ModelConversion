#include "inference.h"
#include <iostream>
#include <fstream>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return 1;
    }

    // --- Configuration for your specific model ---
    const std::string onnx_model_path = "/home/syntonym/workspace_ulas/ModelConversion/tensorrt_export/resnet50_flowers102.onnx";
    const int input_width = 224;
    const int input_height = 224;
    const int input_channels = 3;
    const int num_classes = 102; // For Flowers102 dataset

    // --- Initialize the inference engine ---
    TensorRTInference infer(onnx_model_path, input_width, input_height, input_channels, num_classes);

    // --- Warmup ---
    auto warmup_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 10; i++)
    {
        infer.runInference(argv[1]);
    }
    auto warmup_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> warmup_time = warmup_end - warmup_start;
    std::cout << "Warmup time: " << warmup_time.count() << " ms" << std::endl;

    // --- Measure inference time over 100 iterations ---
    const int num_iterations = 1000;
    double total_inference_time = 0.0;
    int result = -1;
    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        result = infer.runInference(argv[1]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> inference_time = end - start;
        total_inference_time += inference_time.count();
    }
    double avg_inference_time = total_inference_time / num_iterations;
    std::cout << "Average inference time over " << num_iterations << " iterations: " 
              << avg_inference_time << " ms" << std::endl;

    // --- Process and save the result ---
    if (result != -1) {
        std::cout << "Predicted Class Index: " << result << std::endl;
        std::ofstream result_file("prediction_result_trt.txt");
        if (result_file.is_open()) {
            result_file << "Predicted class index: " << result << std::endl;
            result_file.close();
            std::cout << "Result saved to prediction_result_trt.txt" << std::endl;
        } else {
            std::cerr << "Could not open result file for writing." << std::endl;
        }
    }

    return 0;
}