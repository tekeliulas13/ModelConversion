#include "inference.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];

    // Define model paths
    const std::string json_path = "/home/syntonym/workspace_ulas/ModelConversion/tvm_export/resnet50_flowers102.json";
    const std::string params_path = "/home/syntonym/workspace_ulas/ModelConversion/tvm_export/resnet50_flowers102.params";
    const std::string so_path = "/home/syntonym/workspace_ulas/ModelConversion/tvm_export/resnet50_flowers102.so";

    try {
        TVMInference infer(json_path, params_path, so_path);

        // Warmup: run inference 10 times (not timed)
        for (int i = 0; i < 10; ++i) {
            infer.runInference(image_path);
        }

        // Timed inference: run 100 times
        const int num_runs = 100;
        auto start = std::chrono::high_resolution_clock::now();
        int predicted_index = -1;
        for (int i = 0; i < num_runs; ++i) {
            predicted_index = infer.runInference(image_path);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        double avg_time = duration.count() / num_runs;

        std::cout << "Average inference time over " << num_runs << " runs: " << avg_time << " ms" << std::endl;

        if (predicted_index != -1) {
            std::cout << "Predicted class index: " << predicted_index << std::endl;

            std::ofstream result_file("prediction_result.txt");
            if (result_file.is_open()) {
                result_file << "The predicted class index is: " << predicted_index << std::endl;
                result_file << "Average inference time over " << num_runs << " runs: " << avg_time << " ms" << std::endl;
                result_file.close();
                std::cout << "Prediction saved to prediction_result.txt" << std::endl;
            } else {
                std::cerr << "Unable to open file to save prediction." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}