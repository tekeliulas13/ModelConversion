#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <opencv2/opencv.hpp>

class TVMInference {
public:
    TVMInference(const std::string& json_path, const std::string& params_path, const std::string& so_path);
    int runInference(const std::string& image_path);

private:
    cv::Mat preprocess(const cv::Mat& img);
    int getTopPrediction(tvm::runtime::NDArray& output);

    // TVM runtime module and functions
    tvm::runtime::Module graph_executor_module_;
    tvm::runtime::PackedFunc set_input_;
    tvm::runtime::PackedFunc get_output_;
    tvm::runtime::PackedFunc run_;

    // Device context
    DLDevice dev_{kDLCPU, 0};

    // Model specific parameters
    const int input_width_ = 224;
    const int input_height_ = 224;
    const int num_classes_ = 102; // For Flowers102 dataset
};

#endif // INFERENCE_H