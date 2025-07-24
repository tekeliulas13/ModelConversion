#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include "NvInfer.h"

// Logger for TensorRT info, warnings, and errors
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

class TensorRTInference {
public:
    // Constructor now accepts model parameters for genericity
    TensorRTInference(const std::string& onnxModelPath, int inputWidth, int inputHeight, int inputChannels, int numClasses);
    ~TensorRTInference();

    int runInference(const std::string& imagePath);

private:
    void buildEngine();
    cv::Mat preprocessImage(const std::string& imagePath);
    void setup();

    std::string onnx_model_path_;
    std::string engine_file_path_;
    
    Logger logger_;
    // Use smart pointers for automatic memory management
    std::unique_ptr<nvinfer1::IRuntime> runtime_{nullptr};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{nullptr};

    void* device_buffers_[2]; // 0 for input, 1 for output
    cudaStream_t stream_;

    // Model specific parameters are now member variables initialized by the constructor
    int input_width_;
    int input_height_;
    int input_channels_;
    int num_classes_;
};

#endif // INFERENCE_H