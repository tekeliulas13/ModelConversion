#include "inference.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include <numeric> // Required for std::distance
#include <algorithm> // Required for std::max_element

// Logger implementation remains the same
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}

// Constructor now initializes model parameters from arguments
TensorRTInference::TensorRTInference(const std::string& onnxModelPath, int inputWidth, int inputHeight, int inputChannels, int numClasses)
    : onnx_model_path_(onnxModelPath),
      input_width_(inputWidth),
      input_height_(inputHeight),
      input_channels_(inputChannels),
      num_classes_(numClasses) {
    engine_file_path_ = onnx_model_path_ + ".engine";
    setup();
}

// Destructor is simple thanks to smart pointers
TensorRTInference::~TensorRTInference() {
    cudaStreamDestroy(stream_);
    cudaFree(device_buffers_[0]);
    cudaFree(device_buffers_[1]);
}

void TensorRTInference::setup() {
    std::ifstream engine_file(engine_file_path_, std::ios::binary);
    if (engine_file.good()) {
        std::cout << "Loading engine from cache: " << engine_file_path_ << std::endl;
        engine_file.seekg(0, engine_file.end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, engine_file.beg);
        std::unique_ptr<char[]> engine_data(new char[engine_size]);
        engine_file.read(engine_data.get(), engine_size);

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.get(), engine_size));
    } else {
        std::cout << "Building engine from ONNX file: " << onnx_model_path_ << std::endl;
        buildEngine();
    }

    if (!engine_) {
        std::cerr << "Failed to create TensorRT engine." << std::endl;
        return;
    }
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context." << std::endl;
        return;
    }
    // Use member variables for buffer allocation
    const int input_size = input_channels_ * input_height_ * input_width_ * sizeof(float);
    const int output_size = num_classes_ * sizeof(float);
    
    cudaMalloc(&device_buffers_[0], input_size);
    cudaMalloc(&device_buffers_[1], output_size);

    cudaStreamCreate(&stream_);
}

void TensorRTInference::buildEngine() {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) return;

    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
    if (!network) return;

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) return;

    if (!parser->parseFromFile(onnx_model_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) return;

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB

    // --- START: Optimization Profile using member variables ---
    auto profile = builder->createOptimizationProfile();
    const char* input_name = network->getInput(0)->getName(); 

    // Use member variables to define dimension ranges for the profile
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, input_channels_, input_height_, input_width_});
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, input_channels_, input_height_, input_width_});
    profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, input_channels_, input_height_, input_width_});
    
    config->addOptimizationProfile(profile);
    // --- END: Optimization Profile ---

    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build serialized network." << std::endl;
        return;
    }

    std::ofstream engine_file(engine_file_path_, std::ios::binary);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
}

// Preprocessing now uses member variables for image dimensions
cv::Mat TensorRTInference::preprocessImage(const std::string& imagePath) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not read image " << imagePath << std::endl;
        return cv::Mat();
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(input_width_, input_height_));
    cv::Mat f_img;
    resized.convertTo(f_img, CV_32FC3, 1.0 / 255.0);
    cv::Mat mean(input_height_, input_width_, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std(input_height_, input_width_, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::subtract(f_img, mean, f_img);
    cv::divide(f_img, std, f_img);
    cv::Mat chw_img;
    cv::dnn::blobFromImage(f_img, chw_img);
    return chw_img;
}

// Inference run now uses member variables for sizing
int TensorRTInference::runInference(const std::string& imagePath) {
    if (!context_) {
        std::cerr << "Inference not set up properly." << std::endl;
        return -1;
    }

    cv::Mat preprocessed_img = preprocessImage(imagePath);
    if (preprocessed_img.empty()) {
        return -1;
    }
    
    const int input_size = preprocessed_img.total() * preprocessed_img.elemSize();
    cudaMemcpyAsync(device_buffers_[0], preprocessed_img.ptr<float>(), input_size, cudaMemcpyHostToDevice, stream_);

    context_->setTensorAddress(engine_->getIOTensorName(0), device_buffers_[0]);
    context_->setTensorAddress(engine_->getIOTensorName(1), device_buffers_[1]);
    context_->enqueueV3(stream_);

    std::vector<float> output_data(num_classes_);
    const int output_size = num_classes_ * sizeof(float);
    cudaMemcpyAsync(output_data.data(), device_buffers_[1], output_size, cudaMemcpyDeviceToHost, stream_);
    
    cudaStreamSynchronize(stream_);
    
    auto max_it = std::max_element(output_data.begin(), output_data.end());
    int predicted_class_index = std::distance(output_data.begin(), max_it);

    return predicted_class_index;
}