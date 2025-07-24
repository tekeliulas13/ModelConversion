#include "inference.h"
#include <fstream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <dlpack/dlpack.h>

TVMInference::TVMInference(const std::string& json_path, const std::string& params_path, const std::string& so_path) {
    // 1. Load the shared library
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(so_path);

    // 2. Load the graph definition
    std::ifstream json_in(json_path);
    std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
    json_in.close();

    // 3. Load the parameters
    std::ifstream params_in(params_path, std::ios::binary);
    std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
    params_in.close();

    // 4. Create the graph executor module
    graph_executor_module_ = mod_factory.GetFunction("default")(dev_);
    
    // 5. Get the functions from the module
    set_input_ = graph_executor_module_.GetFunction("set_input");
    get_output_ = graph_executor_module_.GetFunction("get_output");
    run_ = graph_executor_module_.GetFunction("run");
    tvm::runtime::PackedFunc load_params = graph_executor_module_.GetFunction("load_params");

    // 6. Load the parameters into the executor
    TVMByteArray params_arr;
    params_arr.data = params_data.c_str();
    params_arr.size = params_data.length();
    load_params(params_arr);
}

cv::Mat TVMInference::preprocess(const cv::Mat& img) {
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(input_width_, input_height_));
    resized_img.convertTo(resized_img, CV_32F, 1.0 / 255.0);

    // Normalize the image: (image - mean) / std
    cv::Mat mean(cv::Size(input_width_, input_height_), CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std(cv::Size(input_width_, input_height_), CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::subtract(resized_img, mean, resized_img);
    cv::divide(resized_img, std, resized_img);

    // Convert from HWC to NCHW format
    cv::Mat final_img;
    cv::dnn::blobFromImage(resized_img, final_img);
    return final_img;
}

int TVMInference::getTopPrediction(tvm::runtime::NDArray& output) {
    const float* data = static_cast<const float*>(output->data);
    auto max_it = std::max_element(data, data + num_classes_);
    return std::distance(data, max_it);
}

// ...existing code...
int TVMInference::runInference(const std::string& image_path) {
    // Load and preprocess image
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not read the image: " << image_path << std::endl;
        return -1;
    }
    cv::Mat input_image = preprocess(image);

    // Create input tensor
    tvm::runtime::NDArray input;
    tvm::runtime::ShapeTuple in_shape({1, 3, input_height_, input_width_});
    input = tvm::runtime::NDArray::Empty(in_shape, DLDataType{kDLFloat, 32, 1}, dev_);
    input.CopyFromBytes(input_image.ptr<float>(), input_image.total() * input_image.elemSize());

    // Set input, run the model
    set_input_("data", input);
    run_();

    // Get output
    tvm::runtime::NDArray output;
    tvm::runtime::ShapeTuple out_shape({1, num_classes_});
    output = tvm::runtime::NDArray::Empty(out_shape, DLDataType{kDLFloat, 32, 1}, dev_);
    get_output_(0, output);
    
    // Find and return the top prediction
    return getTopPrediction(output);
}