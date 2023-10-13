#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// Convert a 16-bit image to an 8-bit image and normalize it based on the provided max value
cv::Mat sixteenBits2EightBits(cv::Mat &image, double &max_value);

// Process a thermal image, preparing it for further analysis
cv::Mat processImageThermal(cv::Mat &image);

// Process an infrared image, converting it to a suitable format for further analysis
cv::Mat processImageIR(cv::Mat &image);

// Convert an OpenCV image to a PyTorch tensor and move it to the specified device
torch::Tensor toTensor(cv::Mat &image, torch::Device device);

// Convert a torch tensor to a vector of torch::jit::IValue for model input
std::vector<torch::jit::IValue> toInput(torch::Tensor &tensor_image);
