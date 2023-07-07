#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// image conversion (16bits -> 8bits)
cv::Mat sixteenBits2EightBits(cv::Mat &image, double &max_value);

// input processing module
cv::Mat processImageThermal(cv::Mat &image);
cv::Mat processImageIR(cv::Mat &image);

torch::Tensor toTensor(cv::Mat &image, torch::Device device);
std::vector<torch::jit::IValue> toInput(torch::Tensor &tensor_image);
